#!/usr/bin/env python3
"""
train.py — Extended multi-session TCN trainer with hyperparameter search

New features vs previous version:
  - Early stopping (--patience, default 15 epochs)
  - Cosine annealing LR scheduler
  - Training curve logged to CSV (train_loss, val_loss, lr per epoch)
  - Hyperparameter sweep mode (--sweep): tries grid of configs, logs results

Usage examples:
  # Train single config
  python train.py --data_dir data/sessions/ --combined_mat data/combined.mat \
    --out_ckpt model.pt

  # Hyperparameter sweep (finds best config automatically)
  python train.py --data_dir data/sessions/ --combined_mat data/combined.mat \
    --out_ckpt model.pt --sweep
"""

import argparse
import csv
import glob
import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.io import loadmat

# ----------------------------
# State mapping
# ----------------------------
STATE_MAP = {
    0: "turn", 1: "forward", 2: "still",
    3: "explore", 4: "rear", 5: "groom",
    -1: "unsigned",
}
VALID_CLASSES = [0, 1, 2, 3, 4, 5]

# ============================================================
# Model
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return x + y


class TCN(nn.Module):
    def __init__(self, in_features, n_classes=6, channels=32, levels=6,
                 kernel_size=5, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, channels, kernel_size=1)
        blocks = []
        for i in range(levels):
            blocks.append(TCNBlock(channels, kernel_size=kernel_size,
                                   dilation=2**i, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(channels, n_classes, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.blocks(x)
        logits = self.out_proj(x)
        return logits.transpose(1, 2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# Data utilities
# ============================================================
def load_dlc_csv(dlc_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(dlc_csv_path, header=[0, 1, 2])
    if df.columns[0][0] == "scorer" and df.columns[0][1] == "bodyparts":
        df = df.drop(columns=df.columns[0])
    return df


def load_dlc_h5(path: str) -> pd.DataFrame:
    df = pd.read_hdf(path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DLC .h5 did not load with MultiIndex columns.")
    return df


def interpolate_low_conf(xy: np.ndarray, conf: np.ndarray, thr: float = 0.6) -> np.ndarray:
    out = xy.copy().astype(np.float32)
    out[conf < thr] = np.nan
    for j in range(2):
        s = pd.Series(out[:, j])
        out[:, j] = s.interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    return out


def compute_velocity(xy: np.ndarray) -> np.ndarray:
    v = np.zeros_like(xy, dtype=np.float32)
    v[1:] = xy[1:] - xy[:-1]
    return v


def window_majority_labels(y: np.ndarray, win: int = 7, ignore_label: int = -1) -> np.ndarray:
    assert win % 2 == 1
    r = win // 2
    T = len(y)
    y_out = y.copy()
    for t in range(T):
        a = max(0, t - r)
        b = min(T, t + r + 1)
        seg = y[a:b]
        seg = seg[seg != ignore_label]
        if len(seg) == 0:
            y_out[t] = ignore_label
        else:
            vals, counts = np.unique(seg, return_counts=True)
            y_out[t] = vals[np.argmax(counts)]
    return y_out


def load_kinematics_from_combined_mat(mat_path: str, session_prefix: str):
    m = loadmat(mat_path)
    allTrackData = m["allTrackData"]
    best_idx = None
    for i in range(allTrackData.shape[1]):
        nm = np.squeeze(allTrackData[0, i]["name"]).item()
        if isinstance(nm, str) and nm.startswith(session_prefix):
            best_idx = i
            break
    if best_idx is None:
        for i in range(allTrackData.shape[1]):
            nm = np.squeeze(allTrackData[0, i]["name"]).item()
            if isinstance(nm, str) and (session_prefix in nm):
                best_idx = i
                break
    if best_idx is None:
        names = [np.squeeze(allTrackData[0, i]["name"]).item()
                 for i in range(allTrackData.shape[1])]
        raise ValueError(
            f"Could not find session '{session_prefix}' in MAT. "
            f"Example names:\n" + "\n".join(map(str, names[:10]))
        )
    entry = allTrackData[0, best_idx]
    speed = np.squeeze(entry["speed_pixels_per_frame"]).astype(np.float32)
    w = np.squeeze(entry["w"]).astype(np.float32)
    fps = float(np.squeeze(entry["fps"]))
    name = np.squeeze(entry["name"]).item()
    return speed, w, fps, name


def build_feature_matrix(
    dlc_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    speed: np.ndarray,
    angvel: np.ndarray,
    label_col: str = "human_labeled_state",
    dlc_conf_thr: float = 0.7,
    smooth_label_win: int = 7,
    include_moseq: bool = True,
    include_latents: bool = True,
    include_centroid_heading: bool = True,
):
    T = min(len(dlc_df), len(ann_df), len(speed), len(angvel))
    dlc_df = dlc_df.iloc[:T]
    ann_df = ann_df.iloc[:T]
    speed = speed[:T]
    angvel = angvel[:T]

    bodyparts = sorted(set([c[1] for c in dlc_df.columns]))
    bodyparts = [bp for bp in bodyparts if bp not in ("bodyparts", "coords")]

    feats = []
    feat_names = []
    conf_list = []

    for bp in bodyparts:
        x = dlc_df.xs((bp, "x"), level=(1, 2), axis=1).to_numpy().squeeze()
        y = dlc_df.xs((bp, "y"), level=(1, 2), axis=1).to_numpy().squeeze()
        p = dlc_df.xs((bp, "likelihood"), level=(1, 2), axis=1).to_numpy().squeeze().astype(np.float32)

        xy = np.stack([x, y], axis=1).astype(np.float32)
        xy = interpolate_low_conf(xy, p, thr=dlc_conf_thr)
        v = compute_velocity(xy)

        feats.append(xy); feat_names += [f"{bp}_x", f"{bp}_y"]
        feats.append(v);  feat_names += [f"{bp}_vx", f"{bp}_vy"]
        conf_list.append(p.reshape(-1, 1))

    conf_mat = np.concatenate(conf_list, axis=1)
    feats.append(np.mean(conf_mat, axis=1, keepdims=True)); feat_names += ["p_mean"]
    feats.append(np.min(conf_mat, axis=1, keepdims=True));  feat_names += ["p_min"]

    if include_moseq and ("syllable" in ann_df.columns):
        feats.append(ann_df["syllable"].to_numpy(dtype=np.float32).reshape(-1, 1))
        feat_names += ["syllable_id"]

    if include_latents:
        for j in range(4):
            col = f"latent_state {j}"
            if col in ann_df.columns:
                feats.append(ann_df[col].to_numpy(dtype=np.float32).reshape(-1, 1))
                feat_names += [col]

    if include_centroid_heading:
        for col in ["centroid x", "centroid y", "heading"]:
            if col in ann_df.columns:
                feats.append(ann_df[col].to_numpy(dtype=np.float32).reshape(-1, 1))
                feat_names += [col]

    feats.append(speed.reshape(-1, 1));  feat_names += ["speed_pixels_per_frame"]
    feats.append(angvel.reshape(-1, 1)); feat_names += ["w_angvel"]

    X = np.concatenate(feats, axis=1).astype(np.float32)

    y_arr = ann_df[label_col].to_numpy(dtype=np.int64)[:T]
    if smooth_label_win is not None and smooth_label_win >= 3:
        y_arr = window_majority_labels(y_arr, win=smooth_label_win, ignore_label=-1)

    mask = np.isin(y_arr, VALID_CLASSES)
    return X, y_arr, mask, feat_names


def compute_norm_stats(X: np.ndarray, eps: float = 1e-6):
    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    std = X.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std

# ============================================================
# Multi-session discovery & loading
# ============================================================

def discover_sessions_from_manifest(data_dir: str) -> List[Dict]:
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No manifest.csv found in {data_dir}")

    df = pd.read_csv(manifest_path)
    required = {"session_prefix", "dlc_h5", "ann_csv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {missing}")

    sessions = []
    for _, row in df.iterrows():
        entry = {
            "session_prefix": row["session_prefix"],
            "dlc_h5": os.path.join(data_dir, row["dlc_h5"]),
            "ann_csv": os.path.join(data_dir, row["ann_csv"]),
        }
        if "mat_path" in df.columns and pd.notna(row.get("mat_path")):
            entry["mat_path"] = os.path.join(data_dir, row["mat_path"])
        sessions.append(entry)
    return sessions


def auto_discover_sessions(data_dir: str) -> List[Dict]:
    data_dir = Path(data_dir)
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    sessions = []
    for h5 in h5_files:
        stem = h5.stem
        prefix = re.sub(r"[_-]?dlc$", "", stem, flags=re.IGNORECASE)
        prefix = re.sub(r"[_-]?DLC_resnet.*$", "", prefix)

        ann_candidates = [
            data_dir / f"{prefix}_annotations.csv",
            data_dir / f"{prefix}_ann.csv",
            data_dir / f"{prefix}.csv",
        ]
        ann_glob = sorted(data_dir.glob(f"{prefix}*ann*.csv"))
        ann_candidates.extend(ann_glob)

        ann_csv = None
        for candidate in ann_candidates:
            if candidate.is_file():
                ann_csv = str(candidate)
                break

        if ann_csv is None:
            print(f"[WARN] No annotation CSV found for {h5.name} (prefix={prefix}), skipping.")
            continue

        sessions.append({
            "session_prefix": prefix,
            "dlc_h5": str(h5),
            "ann_csv": ann_csv,
        })

    if not sessions:
        raise FileNotFoundError(
            f"Could not auto-discover any valid session pairs in {data_dir}. "
            f"Please provide a manifest.csv instead."
        )
    return sessions


def load_sessions(data_dir: str) -> List[Dict]:
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if os.path.isfile(manifest_path):
        print(f"[Sessions] Loading from manifest: {manifest_path}")
        return discover_sessions_from_manifest(data_dir)
    else:
        print(f"[Sessions] No manifest.csv found, auto-discovering from {data_dir}")
        return auto_discover_sessions(data_dir)


def load_all_sessions(
    sessions: List[Dict],
    combined_mat: str,
    label_col: str,
    dlc_conf_thr: float,
    smooth_label_win: int,
) -> List[Dict]:
    loaded = []
    for sess in sessions:
        prefix = sess["session_prefix"]
        mat_path = sess.get("mat_path", combined_mat)

        print(f"\n[Loading] session={prefix}")
        print(f"  dlc_h5:  {sess['dlc_h5']}")
        print(f"  ann_csv: {sess['ann_csv']}")
        print(f"  mat:     {mat_path}")

        try:
            dlc_df = load_dlc_h5(sess["dlc_h5"])
        except Exception as e:
            print(f"  [ERROR] Failed to load DLC h5: {e}")
            continue
        try:
            ann_df = pd.read_csv(sess["ann_csv"])
        except Exception as e:
            print(f"  [ERROR] Failed to load annotation CSV: {e}")
            continue
        try:
            speed, w, fps, mat_name = load_kinematics_from_combined_mat(mat_path, prefix)
        except Exception as e:
            print(f"  [ERROR] Failed to load kinematics: {e}")
            continue

        X, y, mask, feat_names = build_feature_matrix(
            dlc_df, ann_df, speed, w,
            label_col=label_col,
            dlc_conf_thr=dlc_conf_thr,
            smooth_label_win=smooth_label_win,
        )
        print(f"  T={len(X)}, F={X.shape[1]}, valid_frames={mask.sum()}, fps={fps}")
        print(f"  Label counts: {dict(zip(*np.unique(y[mask], return_counts=True)))}")

        loaded.append({
            "session_prefix": prefix,
            "X": X, "y": y, "mask": mask,
            "feat_names": feat_names,
            "fps": fps, "mat_name": mat_name,
        })

    if not loaded:
        raise RuntimeError("No sessions were successfully loaded.")
    return loaded


# ============================================================
# Dataset
# ============================================================
class SequenceDataset(Dataset):
    def __init__(self, X, y, mask, seq_len=256, stride=128):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mask = mask.astype(bool)
        self.seq_len = int(seq_len)
        self.indices = []
        T = len(X)
        for start in range(0, T - self.seq_len + 1, stride):
            end = start + self.seq_len
            if self.mask[start:end].any():
                self.indices.append((start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        a, b = self.indices[i]
        return (
            torch.from_numpy(self.X[a:b]),
            torch.from_numpy(self.y[a:b]),
            torch.from_numpy(self.mask[a:b]),
        )


def masked_ce_loss(logits, y, mask):
    B, T, K = logits.shape
    logits2 = logits.reshape(B * T, K)
    y2 = y.reshape(B * T)
    m2 = mask.reshape(B * T)
    if m2.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits2[m2], y2[m2])


def temporal_tv_penalty(logits, weight=0.02):
    diff = logits[:, 1:] - logits[:, :-1]
    return weight * diff.abs().mean()


# ============================================================
# Train/val splitting
# ============================================================

def split_temporal(
    session_data: List[Dict],
    val_fraction: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[Dataset, Dataset]:
    train_datasets = []
    val_datasets = []

    for sess in session_data:
        X = (sess["X"] - norm_mean) / norm_std
        y, mask = sess["y"], sess["mask"]
        T = len(X)
        split = int((1.0 - val_fraction) * T)

        train_datasets.append(
            SequenceDataset(X[:split], y[:split], mask[:split],
                            seq_len=seq_len, stride=stride)
        )
        val_datasets.append(
            SequenceDataset(X[split:], y[split:], mask[split:],
                            seq_len=seq_len, stride=stride)
        )

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def split_by_session(
    session_data: List[Dict],
    val_fraction: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[Dataset, Dataset]:
    n = len(session_data)
    n_val = max(1, int(round(val_fraction * n)))
    n_train = n - n_val

    if n_train < 1:
        raise ValueError(
            f"Not enough sessions ({n}) to hold out {n_val} for validation. "
            f"Use --split_mode temporal instead."
        )

    indices = list(range(n))
    indices.sort(key=lambda i: session_data[i]["session_prefix"])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"[Split] Train sessions ({len(train_idx)}): "
          f"{[session_data[i]['session_prefix'] for i in train_idx]}")
    print(f"[Split] Val sessions ({len(val_idx)}):   "
          f"{[session_data[i]['session_prefix'] for i in val_idx]}")

    train_datasets = []
    val_datasets = []

    for i in train_idx:
        X = (session_data[i]["X"] - norm_mean) / norm_std
        train_datasets.append(
            SequenceDataset(X, session_data[i]["y"], session_data[i]["mask"],
                            seq_len=seq_len, stride=stride)
        )
    for i in val_idx:
        X = (session_data[i]["X"] - norm_mean) / norm_std
        val_datasets.append(
            SequenceDataset(X, session_data[i]["y"], session_data[i]["mask"],
                            seq_len=seq_len, stride=stride)
        )

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


# ============================================================
# Single training run (reusable by sweep)
# ============================================================

def train_one_config(
    session_data: List[Dict],
    n_features: int,
    feat_names: List[str],
    args,
    # Overrides for sweep
    channels: int = None,
    levels: int = None,
    kernel_size: int = None,
    dropout: float = None,
    lr: float = None,
    tv_weight: float = None,
    seq_len: int = None,
    stride: int = None,
    out_ckpt: str = None,
    verbose: bool = True,
) -> Dict:
    """
    Train one model configuration. Returns dict with:
      best_val_loss, best_epoch, total_epochs, n_params, config, train_curve
    """
    channels = channels or args.channels
    levels = levels or args.levels
    kernel_size = kernel_size or args.kernel_size
    dropout = dropout if dropout is not None else args.dropout
    lr = lr or args.lr
    tv_weight = tv_weight if tv_weight is not None else args.tv_weight
    seq_len = seq_len or args.seq_len
    stride = stride or args.stride
    out_ckpt = out_ckpt or args.out_ckpt

    config = {
        "channels": channels, "levels": levels, "kernel_size": kernel_size,
        "dropout": dropout, "lr": lr, "tv_weight": tv_weight,
        "seq_len": seq_len, "stride": stride,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Normalization stats (from train partition only) ---
    if args.split_mode == "temporal":
        train_chunks = []
        for sess in session_data:
            split = int((1.0 - args.val_fraction) * len(sess["X"]))
            train_chunks.append(sess["X"][:split])
        X_train_all = np.concatenate(train_chunks, axis=0)
    else:
        n = len(session_data)
        n_val = max(1, int(round(args.val_fraction * n)))
        indices = list(range(n))
        indices.sort(key=lambda i: session_data[i]["session_prefix"])
        train_idx = indices[:n - n_val]
        X_train_all = np.concatenate(
            [session_data[i]["X"] for i in train_idx], axis=0
        )
    norm_mean, norm_std = compute_norm_stats(X_train_all)
    del X_train_all

    # --- Build datasets ---
    if args.split_mode == "temporal":
        train_ds, val_ds = split_temporal(
            session_data, args.val_fraction,
            norm_mean, norm_std, seq_len, stride,
        )
    else:
        if len(session_data) < 2:
            train_ds, val_ds = split_temporal(
                session_data, args.val_fraction,
                norm_mean, norm_std, seq_len, stride,
            )
        else:
            train_ds, val_ds = split_by_session(
                session_data, args.val_fraction,
                norm_mean, norm_std, seq_len, stride,
            )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = TCN(
        in_features=n_features, n_classes=6,
        channels=channels, levels=levels,
        kernel_size=kernel_size, dropout=dropout,
    ).to(device)

    n_params = model.count_parameters()
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Config] ch={channels} lv={levels} ks={kernel_size} "
              f"do={dropout} lr={lr} tv={tv_weight} seq={seq_len} str={stride}")
        print(f"[Model]  {n_params:,} parameters")
        print(f"[Data]   Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
        print(f"{'='*60}")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    @torch.no_grad()
    def eval_val():
        model.eval()
        losses = []
        for xb, yb, mb in val_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            logits = model(xb)
            loss = masked_ce_loss(logits, yb, mb)
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    # --- Training loop ---
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_curve = []
    session_prefixes = [s["session_prefix"] for s in session_data]

    for ep in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xb)
                loss = (masked_ce_loss(logits, yb, mb)
                        + temporal_tv_penalty(logits, weight=tv_weight))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_losses.append(loss.item())

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_loss = eval_val()
        train_loss = float(np.mean(epoch_losses))
        train_curve.append((ep, train_loss, val_loss, current_lr))

        improved = val_loss < best_val
        if verbose:
            marker = " *" if improved else ""
            print(f"epoch {ep:03d} | train={train_loss:.4f} val={val_loss:.4f} "
                  f"lr={current_lr:.2e}{marker}")

        if improved:
            best_val = val_loss
            best_epoch = ep
            patience_counter = 0

            ckpt = {
                "state_dict": model.state_dict(),
                "in_features": int(n_features),
                "channels": int(channels),
                "levels": int(levels),
                "kernel_size": int(kernel_size),
                "dropout": float(dropout),
                "feature_names": feat_names,
                "norm_mean": norm_mean.astype(np.float32),
                "norm_std": norm_std.astype(np.float32),
                "label_col": args.label_col,
                "session_prefixes": session_prefixes,
                "n_sessions": len(session_data),
                "split_mode": args.split_mode,
                "fps": float(session_data[0]["fps"]),
                "smooth_label_win": int(args.smooth_label_win),
                "dlc_conf_thr": float(args.dlc_conf_thr),
                "best_val_loss": float(best_val),
                "best_epoch": int(best_epoch),
                "config": config,
            }
            torch.save(ckpt, out_ckpt)
            if verbose:
                print(f"  [CKPT] saved -> {out_ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                if verbose:
                    print(f"[Early stop] No improvement for {args.patience} epochs. "
                          f"Best: epoch {best_epoch}, val_loss={best_val:.4f}")
                break

    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "total_epochs": ep,
        "n_params": n_params,
        "config": config,
        "train_curve": train_curve,
        "out_ckpt": out_ckpt,
    }


# ============================================================
# Hyperparameter sweep
# ============================================================

# Search grid — targeted for small-data regime (3-5 sessions)
SWEEP_GRID = {
    "channels":    [32, 64, 96],
    "levels":      [4, 6, 8],
    "dropout":     [0.15, 0.25, 0.35],
    "lr":          [1e-4, 3e-4],
    "tv_weight":   [0.04, 0.08],
    "kernel_size": [5],
    "seq_len":     [384],
    "stride":      [256],
}
# Total: 3 × 3 × 3 × 2 × 2 = 108 configs


def generate_sweep_configs(grid: Dict) -> List[Dict]:
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, combo)) for combo in combos]


def run_sweep(session_data, n_features, feat_names, args):
    configs = generate_sweep_configs(SWEEP_GRID)
    n_configs = len(configs)

    out_dir = Path(args.out_ckpt).parent
    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_log_path = sweep_dir / "sweep_results.csv"
    print(f"\n{'#'*60}")
    print(f"[SWEEP] {n_configs} configurations to evaluate")
    print(f"[SWEEP] Results -> {sweep_log_path}")
    print(f"{'#'*60}")

    results = []

    for i, cfg in enumerate(configs, 1):
        tag = (f"ch{cfg['channels']}_lv{cfg['levels']}_ks{cfg['kernel_size']}"
               f"_do{cfg['dropout']}_lr{cfg['lr']}_tv{cfg['tv_weight']}")
        ckpt_path = str(sweep_dir / f"sweep_{tag}.pt")

        print(f"\n[SWEEP {i}/{n_configs}] {tag}")
        t0 = time.time()

        try:
            result = train_one_config(
                session_data, n_features, feat_names, args,
                channels=cfg["channels"], levels=cfg["levels"],
                kernel_size=cfg["kernel_size"], dropout=cfg["dropout"],
                lr=cfg["lr"], tv_weight=cfg["tv_weight"],
                seq_len=cfg["seq_len"], stride=cfg["stride"],
                out_ckpt=ckpt_path, verbose=True,
            )
            elapsed = time.time() - t0
            result["tag"] = tag
            result["elapsed_s"] = elapsed
            results.append(result)

            print(f"[SWEEP {i}/{n_configs}] best_val={result['best_val_loss']:.4f} "
                  f"@ ep {result['best_epoch']} | {result['n_params']:,} params | {elapsed:.0f}s")

        except Exception as e:
            print(f"[SWEEP {i}/{n_configs}] FAILED: {e}")
            results.append({
                "tag": tag, "config": cfg,
                "best_val_loss": float("inf"),
                "best_epoch": 0, "total_epochs": 0,
                "n_params": 0, "elapsed_s": 0, "error": str(e),
            })

    # --- Write sweep CSV ---
    with open(sweep_log_path, "w", newline="") as f:
        fieldnames = [
            "rank", "tag", "best_val_loss", "best_epoch", "total_epochs",
            "n_params", "elapsed_s",
            "channels", "levels", "kernel_size", "dropout", "lr", "tv_weight",
            "seq_len", "stride",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        sorted_results = sorted(results, key=lambda r: r["best_val_loss"])
        for rank, r in enumerate(sorted_results, 1):
            cfg = r.get("config", {})
            writer.writerow({
                "rank": rank, "tag": r.get("tag", ""),
                "best_val_loss": f"{r['best_val_loss']:.6f}",
                "best_epoch": r.get("best_epoch", 0),
                "total_epochs": r.get("total_epochs", 0),
                "n_params": r.get("n_params", 0),
                "elapsed_s": f"{r.get('elapsed_s', 0):.1f}",
                "channels": cfg.get("channels", ""),
                "levels": cfg.get("levels", ""),
                "kernel_size": cfg.get("kernel_size", ""),
                "dropout": cfg.get("dropout", ""),
                "lr": cfg.get("lr", ""),
                "tv_weight": cfg.get("tv_weight", ""),
                "seq_len": cfg.get("seq_len", ""),
                "stride": cfg.get("stride", ""),
            })

    # --- Copy best checkpoint ---
    valid = [r for r in results if r["best_val_loss"] < float("inf")]
    if valid:
        best = min(valid, key=lambda r: r["best_val_loss"])
        best_ckpt = best.get("out_ckpt")
        if best_ckpt and os.path.isfile(best_ckpt):
            import shutil
            shutil.copy2(best_ckpt, args.out_ckpt)
            print(f"\n{'='*60}")
            print(f"[SWEEP DONE] Best config: {best['tag']}")
            print(f"  val_loss = {best['best_val_loss']:.4f} @ epoch {best['best_epoch']}")
            print(f"  params   = {best['n_params']:,}")
            print(f"  config   = {best['config']}")
            print(f"  Saved to: {args.out_ckpt}")
            print(f"  All results: {sweep_log_path}")
            print(f"{'='*60}")

        # Save top-5 training curves
        top5 = sorted(valid, key=lambda r: r["best_val_loss"])[:5]
        for r in top5:
            curve = r.get("train_curve", [])
            if curve:
                curve_path = sweep_dir / f"curve_{r['tag']}.csv"
                with open(curve_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epoch", "train_loss", "val_loss", "lr"])
                    for row in curve:
                        w.writerow(row)
    else:
        print("[SWEEP] All configurations failed!")

    return results


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Train TCN for mouse behavior classification. "
                    "Use --sweep for automatic hyperparameter search."
    )

    # Input
    ap.add_argument("--dlc_h5", default=None)
    ap.add_argument("--ann_csv", default=None)
    ap.add_argument("--session_prefix", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--combined_mat", default=None)
    ap.add_argument("--label_col", default="human_labeled_state")
    ap.add_argument("--out_ckpt", required=True)

    # Split
    ap.add_argument("--split_mode", choices=["temporal", "session"], default="temporal")
    ap.add_argument("--val_fraction", type=float, default=0.2)

    # Preprocessing
    ap.add_argument("--dlc_conf_thr", type=float, default=0.7)
    ap.add_argument("--smooth_label_win", type=int, default=9)

    # Training
    ap.add_argument("--seq_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping: stop after N epochs with no val improvement.")

    # Model
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--tv_weight", type=float, default=0.04)

    # Sweep
    ap.add_argument("--sweep", action="store_true",
                    help="Run hyperparameter sweep over predefined grid.")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # =============================================
    # Load data
    # =============================================
    if args.data_dir is not None:
        if args.combined_mat is None:
            mat_files = list(Path(args.data_dir).glob("*.mat"))
            if len(mat_files) == 1:
                args.combined_mat = str(mat_files[0])
                print(f"[Auto] Found single .mat: {args.combined_mat}")
            else:
                raise ValueError("--combined_mat required (or place one .mat in --data_dir).")

        sessions = load_sessions(args.data_dir)
        print(f"\n[Sessions] Found {len(sessions)} session(s):")
        for s in sessions:
            print(f"  - {s['session_prefix']}")

        session_data = load_all_sessions(
            sessions, combined_mat=args.combined_mat,
            label_col=args.label_col,
            dlc_conf_thr=args.dlc_conf_thr,
            smooth_label_win=args.smooth_label_win,
        )
    else:
        if not all([args.dlc_h5, args.ann_csv, args.combined_mat, args.session_prefix]):
            raise ValueError(
                "Single-session mode requires: --dlc_h5, --ann_csv, --combined_mat, --session_prefix. "
                "Or use --data_dir for multi-session."
            )
        dlc_df = load_dlc_h5(args.dlc_h5)
        ann_df = pd.read_csv(args.ann_csv)
        speed, w, fps, mat_name = load_kinematics_from_combined_mat(
            args.combined_mat, args.session_prefix)
        print(f"[MAT] matched: {mat_name} | fps={fps}")

        X, y, mask, feat_names = build_feature_matrix(
            dlc_df, ann_df, speed, w,
            label_col=args.label_col,
            dlc_conf_thr=args.dlc_conf_thr,
            smooth_label_win=args.smooth_label_win,
        )
        print(f"[Data] T={len(X)} F={X.shape[1]} valid_frames={mask.sum()}")
        session_data = [{
            "session_prefix": args.session_prefix,
            "X": X, "y": y, "mask": mask,
            "feat_names": feat_names, "fps": fps, "mat_name": mat_name,
        }]

    # Validate features
    n_features = session_data[0]["X"].shape[1]
    feat_names = session_data[0]["feat_names"]
    for sess in session_data[1:]:
        if sess["X"].shape[1] != n_features:
            raise ValueError(
                f"Feature mismatch: {sess['session_prefix']} has "
                f"{sess['X'].shape[1]} features, expected {n_features}."
            )

    # =============================================
    # Train or sweep
    # =============================================
    if args.sweep:
        run_sweep(session_data, n_features, feat_names, args)
    else:
        result = train_one_config(
            session_data, n_features, feat_names, args, verbose=True,
        )
        print(f"\n[Done] best val_loss={result['best_val_loss']:.4f} "
              f"@ epoch {result['best_epoch']}/{result['total_epochs']} "
              f"| {result['n_params']:,} params")

        # Save training curve
        curve_path = Path(args.out_ckpt).with_suffix(".curve.csv")
        with open(curve_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "lr"])
            for row in result["train_curve"]:
                w.writerow(row)
        print(f"[Curve] {curve_path}")


if __name__ == "__main__":
    main()