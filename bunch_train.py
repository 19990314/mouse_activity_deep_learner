#!/usr/bin/env python3
"""
train.py — Extended multi-session TCN trainer

Supports two input modes:
  1. Single-session (original):
       --dlc_h5 FILE --ann_csv FILE --combined_mat FILE --session_prefix PREFIX

  2. Multi-session via data directory:
       --data_dir FOLDER --combined_mat FILE
     The folder must contain a manifest.csv with columns:
       session_prefix, dlc_h5, ann_csv
     Paths in manifest.csv are relative to --data_dir.
     Optionally, each row can include a `mat_path` column to override --combined_mat.

     If no manifest.csv exists, the script auto-discovers sessions by scanning
     for paired files: *_dlc.h5 / *.h5 and *_annotations.csv / *_ann.csv.

Train/val split modes (--split_mode):
  - "temporal" (default): 80/20 time-split within each session, then pool.
  - "session": hold out ~20% of sessions entirely for validation.

Usage examples:
  # Single session (backward-compatible)
  python train.py --dlc_h5 data/sc04_d1_of.h5 --ann_csv data/sc04_d1_of_ann.csv \
    --combined_mat data/combined.mat --session_prefix sc04_d1_of --out_ckpt model.pt

  # Multi-session from folder
  python train.py --data_dir data/sessions/ --combined_mat data/combined.mat \
    --out_ckpt model.pt --split_mode temporal

  # Multi-session, hold-out session split
  python train.py --data_dir data/sessions/ --combined_mat data/combined.mat \
    --out_ckpt model.pt --split_mode session --val_fraction 0.2
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
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
# Model (unchanged)
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
        x = x.transpose(1, 2)       # (B, T, F) -> (B, F, T)
        x = self.in_proj(x)
        x = self.blocks(x)
        logits = self.out_proj(x)    # (B, K, T)
        return logits.transpose(1, 2)  # (B, T, K)

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
    """Read manifest.csv from data_dir. Expected columns: session_prefix, dlc_h5, ann_csv.
    Optional column: mat_path (overrides --combined_mat per session)."""
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
    """Heuristic: scan data_dir for paired .h5 and _ann.csv / _annotations.csv files.
    Derives session_prefix from the common filename stem."""
    data_dir = Path(data_dir)

    # Find all h5 files
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    sessions = []
    for h5 in h5_files:
        stem = h5.stem  # e.g. "sc04_d1_of_dlc" or "sc04_d1_of"

        # Strip common DLC suffixes to get session prefix
        prefix = re.sub(r"[_-]?dlc$", "", stem, flags=re.IGNORECASE)
        prefix = re.sub(r"[_-]?DLC_resnet.*$", "", prefix)

        # Search for matching annotation CSV
        ann_candidates = [
            data_dir / f"{prefix}_annotations.csv",
            data_dir / f"{prefix}_ann.csv",
            data_dir / f"{prefix}.csv",
        ]
        # Also try glob for partial matches
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
    """Try manifest.csv first, fall back to auto-discovery."""
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
    """Load and build features for every session. Returns list of dicts with X, y, mask, etc."""
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
            "X": X,
            "y": y,
            "mask": mask,
            "feat_names": feat_names,
            "fps": fps,
            "mat_name": mat_name,
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
# Train/val splitting strategies
# ============================================================

def split_temporal(
    session_data: List[Dict],
    val_fraction: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[Dataset, Dataset]:
    """80/20 temporal split within each session, then concatenate."""
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
    """Hold out entire sessions for validation."""
    n = len(session_data)
    n_val = max(1, int(round(val_fraction * n)))
    n_train = n - n_val

    if n_train < 1:
        raise ValueError(
            f"Not enough sessions ({n}) to hold out {n_val} for validation. "
            f"Use --split_mode temporal instead."
        )

    # Deterministic shuffle by session name for reproducibility
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
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Train TCN for mouse behavior classification (single or multi-session)."
    )

    # --- Input mode: single-session ---
    ap.add_argument("--dlc_h5", default=None,
                    help="Single-session DLC .h5 file.")
    ap.add_argument("--ann_csv", default=None,
                    help="Single-session annotation CSV.")
    ap.add_argument("--session_prefix", default=None,
                    help="Session prefix for MAT lookup (single-session mode).")

    # --- Input mode: multi-session ---
    ap.add_argument("--data_dir", default=None,
                    help="Directory containing session files + optional manifest.csv. "
                         "If provided, overrides --dlc_h5/--ann_csv/--session_prefix.")

    # --- Shared ---
    ap.add_argument("--combined_mat", default=None,
                    help="Path to combined .mat with kinematics (shared across sessions).")
    ap.add_argument("--label_col", default="human_labeled_state")
    ap.add_argument("--out_ckpt", required=True)

    # --- Split ---
    ap.add_argument("--split_mode", choices=["temporal", "session"], default="temporal",
                    help="'temporal': 80/20 time-split per session. "
                         "'session': hold out entire sessions for val.")
    ap.add_argument("--val_fraction", type=float, default=0.2,
                    help="Fraction of data (or sessions) for validation.")

    # --- Preprocessing ---
    ap.add_argument("--dlc_conf_thr", type=float, default=0.7)
    ap.add_argument("--smooth_label_win", type=int, default=9)

    # --- Training ---
    ap.add_argument("--seq_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)

    # --- Model ---
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--tv_weight", type=float, default=0.04)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # =============================================
    # Resolve input mode
    # =============================================
    if args.data_dir is not None:
        # --- Multi-session mode ---
        if args.combined_mat is None:
            # Check if there's a single .mat in data_dir
            mat_files = list(Path(args.data_dir).glob("*.mat"))
            if len(mat_files) == 1:
                args.combined_mat = str(mat_files[0])
                print(f"[Auto] Found single .mat in data_dir: {args.combined_mat}")
            else:
                raise ValueError(
                    "--combined_mat is required (or place exactly one .mat in --data_dir)."
                )

        sessions = load_sessions(args.data_dir)
        print(f"\n[Sessions] Found {len(sessions)} session(s):")
        for s in sessions:
            print(f"  - {s['session_prefix']}")

        session_data = load_all_sessions(
            sessions,
            combined_mat=args.combined_mat,
            label_col=args.label_col,
            dlc_conf_thr=args.dlc_conf_thr,
            smooth_label_win=args.smooth_label_win,
        )

    else:
        # --- Single-session mode (original) ---
        if not all([args.dlc_h5, args.ann_csv, args.combined_mat, args.session_prefix]):
            raise ValueError(
                "Single-session mode requires: --dlc_h5, --ann_csv, --combined_mat, --session_prefix. "
                "Alternatively, use --data_dir for multi-session mode."
            )

        dlc_df = load_dlc_h5(args.dlc_h5)
        ann_df = pd.read_csv(args.ann_csv)
        speed, w, fps, mat_name = load_kinematics_from_combined_mat(
            args.combined_mat, args.session_prefix
        )
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
            "feat_names": feat_names,
            "fps": fps, "mat_name": mat_name,
        }]

    # =============================================
    # Validate feature consistency across sessions
    # =============================================
    n_features = session_data[0]["X"].shape[1]
    feat_names = session_data[0]["feat_names"]
    for sess in session_data[1:]:
        if sess["X"].shape[1] != n_features:
            raise ValueError(
                f"Feature dimension mismatch: {sess['session_prefix']} has "
                f"{sess['X'].shape[1]} features, expected {n_features}. "
                f"All sessions must have the same DLC bodyparts and annotation columns."
            )

    # =============================================
    # Compute normalization stats from TRAIN data
    # =============================================
    if args.split_mode == "temporal":
        # Use first 80% of each session for norm stats
        train_chunks = []
        for sess in session_data:
            split = int((1.0 - args.val_fraction) * len(sess["X"]))
            train_chunks.append(sess["X"][:split])
        X_train_all = np.concatenate(train_chunks, axis=0)
    else:
        # Use all data from train sessions
        n = len(session_data)
        n_val = max(1, int(round(args.val_fraction * n)))
        indices = list(range(n))
        indices.sort(key=lambda i: session_data[i]["session_prefix"])
        train_idx = indices[:n - n_val]
        X_train_all = np.concatenate(
            [session_data[i]["X"] for i in train_idx], axis=0
        )

    norm_mean, norm_std = compute_norm_stats(X_train_all)
    del X_train_all  # free memory

    # =============================================
    # Build datasets
    # =============================================
    if args.split_mode == "temporal":
        train_ds, val_ds = split_temporal(
            session_data, args.val_fraction,
            norm_mean, norm_std,
            args.seq_len, args.stride,
        )
    else:
        if len(session_data) < 2:
            print("[WARN] Only 1 session loaded — falling back to temporal split.")
            train_ds, val_ds = split_temporal(
                session_data, args.val_fraction,
                norm_mean, norm_std,
                args.seq_len, args.stride,
            )
        else:
            train_ds, val_ds = split_by_session(
                session_data, args.val_fraction,
                norm_mean, norm_std,
                args.seq_len, args.stride,
            )

    print(f"\n[Dataset] Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # =============================================
    # Model
    # =============================================
    model = TCN(
        in_features=n_features,
        n_classes=6,
        channels=args.channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

    # =============================================
    # Training loop
    # =============================================
    best = float("inf")
    session_prefixes = [s["session_prefix"] for s in session_data]

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xb)
                loss = (masked_ce_loss(logits, yb, mb)
                        + temporal_tv_penalty(logits, weight=args.tv_weight))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        val_loss = eval_val()
        print(f"epoch {ep:03d} | val_loss={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "in_features": int(n_features),
                "channels": int(args.channels),
                "levels": int(args.levels),
                "kernel_size": int(args.kernel_size),
                "dropout": float(args.dropout),
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
            }
            torch.save(ckpt, args.out_ckpt)
            print(f"  [CKPT] saved best -> {args.out_ckpt}")

    print(f"\n[Done] best val_loss={best:.4f} | sessions={session_prefixes}")


if __name__ == "__main__":
    main()