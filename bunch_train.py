#!/usr/bin/env python3
"""
train.py — Multi-session TCN trainer v3 (egocentric features)

Key changes vs v2:
  - EGOCENTRIC FEATURES: bodypart positions are now relative to the body
    centroid, eliminating dependence on absolute arena position. This is
    critical for cross-session generalization.
  - POSTURE FEATURES: inter-bodypart distances (body length, head width)
    and body angle encode posture in a position/rotation-invariant way.
  - CENTROID VELOCITY: captures overall locomotion direction.
  - TRAINING AUGMENTATION: optional Gaussian noise (--augment) to further
    reduce overfitting to session-specific patterns.
  - Absolute centroid x/y from MoSeq is EXCLUDED (position-dependent).

Feature set per bodypart (4 features each):
  {bp}_rx, {bp}_ry    — position relative to body centroid
  {bp}_vx, {bp}_vy    — frame-to-frame velocity

Global features:
  centroid_vx/vy       — centroid velocity (locomotion direction)
  p_mean, p_min        — DLC confidence
  dist_*               — inter-bodypart distances (posture)
  body_angle_sin/cos   — nose-to-tail orientation
  syllable_id          — MoSeq syllable
  latent_state 0-3     — MoSeq latents
  heading              — MoSeq heading (position-invariant)
  speed_cm_per_s       — from allTrackData
  w_angvel             — angular velocity

Usage:
  python train.py --data_dir data/ --combined_mat data/combined.mat \
    --out_ckpt model.pt --split_mode loso --augment
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
N_CLASSES = len(VALID_CLASSES)

# ============================================================
# Model
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.3):
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
    def __init__(self, in_features, n_classes=6, channels=32, levels=8,
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


def enforce_stillness_by_speed(y: np.ndarray, speed: np.ndarray,
                               threshold: float = 0.2,
                               min_duration: int = 30,
                               still_label: int = 2) -> np.ndarray:
    """
    Overrides human_labeled_state to 'still' (default 2) ONLY if the speed
    remains below the threshold for a continuous sequence longer than min_duration.

    Args:
        y: The state label array.
        speed: The speed array (cm/s).
        threshold: Speed cutoff (default 0.1).
        min_duration: Minimum consecutive frames required to trigger the override (default 30).
        still_label: The integer label for 'still' (default 2).
    """
    # 1. Ensure lengths match
    n = min(len(y), len(speed))
    y = y[:n]
    speed = speed[:n]

    # 2. create a boolean mask where speed is low
    low_speed_mask = (speed < threshold)

    # 3. Find runs (start and end indices) of low speed
    # Padding with False ensures we detect runs at the very start or end of the array
    padded = np.concatenate(([False], low_speed_mask, [False]))
    # diff gives 1 at start of run, -1 at end of run
    diffs = np.diff(padded.astype(int))

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    frames_modified = 0

    # 4. Iterate through runs and apply correction if length > min_duration
    for start, end in zip(starts, ends):
        run_length = end - start
        if run_length > min_duration:
            # Count how many we are actually changing (for logging)
            current_labels = y[start:end]
            frames_modified += np.sum(current_labels != still_label)

            # Force to still
            y[start:end] = still_label

    if frames_modified > 0:
        print(f"  [LABEL] Enforced stillness (speed < {threshold} for > {min_duration} frames) "
              f"on {frames_modified} frames.")

    return y




# ============================================================
# Loss functions
# ============================================================

def compute_class_weights(session_data: List[Dict], indices: List[int] = None,
                          n_classes: int = N_CLASSES) -> torch.Tensor:
    """
    Compute class weights using sqrt of inverse frequency.

    Raw inverse-frequency weights over-compensate for rare classes (like
    'still' with ~400 frames), causing massive over-prediction. Sqrt
    scaling provides a gentler boost:
      raw inverse:  still ≈ 8x, explore ≈ 0.5x  → too aggressive
      sqrt inverse: still ≈ 2.8x, explore ≈ 0.7x → balanced
    """
    if indices is None:
        indices = list(range(len(session_data)))
    counts = np.zeros(n_classes, dtype=np.float64)
    for i in indices:
        y = session_data[i]["y"]
        mask = session_data[i]["mask"]
        for c in range(n_classes):
            counts[c] += ((y == c) & mask).sum()
    total = counts.sum()
    # sqrt of inverse frequency, normalized to sum to n_classes
    raw_weights = np.where(counts > 0, total / (n_classes * counts), 1.0)
    weights = np.sqrt(raw_weights)
    # Re-normalize so weights sum to n_classes (preserves loss scale)
    weights = weights * (n_classes / weights.sum())
    # Clamp to prevent any extreme values
    weights = np.clip(weights, 0.5, 3.0)
    return torch.tensor(weights, dtype=torch.float32)


def masked_ce_loss(logits, y, mask, class_weights=None):
    B, T, K = logits.shape
    logits2 = logits.reshape(B * T, K)
    y2 = y.reshape(B * T)
    m2 = mask.reshape(B * T)
    if m2.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits2[m2], y2[m2], weight=class_weights)


def masked_focal_loss(logits, y, mask, class_weights=None, gamma=2.0):
    B, T, K = logits.shape
    logits2 = logits.reshape(B * T, K)
    y2 = y.reshape(B * T)
    m2 = mask.reshape(B * T)
    if m2.sum() == 0:
        return logits.sum() * 0.0
    logits_m = logits2[m2]
    y_m = y2[m2]
    ce = F.cross_entropy(logits_m, y_m, weight=class_weights, reduction='none')
    probs = F.softmax(logits_m, dim=1)
    p_t = probs.gather(1, y_m.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce).mean()


def temporal_tv_penalty(logits, weight=0.02):
    diff = logits[:, 1:] - logits[:, :-1]
    return weight * diff.abs().mean()


# ============================================================
# Data utilities
# ============================================================
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
            f"Example names:\n" + "\n".join(map(str, names[:10])))
    entry = allTrackData[0, best_idx]
    speed = np.squeeze(entry["speed_cm_per_s_origin"]).astype(np.float32)
    w = np.squeeze(entry["w"]).astype(np.float32)
    fps = float(np.squeeze(entry["fps"]))
    name = np.squeeze(entry["name"]).item()

    if len(speed) != len(w):
        min_len = min(len(speed), len(w))
        print(f"  [KIN] Length mismatch: speed={len(speed)}, w={len(w)} — "
              f"truncating both to {min_len}")
        speed = speed[:min_len]
        w = w[:min_len]

    kin_valid = np.isfinite(speed) & np.isfinite(w)
    n_nan = int((~kin_valid).sum())
    if n_nan > 0:
        print(f"  [KIN] {n_nan}/{len(speed)} frames have NaN kinematics — will be masked out")
        speed = np.where(kin_valid, speed, 0.0).astype(np.float32)
        w = np.where(kin_valid, w, 0.0).astype(np.float32)

    return speed, w, kin_valid, fps, name


def _find_bodypart(bodyparts: List[str], candidates: List[str]) -> Optional[str]:
    """Find a bodypart name from a list of possible names."""
    for c in candidates:
        if c in bodyparts:
            return c
        # Try case-insensitive
        for bp in bodyparts:
            if bp.lower() == c.lower():
                return bp
    return None


def build_feature_matrix(
    dlc_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    speed: np.ndarray,
    angvel: np.ndarray,
    kin_valid: np.ndarray,
    label_col: str = "human_labeled_state",
    dlc_conf_thr: float = 0.7,
    smooth_label_win: int = 7,
    include_moseq: bool = True,
    include_latents: bool = True,
    include_centroid_heading: bool = True,
):
    """
    Build EGOCENTRIC feature matrix.

    All bodypart positions are relative to the body centroid (mean of all
    tracked bodyparts). This eliminates dependence on absolute arena
    position, which is the primary source of cross-session generalization
    failure.

    Additional posture features:
      - Inter-bodypart distances (body length, head width, etc.)
      - Body orientation angle (nose-to-tail direction, sin/cos encoded)
      - Centroid velocity (locomotion direction)
    """
    T = min(len(dlc_df), len(ann_df), len(speed), len(angvel), len(kin_valid))
    dlc_df = dlc_df.iloc[:T]
    ann_df = ann_df.iloc[:T]
    speed = speed[:T]
    angvel = angvel[:T]
    kin_valid = kin_valid[:T]

    bodyparts = sorted(set([c[1] for c in dlc_df.columns]))
    bodyparts = [bp for bp in bodyparts if bp not in ("bodyparts", "coords")]

    # =========================================================
    # Step 1: Extract all bodypart positions & confidences
    # =========================================================
    bp_xy = {}      # {name: (T, 2) float32}
    bp_conf = {}    # {name: (T,) float32}

    for bp in bodyparts:
        x = dlc_df.xs((bp, "x"), level=(1, 2), axis=1).to_numpy().squeeze()
        y = dlc_df.xs((bp, "y"), level=(1, 2), axis=1).to_numpy().squeeze()
        p = dlc_df.xs((bp, "likelihood"), level=(1, 2), axis=1).to_numpy().squeeze().astype(np.float32)
        xy = np.stack([x, y], axis=1).astype(np.float32)
        xy = interpolate_low_conf(xy, p, thr=dlc_conf_thr)
        bp_xy[bp] = xy
        bp_conf[bp] = p

    # =========================================================
    # Step 2: Body reference point = "neck" bodypart
    # =========================================================
    bp_neck = _find_bodypart(bodyparts, ["neck", "Neck", "neck_base"])
    if bp_neck is None:
        print(f"  [FEAT WARN] 'neck' not found in bodyparts: {bodyparts}")
        print(f"  [FEAT WARN] Falling back to mean of all bodyparts as centroid")
        all_xy = np.stack(list(bp_xy.values()), axis=1)  # (T, N_bp, 2)
        centroid = np.nanmean(all_xy, axis=1)  # (T, 2)
    else:
        centroid = bp_xy[bp_neck].copy()  # (T, 2)
        print(f"  [FEAT] Using '{bp_neck}' as body reference point")

    # Fill any remaining NaN in centroid (edge case)
    for j in range(2):
        s = pd.Series(centroid[:, j])
        centroid[:, j] = s.interpolate(limit_direction="both").to_numpy(dtype=np.float32)

    # =========================================================
    # Step 3: Egocentric features per bodypart
    # =========================================================
    feats = []
    feat_names = []

    for bp in bodyparts:
        xy = bp_xy[bp]
        # Relative position (centroid-subtracted)
        rel_xy = (xy - centroid).astype(np.float32)
        # Velocity (already shift-invariant)
        vel = compute_velocity(xy)

        feats.append(rel_xy)
        feat_names += [f"{bp}_rx", f"{bp}_ry"]
        feats.append(vel)
        feat_names += [f"{bp}_vx", f"{bp}_vy"]

    # =========================================================
    # Step 4: Centroid velocity (locomotion direction & speed)
    # =========================================================
    centroid_vel = compute_velocity(centroid)
    feats.append(centroid_vel)
    feat_names += ["centroid_vx", "centroid_vy"]

    # =========================================================
    # Step 5: Confidence features
    # =========================================================
    conf_mat = np.stack(list(bp_conf.values()), axis=1)  # (T, N_bp)
    feats.append(np.mean(conf_mat, axis=1, keepdims=True))
    feat_names += ["p_mean"]
    feats.append(np.min(conf_mat, axis=1, keepdims=True))
    feat_names += ["p_min"]

    # =========================================================
    # Step 6: Inter-bodypart distances (posture features)
    # =========================================================
    # Try common DLC bodypart name variants
    bp_nose = _find_bodypart(bodyparts, ["nose", "snout", "Nose"])
    bp_tail = _find_bodypart(bodyparts, ["tail_base", "tailbase", "tail_start", "Tail_base"])
    bp_left_ear = _find_bodypart(bodyparts, ["left_ear", "leftear", "Left_ear", "ear_left"])
    bp_right_ear = _find_bodypart(bodyparts, ["right_ear", "rightear", "Right_ear", "ear_right"])

    posture_pairs = []
    if bp_nose and bp_tail:
        posture_pairs.append((bp_nose, bp_tail, "body_length"))
    if bp_nose and bp_left_ear:
        posture_pairs.append((bp_nose, bp_left_ear, "nose_lear"))
    if bp_nose and bp_right_ear:
        posture_pairs.append((bp_nose, bp_right_ear, "nose_rear"))
    if bp_left_ear and bp_right_ear:
        posture_pairs.append((bp_left_ear, bp_right_ear, "head_width"))

    for bp_a, bp_b, name in posture_pairs:
        diff = bp_xy[bp_a] - bp_xy[bp_b]
        dist = np.sqrt((diff ** 2).sum(axis=1, keepdims=True) + 1e-8).astype(np.float32)
        feats.append(dist)
        feat_names += [f"dist_{name}"]

    n_posture = len(posture_pairs)
    print(f"  [FEAT] {len(bodyparts)} bodyparts, {n_posture} posture distances")
    if posture_pairs:
        print(f"         distances: {[name for _, _, name in posture_pairs]}")

    # =========================================================
    # Step 7: Body orientation angle (nose → tail direction)
    # =========================================================
    if bp_nose and bp_tail:
        diff = bp_xy[bp_nose] - bp_xy[bp_tail]
        angle = np.arctan2(diff[:, 1], diff[:, 0])  # (T,)
        feats.append(np.sin(angle).reshape(-1, 1).astype(np.float32))
        feat_names += ["body_angle_sin"]
        feats.append(np.cos(angle).reshape(-1, 1).astype(np.float32))
        feat_names += ["body_angle_cos"]
        print(f"  [FEAT] body_angle from {bp_nose} → {bp_tail}")
    else:
        avail = [bp_nose, bp_tail, bp_left_ear, bp_right_ear]
        print(f"  [FEAT WARN] Could not compute body_angle. "
              f"Found: nose={bp_nose}, tail={bp_tail}")

    # =========================================================
    # Step 8: MoSeq features (EXCLUDING absolute centroid x/y)
    # =========================================================
    if include_moseq and ("syllable" in ann_df.columns):
        feats.append(ann_df["syllable"].to_numpy(dtype=np.float32).reshape(-1, 1))
        feat_names += ["syllable_id"]

    if include_latents:
        for j in range(4):
            col = f"latent_state {j}"
            if col in ann_df.columns:
                feats.append(ann_df[col].to_numpy(dtype=np.float32).reshape(-1, 1))
                feat_names += [col]

    # heading is an angle (position-invariant) — KEEP
    # centroid x/y are absolute positions — EXCLUDED
    if include_centroid_heading and ("heading" in ann_df.columns):
        feats.append(ann_df["heading"].to_numpy(dtype=np.float32).reshape(-1, 1))
        feat_names += ["heading"]
    # NOTE: "centroid x" and "centroid y" intentionally excluded —
    # they encode absolute arena position and hurt cross-session transfer.

    # =========================================================
    # Step 9: Kinematics from allTrackData
    # =========================================================
    feats.append(speed.reshape(-1, 1))
    feat_names += ["speed_cm_per_s"]
    feats.append(angvel.reshape(-1, 1))
    feat_names += ["w_angvel"]

    # =========================================================
    # Assemble
    # =========================================================
    X = np.concatenate(feats, axis=1).astype(np.float32)

    # Replace any remaining NaN/Inf with 0 (safety net)
    nan_count = np.isnan(X).sum() + np.isinf(X).sum()
    if nan_count > 0:
        print(f"  [FEAT WARN] {nan_count} NaN/Inf values in features — replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_arr = ann_df[label_col].to_numpy(dtype=np.int64)[:T]
    if smooth_label_win is not None and smooth_label_win >= 3:
        y_arr = window_majority_labels(y_arr, win=smooth_label_win, ignore_label=-1)

    y_arr = enforce_stillness_by_speed(y_arr, speed, threshold=0.2, still_label=2)

    label_valid = np.isin(y_arr, VALID_CLASSES)
    mask = label_valid & kin_valid

    n_kin_masked = int(label_valid.sum() - mask.sum())
    if n_kin_masked > 0:
        print(f"  [MASK] {n_kin_masked} additional frames masked due to NaN kinematics "
              f"(total valid: {mask.sum()}/{T})")

    print(f"  [FEAT] Total features: {X.shape[1]}  "
          f"(was ~56 with absolute coords, now egocentric)")

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
            print(f"[WARN] No annotation CSV found for {h5.name}, skipping.")
            continue
        sessions.append({"session_prefix": prefix, "dlc_h5": str(h5), "ann_csv": ann_csv})
    if not sessions:
        raise FileNotFoundError(f"Could not auto-discover sessions in {data_dir}.")
    return sessions


def load_sessions(data_dir: str) -> List[Dict]:
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if os.path.isfile(manifest_path):
        print(f"[Sessions] Loading from manifest: {manifest_path}")
        return discover_sessions_from_manifest(data_dir)
    else:
        print(f"[Sessions] Auto-discovering from {data_dir}")
        return auto_discover_sessions(data_dir)


def load_all_sessions(sessions, combined_mat, label_col, dlc_conf_thr, smooth_label_win):
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
            print(f"  [ERROR] Failed to load DLC h5: {e}"); continue
        try:
            ann_df = pd.read_csv(sess["ann_csv"])
        except Exception as e:
            print(f"  [ERROR] Failed to load annotation CSV: {e}"); continue
        try:
            speed, w, kin_valid, fps, mat_name = load_kinematics_from_combined_mat(
                mat_path, prefix)
        except Exception as e:
            print(f"  [ERROR] Failed to load kinematics: {e}"); continue

        # Dimension diagnostic
        lens = {"dlc_h5": len(dlc_df), "ann_csv": len(ann_df),
                "speed": len(speed), "w": len(w)}
        print(f"  [DIM] " + "  ".join(f"{k}={v}" for k, v in lens.items()))
        if len(set(lens.values())) > 1:
            print(f"  [DIM WARN] Mismatch — will truncate to {min(lens.values())}")

        X, y, mask, feat_names = build_feature_matrix(
            dlc_df, ann_df, speed, w, kin_valid,
            label_col=label_col, dlc_conf_thr=dlc_conf_thr,
            smooth_label_win=smooth_label_win,
        )
        print(f"  T={len(X)}, F={X.shape[1]}, valid_frames={mask.sum()}, fps={fps}")
        print(f"  Label counts: {dict(zip(*np.unique(y[mask], return_counts=True)))}")

        loaded.append({
            "session_prefix": prefix, "X": X, "y": y, "mask": mask,
            "feat_names": feat_names, "fps": fps, "mat_name": mat_name,
        })

    if not loaded:
        raise RuntimeError("No sessions were successfully loaded.")
    return loaded


# ============================================================
# Dataset (with optional training augmentation)
# ============================================================
class SequenceDataset(Dataset):
    def __init__(self, X, y, mask, seq_len=256, stride=128,
                 augment=False, noise_std=0.03):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mask = mask.astype(bool)
        self.seq_len = int(seq_len)
        self.augment = augment
        self.noise_std = noise_std
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
        x = torch.from_numpy(self.X[a:b])
        if self.augment:
            # Small Gaussian noise helps generalization across sessions
            x = x + torch.randn_like(x) * self.noise_std
        return (
            x,
            torch.from_numpy(self.y[a:b]),
            torch.from_numpy(self.mask[a:b]),
        )


# ============================================================
# Train/val splitting
# ============================================================

def split_temporal(session_data, val_fraction, norm_mean, norm_std,
                   seq_len, stride, augment=False):
    train_datasets, val_datasets = [], []
    for sess in session_data:
        X = (sess["X"] - norm_mean) / norm_std
        y, mask = sess["y"], sess["mask"]
        T = len(X)
        split = int((1.0 - val_fraction) * T)
        train_datasets.append(SequenceDataset(X[:split], y[:split], mask[:split],
                                              seq_len=seq_len, stride=stride,
                                              augment=augment))
        val_datasets.append(SequenceDataset(X[split:], y[split:], mask[split:],
                                            seq_len=seq_len, stride=stride,
                                            augment=False))
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


# ============================================================
# Core training function
# ============================================================

def train_one_config(
    session_data, n_features, feat_names, args,
    channels=None, levels=None, kernel_size=None, dropout=None,
    lr=None, tv_weight=None, seq_len=None, stride=None,
    out_ckpt=None, verbose=True,
    train_indices=None, val_indices=None,
) -> Dict:
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
        "seq_len": seq_len, "stride": stride, "loss": args.loss,
        "augment": args.augment,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Determine train/val split ---
    if train_indices is not None and val_indices is not None:
        train_idx = train_indices
        val_idx = val_indices
    elif args.split_mode == "session":
        n = len(session_data)
        n_val = max(1, int(round(args.val_fraction * n)))
        all_idx = sorted(range(n), key=lambda i: session_data[i]["session_prefix"])
        train_idx = all_idx[:n - n_val]
        val_idx = all_idx[n - n_val:]
    else:
        train_idx = val_idx = None

    # --- Normalization from train data only ---
    if train_idx is not None:
        X_train_all = np.concatenate([session_data[i]["X"] for i in train_idx], axis=0)
    else:
        chunks = []
        for sess in session_data:
            split = int((1.0 - args.val_fraction) * len(sess["X"]))
            chunks.append(sess["X"][:split])
        X_train_all = np.concatenate(chunks, axis=0)
    norm_mean, norm_std = compute_norm_stats(X_train_all)
    del X_train_all

    # --- Build datasets ---
    augment = args.augment
    if train_idx is not None:
        train_datasets, val_datasets = [], []
        for i in train_idx:
            X = (session_data[i]["X"] - norm_mean) / norm_std
            train_datasets.append(SequenceDataset(X, session_data[i]["y"],
                                                  session_data[i]["mask"],
                                                  seq_len=seq_len, stride=stride,
                                                  augment=augment))
        for i in val_idx:
            X = (session_data[i]["X"] - norm_mean) / norm_std
            val_datasets.append(SequenceDataset(X, session_data[i]["y"],
                                                session_data[i]["mask"],
                                                seq_len=seq_len, stride=stride,
                                                augment=False))
        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
    else:
        train_ds, val_ds = split_temporal(
            session_data, args.val_fraction, norm_mean, norm_std,
            seq_len, stride, augment=augment)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Class weights ---
    cw = compute_class_weights(session_data, train_idx)
    cw_device = cw.to(device)
    if verbose:
        cw_str = "  ".join(f"{STATE_MAP[i]}={cw[i]:.2f}" for i in range(N_CLASSES))
        print(f"[ClassWeights] {cw_str}")

    # --- Loss function ---
    if args.loss == "focal":
        loss_fn = lambda logits, y, mask: masked_focal_loss(
            logits, y, mask, class_weights=cw_device, gamma=args.focal_gamma)
    else:
        loss_fn = lambda logits, y, mask: masked_ce_loss(
            logits, y, mask, class_weights=cw_device)

    # --- Model ---
    model = TCN(
        in_features=n_features, n_classes=N_CLASSES,
        channels=channels, levels=levels,
        kernel_size=kernel_size, dropout=dropout,
    ).to(device)

    n_params = model.count_parameters()
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Config] ch={channels} lv={levels} ks={kernel_size} "
              f"do={dropout} lr={lr} tv={tv_weight} loss={args.loss} aug={augment}")
        print(f"[Model]  {n_params:,} parameters")
        print(f"[Data]   Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
        if train_idx is not None:
            print(f"[Split]  Train: {[session_data[i]['session_prefix'] for i in train_idx]}")
            print(f"[Split]  Val:   {[session_data[i]['session_prefix'] for i in val_idx]}")
        print(f"{'='*60}")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    @torch.no_grad()
    def eval_val():
        model.eval()
        losses = []
        for xb, yb, mb in val_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb, mb)
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
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(xb)
                loss = loss_fn(logits, yb, mb) + temporal_tv_penalty(logits, weight=tv_weight)
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
            print(f"epoch {ep:03d} | train={train_loss:.4f} val={val_loss:.4f} "
                  f"lr={current_lr:.2e}{' *' if improved else ''}")

        if improved:
            best_val = val_loss
            best_epoch = ep
            patience_counter = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "in_features": int(n_features),
                "channels": int(channels), "levels": int(levels),
                "kernel_size": int(kernel_size), "dropout": float(dropout),
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
                "class_weights": cw.numpy(),
                "feature_version": "egocentric_v3",
            }
            torch.save(ckpt, out_ckpt)
            if verbose:
                print(f"  [CKPT] saved -> {out_ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                if verbose:
                    print(f"[Early stop] No improvement for {args.patience} epochs. "
                          f"Best: ep {best_epoch}, val={best_val:.4f}")
                break

    return {
        "best_val_loss": best_val, "best_epoch": best_epoch,
        "total_epochs": ep, "n_params": n_params,
        "config": config, "train_curve": train_curve,
        "out_ckpt": out_ckpt,
    }


# ============================================================
# LOSO cross-validation
# ============================================================

def run_loso(session_data, n_features, feat_names, args):
    n = len(session_data)
    if n < 2:
        raise ValueError("LOSO requires at least 2 sessions.")

    out_dir = Path(args.out_ckpt).parent
    loso_dir = out_dir / "loso"
    loso_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"[LOSO] {n}-fold cross-validation ({n} sessions)")
    print(f"[LOSO] Results -> {loso_dir}")
    print(f"{'#'*60}")

    fold_results = []
    all_preds = []
    all_gt = []

    for fold_i in range(n):
        val_prefix = session_data[fold_i]["session_prefix"]
        train_idx = [j for j in range(n) if j != fold_i]
        val_idx = [fold_i]

        fold_ckpt = str(loso_dir / f"fold_{fold_i}_{val_prefix}.pt")
        print(f"\n{'='*60}")
        print(f"[LOSO fold {fold_i+1}/{n}] Held-out: {val_prefix}")
        print(f"[LOSO fold {fold_i+1}/{n}] Train on: "
              f"{[session_data[j]['session_prefix'] for j in train_idx]}")

        result = train_one_config(
            session_data, n_features, feat_names, args,
            out_ckpt=fold_ckpt, verbose=True,
            train_indices=train_idx, val_indices=val_idx,
        )
        result["fold"] = fold_i
        result["held_out"] = val_prefix
        fold_results.append(result)

        # --- Evaluate on held-out session ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(fold_ckpt, map_location=device, weights_only=False)
        model = TCN(
            in_features=n_features, n_classes=N_CLASSES,
            channels=ckpt["channels"], levels=ckpt["levels"],
            kernel_size=ckpt["kernel_size"], dropout=ckpt["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        sess = session_data[fold_i]
        X_norm = (sess["X"] - ckpt["norm_mean"]) / ckpt["norm_std"]
        mask = sess["mask"]

        with torch.no_grad():
            X_t = torch.from_numpy(X_norm).unsqueeze(0).to(device)
            logits = model(X_t)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        y_true = sess["y"][mask]
        y_pred = preds[mask]
        all_gt.append(y_true)
        all_preds.append(y_pred)

        acc = (y_true == y_pred).mean()
        print(f"\n[LOSO fold {fold_i+1}] {val_prefix} — accuracy={acc:.3f}")
        for c in VALID_CLASSES:
            tp = ((y_pred == c) & (y_true == c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            fn = ((y_pred != c) & (y_true == c)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2*p*r / (p + r) if (p + r) > 0 else 0
            n_gt = int((y_true == c).sum())
            print(f"  {STATE_MAP[c]:>10s}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}  n={n_gt}")
        result["accuracy"] = float(acc)

    # --- Aggregate ---
    all_gt = np.concatenate(all_gt)
    all_preds = np.concatenate(all_preds)
    overall_acc = (all_gt == all_preds).mean()

    print(f"\n{'#'*60}")
    print(f"[LOSO AGGREGATE] Pooled across all {n} held-out sessions")
    print(f"  Overall accuracy: {overall_acc:.3f}")

    macro_f1s = []
    summary_rows = []
    for c in VALID_CLASSES:
        tp = ((all_preds == c) & (all_gt == c)).sum()
        fp = ((all_preds == c) & (all_gt != c)).sum()
        fn = ((all_preds != c) & (all_gt == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r / (p + r) if (p + r) > 0 else 0
        n_gt = int((all_gt == c).sum())
        macro_f1s.append(f1)
        print(f"  {STATE_MAP[c]:>10s}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}  n={n_gt}")
        summary_rows.append({"class": STATE_MAP[c], "precision": f"{p:.3f}",
                             "recall": f"{r:.3f}", "f1": f"{f1:.3f}", "n": n_gt})

    macro_f1 = np.mean(macro_f1s)
    print(f"  {'macro_F1':>10s}  = {macro_f1:.3f}")
    print(f"{'#'*60}")

    # Save CSVs
    summary_path = loso_dir / "loso_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "held_out", "accuracy",
                                           "best_val_loss", "best_epoch"])
        w.writeheader()
        for r in fold_results:
            w.writerow({"fold": r["fold"], "held_out": r["held_out"],
                         "accuracy": f"{r.get('accuracy', 0):.3f}",
                         "best_val_loss": f"{r['best_val_loss']:.4f}",
                         "best_epoch": r["best_epoch"]})

    per_class_path = loso_dir / "loso_per_class.csv"
    with open(per_class_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "n"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[LOSO] Saved: {summary_path}")
    print(f"[LOSO] Saved: {per_class_path}")

    # Train final model on all sessions
    print(f"\n[LOSO] Training final model on all {n} sessions for deployment...")
    final_result = train_one_config(
        session_data, n_features, feat_names, args,
        out_ckpt=args.out_ckpt, verbose=True,
    )
    print(f"[LOSO] Final model saved to: {args.out_ckpt}")
    return fold_results


# ============================================================
# Hyperparameter sweep
# ============================================================

SWEEP_GRID = {
    "channels":    [32, 64, 96, 128],
    "levels":      [4, 6],
    "dropout":     [0.1, 0.15, 0.2],
    "lr":          [1e-4, 3e-4, 1e-3],
    "tv_weight":   [0.08, 0.1, 0.15],
    "kernel_size": [5],
    "seq_len":     [384],
    "stride":      [384],
}


def generate_sweep_configs(grid):
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
            results.append({"tag": tag, "config": cfg, "best_val_loss": float("inf"),
                            "best_epoch": 0, "total_epochs": 0, "n_params": 0,
                            "elapsed_s": 0, "error": str(e)})

    with open(sweep_log_path, "w", newline="") as f:
        fieldnames = ["rank", "tag", "best_val_loss", "best_epoch", "total_epochs",
                      "n_params", "elapsed_s", "channels", "levels", "kernel_size",
                      "dropout", "lr", "tv_weight", "seq_len", "stride"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(sorted(results, key=lambda r: r["best_val_loss"]), 1):
            cfg = r.get("config", {})
            writer.writerow({
                "rank": rank, "tag": r.get("tag", ""),
                "best_val_loss": f"{r['best_val_loss']:.6f}",
                "best_epoch": r.get("best_epoch", 0),
                "total_epochs": r.get("total_epochs", 0),
                "n_params": r.get("n_params", 0),
                "elapsed_s": f"{r.get('elapsed_s', 0):.1f}",
                **{k: cfg.get(k, "") for k in ["channels", "levels", "kernel_size",
                                                 "dropout", "lr", "tv_weight",
                                                 "seq_len", "stride"]},
            })

    valid = [r for r in results if r["best_val_loss"] < float("inf")]
    if valid:
        best = min(valid, key=lambda r: r["best_val_loss"])
        best_ckpt = best.get("out_ckpt")
        if best_ckpt and os.path.isfile(best_ckpt):
            import shutil
            shutil.copy2(best_ckpt, args.out_ckpt)
            print(f"\n{'='*60}")
            print(f"[SWEEP DONE] Best: {best['tag']}")
            print(f"  val_loss={best['best_val_loss']:.4f} | config={best['config']}")
            print(f"  Saved to: {args.out_ckpt}")
            print(f"{'='*60}")

        for r in sorted(valid, key=lambda r: r["best_val_loss"])[:5]:
            curve = r.get("train_curve", [])
            if curve:
                curve_path = sweep_dir / f"curve_{r['tag']}.csv"
                with open(curve_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epoch", "train_loss", "val_loss", "lr"])
                    for row in curve:
                        w.writerow(row)
    return results


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Train TCN for mouse behavior (egocentric features v3).")

    # Input
    ap.add_argument("--dlc_h5", default=None)
    ap.add_argument("--ann_csv", default=None)
    ap.add_argument("--session_prefix", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--combined_mat", default=None)
    ap.add_argument("--label_col", default="human_labeled_state")
    ap.add_argument("--out_ckpt", required=True)

    # Split
    ap.add_argument("--split_mode", choices=["temporal", "session", "loso"],
                    default="temporal")
    ap.add_argument("--val_fraction", type=float, default=0.2)

    # Preprocessing
    ap.add_argument("--dlc_conf_thr", type=float, default=0.8)
    ap.add_argument("--smooth_label_win", type=int, default=9)

    # Training
    ap.add_argument("--seq_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--augment", action="store_true",
                    help="Add Gaussian noise augmentation during training.")

    # Model
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--tv_weight", type=float, default=0.04)

    # Loss
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    # Sweep
    ap.add_argument("--sweep", action="store_true")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Features] EGOCENTRIC v3 — centroid-relative positions, "
          f"posture distances, body angle")

    # --- Load data ---
    if args.data_dir is not None:
        if args.combined_mat is None:
            mat_files = list(Path(args.data_dir).glob("*.mat"))
            if len(mat_files) == 1:
                args.combined_mat = str(mat_files[0])
                print(f"[Auto] Found single .mat: {args.combined_mat}")
            else:
                raise ValueError("--combined_mat required.")

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
            raise ValueError("Single-session requires: --dlc_h5, --ann_csv, "
                             "--combined_mat, --session_prefix.")
        dlc_df = load_dlc_h5(args.dlc_h5)
        ann_df = pd.read_csv(args.ann_csv)
        speed, w, kin_valid, fps, mat_name = load_kinematics_from_combined_mat(
            args.combined_mat, args.session_prefix)
        print(f"[MAT] matched: {mat_name} | fps={fps}")
        X, y, mask, feat_names = build_feature_matrix(
            dlc_df, ann_df, speed, w, kin_valid,
            label_col=args.label_col, dlc_conf_thr=args.dlc_conf_thr,
            smooth_label_win=args.smooth_label_win,
        )
        print(f"[Data] T={len(X)} F={X.shape[1]} valid_frames={mask.sum()}")
        session_data = [{"session_prefix": args.session_prefix, "X": X, "y": y,
                         "mask": mask, "feat_names": feat_names, "fps": fps,
                         "mat_name": mat_name}]

    # Validate features
    n_features = session_data[0]["X"].shape[1]
    feat_names = session_data[0]["feat_names"]
    for sess in session_data[1:]:
        if sess["X"].shape[1] != n_features:
            raise ValueError(f"Feature mismatch: {sess['session_prefix']} has "
                             f"{sess['X'].shape[1]} features, expected {n_features}.")

    print(f"\n[Features] {n_features} features: {feat_names[:6]} ... {feat_names[-4:]}")

    # --- Run ---
    if args.sweep:
        run_sweep(session_data, n_features, feat_names, args)
    elif args.split_mode == "loso":
        run_loso(session_data, n_features, feat_names, args)
    else:
        result = train_one_config(
            session_data, n_features, feat_names, args, verbose=True)
        print(f"\n[Done] best val_loss={result['best_val_loss']:.4f} "
              f"@ epoch {result['best_epoch']}/{result['total_epochs']} "
              f"| {result['n_params']:,} params")

        curve_path = Path(args.out_ckpt).with_suffix(".curve.csv")
        with open(curve_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "lr"])
            for row in result["train_curve"]:
                w.writerow(row)
        print(f"[Curve] {curve_path}")


if __name__ == "__main__":
    main()