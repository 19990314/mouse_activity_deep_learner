#!/usr/bin/env python3
"""
Inference script for mouse open-field behavior classification
using a trained PyTorch TCN.

Features match train.py exactly:
  - Egocentric bodypart positions (relative to neck)
  - Inter-bodypart distances (body_length, head_width, etc.)
  - Body orientation angle (sin/cos)
  - Kinematics: speed_cm_per_s_origin + angular velocity
  - No absolute centroid x/y

Supports two input modes:
  1. Single-session:
       --dlc_h5 FILE --ann_csv FILE --combined_mat FILE --session_prefix PREFIX
       --checkpoint FILE --out_csv FILE

  2. Multi-session via manifest:
       --manifest_csv FILE --combined_mat FILE --checkpoint FILE --out_dir DIR
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat

# ============================================================
# State mapping (must match training)
# ============================================================
STATE_MAP = {
    0: "turn", 1: "forward", 2: "still",
    3: "explore", 4: "rear", 5: "groom",
    -1: "unsigned",
}

# ============================================================
# TCN model (IDENTICAL to train.py)
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

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
    def __init__(self, in_features, n_classes=6,
                 channels=32, levels=6,
                 kernel_size=5, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, channels, 1)
        blocks = []
        for i in range(levels):
            blocks.append(
                TCNBlock(channels, kernel_size=kernel_size,
                         dilation=2 ** i, dropout=dropout)
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(channels, n_classes, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x.transpose(1, 2)

# ============================================================
# DLC utilities (identical to train.py)
# ============================================================
def load_dlc_h5(path):
    df = pd.read_hdf(path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DLC .h5 did not load with MultiIndex columns.")
    return df


def interpolate_low_conf(xy, conf, thr=0.6):
    out = xy.astype(np.float32).copy()
    out[conf < thr] = np.nan
    for j in range(2):
        out[:, j] = pd.Series(out[:, j]).interpolate(
            limit_direction="both").to_numpy(dtype=np.float32)
    return out


def compute_velocity(xy):
    v = np.zeros_like(xy, dtype=np.float32)
    v[1:] = xy[1:] - xy[:-1]
    return v


def _find_bodypart(bodyparts: List[str], candidates: List[str]) -> Optional[str]:
    """Find a bodypart name from a list of possible names."""
    for c in candidates:
        if c in bodyparts:
            return c
        for bp in bodyparts:
            if bp.lower() == c.lower():
                return bp
    return None


# ============================================================
# MAT loader (matches train.py — speed_cm_per_s_origin + NaN handling)
# ============================================================
def load_kinematics(mat_path, session_prefix):
    """
    Load speed (cm/s) and angular velocity from allTrackData.
    Returns speed, w, kin_valid mask, fps, name.
    """
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
        raise RuntimeError(
            f"Session '{session_prefix}' not found in MAT. "
            f"Available:\n" + "\n".join(map(str, names[:10]))
        )

    entry = allTrackData[0, best_idx]
    speed = np.squeeze(entry["speed_cm_per_s_origin"]).astype(np.float32)
    w = np.squeeze(entry["w"]).astype(np.float32)
    fps = float(np.squeeze(entry["fps"]))
    name = np.squeeze(entry["name"]).item()

    # Handle length mismatch
    if len(speed) != len(w):
        min_len = min(len(speed), len(w))
        print(f"  [KIN] Length mismatch: speed={len(speed)}, w={len(w)} — "
              f"truncating both to {min_len}")
        speed = speed[:min_len]
        w = w[:min_len]

    # Validity mask
    kin_valid = np.isfinite(speed) & np.isfinite(w)
    n_nan = int((~kin_valid).sum())
    if n_nan > 0:
        print(f"  [KIN] {n_nan}/{len(speed)} frames have NaN kinematics — zero-filled")
        speed = np.where(kin_valid, speed, 0.0).astype(np.float32)
        w = np.where(kin_valid, w, 0.0).astype(np.float32)

    return speed, w, kin_valid, fps, name


# ============================================================
# Feature construction (MUST match train.py build_feature_matrix)
# ============================================================
def build_features(dlc_df, ann_df, speed, angvel, kin_valid, conf_thr=0.7):
    """
    Build EGOCENTRIC feature matrix — identical to train.py.
    For inference we don't need labels/mask, just features.
    """
    T = min(len(dlc_df), len(ann_df), len(speed), len(angvel), len(kin_valid))
    dlc_df = dlc_df.iloc[:T]
    ann_df = ann_df.iloc[:T]
    speed = speed[:T]
    angvel = angvel[:T]

    bodyparts = sorted(set([c[1] for c in dlc_df.columns]))
    bodyparts = [bp for bp in bodyparts if bp not in ("bodyparts", "coords")]

    # ---- Step 1: Extract all bodypart positions & confidences ----
    bp_xy = {}
    bp_conf = {}
    for bp in bodyparts:
        x = dlc_df.xs((bp, "x"), level=(1, 2), axis=1).to_numpy().squeeze()
        y = dlc_df.xs((bp, "y"), level=(1, 2), axis=1).to_numpy().squeeze()
        p = dlc_df.xs((bp, "likelihood"), level=(1, 2), axis=1).to_numpy().squeeze().astype(np.float32)
        xy = np.stack([x, y], axis=1).astype(np.float32)
        xy = interpolate_low_conf(xy, p, thr=conf_thr)
        bp_xy[bp] = xy
        bp_conf[bp] = p

    # ---- Step 2: Body reference point = "neck" ----
    bp_neck = _find_bodypart(bodyparts, ["neck", "Neck", "neck_base"])
    if bp_neck is None:
        print(f"  [FEAT WARN] 'neck' not found in bodyparts: {bodyparts}")
        print(f"  [FEAT WARN] Falling back to mean of all bodyparts")
        all_xy = np.stack(list(bp_xy.values()), axis=1)
        centroid = np.nanmean(all_xy, axis=1)
    else:
        centroid = bp_xy[bp_neck].copy()
        print(f"  [FEAT] Using '{bp_neck}' as body reference point")

    for j in range(2):
        s = pd.Series(centroid[:, j])
        centroid[:, j] = s.interpolate(limit_direction="both").to_numpy(dtype=np.float32)

    # ---- Step 3: Egocentric features per bodypart ----
    feats = []
    feat_names = []

    for bp in bodyparts:
        xy = bp_xy[bp]
        rel_xy = (xy - centroid).astype(np.float32)
        vel = compute_velocity(xy)
        feats.append(rel_xy)
        feat_names += [f"{bp}_rx", f"{bp}_ry"]
        feats.append(vel)
        feat_names += [f"{bp}_vx", f"{bp}_vy"]

    # ---- Step 4: Centroid velocity ----
    centroid_vel = compute_velocity(centroid)
    feats.append(centroid_vel)
    feat_names += ["centroid_vx", "centroid_vy"]

    # ---- Step 5: Confidence features ----
    conf_mat = np.stack(list(bp_conf.values()), axis=1)
    feats.append(np.mean(conf_mat, axis=1, keepdims=True))
    feat_names += ["p_mean"]
    feats.append(np.min(conf_mat, axis=1, keepdims=True))
    feat_names += ["p_min"]

    # ---- Step 6: Inter-bodypart distances ----
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

    print(f"  [FEAT] {len(bodyparts)} bodyparts, {len(posture_pairs)} posture distances")

    # ---- Step 7: Body orientation angle ----
    if bp_nose and bp_tail:
        diff = bp_xy[bp_nose] - bp_xy[bp_tail]
        angle = np.arctan2(diff[:, 1], diff[:, 0])
        feats.append(np.sin(angle).reshape(-1, 1).astype(np.float32))
        feat_names += ["body_angle_sin"]
        feats.append(np.cos(angle).reshape(-1, 1).astype(np.float32))
        feat_names += ["body_angle_cos"]

    # ---- Step 8: MoSeq features (EXCLUDING absolute centroid x/y) ----
    if "syllable" in ann_df.columns:
        feats.append(ann_df["syllable"].to_numpy(dtype=np.float32).reshape(-1, 1))
        feat_names += ["syllable_id"]

    for j in range(4):
        col = f"latent_state {j}"
        if col in ann_df.columns:
            feats.append(ann_df[col].to_numpy(dtype=np.float32).reshape(-1, 1))
            feat_names += [col]

    # heading is angle (position-invariant) — KEEP
    # centroid x/y are absolute positions — EXCLUDED
    if "heading" in ann_df.columns:
        feats.append(ann_df["heading"].to_numpy(dtype=np.float32).reshape(-1, 1))
        feat_names += ["heading"]

    # ---- Step 9: Kinematics ----
    feats.append(speed.reshape(-1, 1))
    feat_names += ["speed_cm_per_s"]
    feats.append(angvel.reshape(-1, 1))
    feat_names += ["w_angvel"]

    # ---- Assemble ----
    X = np.concatenate(feats, axis=1).astype(np.float32)

    nan_count = np.isnan(X).sum() + np.isinf(X).sum()
    if nan_count > 0:
        print(f"  [FEAT WARN] {nan_count} NaN/Inf values — replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  [FEAT] Total features: {X.shape[1]} (egocentric)")
    return X, T, feat_names


# ============================================================
# Decoding
# ============================================================
def viterbi(logp, switch_penalty=2.5):
    T, K = logp.shape
    dp = np.full((T, K), -np.inf)
    back = np.zeros((T, K), dtype=int)
    trans = -switch_penalty * (np.ones((K, K)) - np.eye(K))

    dp[0] = logp[0]
    for t in range(1, T):
        scores = dp[t-1][:, None] + trans
        back[t] = scores.argmax(axis=0)
        dp[t] = logp[t] + scores[back[t], range(K)]

    path = np.zeros(T, dtype=int)
    path[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]
    return path


def enforce_min_duration(states, min_len_by_class):
    s = states.copy()
    T = len(s)
    i = 0
    while i < T:
        j = i + 1
        while j < T and s[j] == s[i]:
            j += 1
        cls = int(s[i])
        bout_len = j - i
        min_len = min_len_by_class.get(cls, 1)
        if bout_len < min_len:
            left = s[i - 1] if i > 0 else None
            right = s[j] if j < T else None
            if left is not None:
                s[i:j] = left
            elif right is not None:
                s[i:j] = right
        i = j
    return s


# ============================================================
# Manifest loading
# ============================================================
def load_manifest(manifest_csv, session_prefix_filter=None):
    manifest_path = Path(manifest_csv)
    base_dir = manifest_path.parent
    df = pd.read_csv(manifest_path)

    required = {"session_prefix", "dlc_h5", "ann_csv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {missing}")

    sessions = []
    for _, row in df.iterrows():
        prefix = str(row["session_prefix"]).strip()
        if session_prefix_filter is not None:
            filt = session_prefix_filter.strip()
            if not (prefix == filt or prefix.startswith(filt)):
                continue
        entry = {
            "session_prefix": prefix,
            "dlc_h5": str(base_dir / row["dlc_h5"]),
            "ann_csv": str(base_dir / row["ann_csv"]),
        }
        if "mat_path" in df.columns and pd.notna(row.get("mat_path")):
            entry["mat_path"] = str(base_dir / row["mat_path"])
        sessions.append(entry)

    if not sessions:
        raise ValueError(f"No sessions matched filter '{session_prefix_filter}'.")
    return sessions


# ============================================================
# Single-session inference
# ============================================================
def infer_session(
    dlc_h5, ann_csv, mat_path, session_prefix,
    model, ckpt, device,
    switch_penalty, min_bout_len, out_csv,
):
    print(f"\n[Infer] session={session_prefix}")
    print(f"  dlc_h5:  {dlc_h5}")
    print(f"  ann_csv: {ann_csv}")
    print(f"  mat:     {mat_path}")

    # Load data
    dlc_df = load_dlc_h5(dlc_h5)
    ann_df = pd.read_csv(ann_csv)
    speed, w, kin_valid, fps, mat_name = load_kinematics(mat_path, session_prefix)
    print(f"  [MAT] matched: {mat_name} | fps={fps}")

    # Build features (egocentric, matching train.py)
    conf_thr = ckpt.get("dlc_conf_thr", 0.7)
    X, T, feat_names = build_features(dlc_df, ann_df, speed, w, kin_valid,
                                       conf_thr=conf_thr)

    # Validate feature count matches checkpoint
    expected_feats = ckpt["in_features"]
    if X.shape[1] != expected_feats:
        raise ValueError(
            f"Feature mismatch: built {X.shape[1]} features but checkpoint "
            f"expects {expected_feats}. Features built: {feat_names}\n"
            f"Checkpoint features: {ckpt.get('feature_names', 'not stored')}"
        )

    # Normalize using training stats
    X = (X - ckpt["norm_mean"]) / ckpt["norm_std"]

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).unsqueeze(0).to(device))
    logp = F.log_softmax(logits.squeeze(0), dim=-1).cpu().numpy()

    pred_raw = logp.argmax(axis=1)
    pred_vit = viterbi(logp, switch_penalty)
    pred_final = enforce_min_duration(pred_vit, min_bout_len)

    # Output
    out = ann_df.iloc[:T].copy()
    out["session_prefix"] = session_prefix
    out["speed_cm_per_s"] = speed[:T]

    out["tcn_pred"] = pred_raw
    out["tcn_pred_label"] = out["tcn_pred"].map(STATE_MAP)

    out["tcn_viterbi"] = pred_vit
    out["tcn_viterbi_label"] = out["tcn_viterbi"].map(STATE_MAP)

    out["tcn_final"] = pred_final
    out["tcn_final_label"] = out["tcn_final"].map(STATE_MAP)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  [OK] saved {out_csv}  ({T} frames)")
    return out_csv


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="TCN inference — single session or batch via manifest.csv"
    )

    # Single-session
    ap.add_argument("--dlc_h5", default=None)
    ap.add_argument("--ann_csv", default=None)
    ap.add_argument("--out_csv", default=None)

    # Multi-session
    ap.add_argument("--manifest_csv", default=None)
    ap.add_argument("--out_dir", default=None)

    # Shared
    ap.add_argument("--combined_mat", required=True)
    ap.add_argument("--session_prefix", default=None)
    ap.add_argument("--checkpoint", required=True)

    # Decoding
    ap.add_argument("--switch_penalty", type=float, default=4)
    ap.add_argument("--min_turn", type=int, default=8)
    ap.add_argument("--min_forward", type=int, default=10)
    ap.add_argument("--min_still", type=int, default=12)
    ap.add_argument("--min_explore", type=int, default=8)
    ap.add_argument("--min_rear", type=int, default=8)
    ap.add_argument("--min_groom", type=int, default=14)

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MIN_BOUT_LEN = {
        0: args.min_turn, 1: args.min_forward, 2: args.min_still,
        3: args.min_explore, 4: args.min_rear, 5: args.min_groom,
    }

    # Load checkpoint & model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TCN(
        in_features=ckpt["in_features"],
        channels=ckpt["channels"],
        levels=ckpt["levels"],
        kernel_size=ckpt["kernel_size"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    print(f"[Model] {ckpt['in_features']} features, ch={ckpt['channels']}, "
          f"lv={ckpt['levels']}, ks={ckpt['kernel_size']}")
    if "config" in ckpt:
        print(f"[Config] {ckpt['config']}")

    # Route: manifest vs single-session
    if args.manifest_csv is not None:
        if args.out_dir is None:
            args.out_dir = "outputs/predictions"
        sessions = load_manifest(args.manifest_csv, args.session_prefix)
        print(f"[Manifest] {len(sessions)} session(s) to process")

        results, errors = [], []
        for sess in sessions:
            prefix = sess["session_prefix"]
            mat_path = sess.get("mat_path", args.combined_mat)
            out_csv = os.path.join(args.out_dir, f"{prefix}_predictions.csv")
            try:
                path = infer_session(
                    dlc_h5=sess["dlc_h5"], ann_csv=sess["ann_csv"],
                    mat_path=mat_path, session_prefix=prefix,
                    model=model, ckpt=ckpt, device=device,
                    switch_penalty=args.switch_penalty,
                    min_bout_len=MIN_BOUT_LEN, out_csv=out_csv,
                )
                results.append(path)
            except Exception as e:
                print(f"  [ERROR] {prefix}: {e}")
                errors.append((prefix, str(e)))

        print(f"\n{'='*50}")
        print(f"[Done] {len(results)}/{len(sessions)} sessions succeeded.")
        if errors:
            for prefix, err in errors:
                print(f"  [FAIL] {prefix}: {err}")

    else:
        if not all([args.dlc_h5, args.ann_csv, args.session_prefix]):
            raise ValueError("Single-session requires: --dlc_h5, --ann_csv, --session_prefix")
        if args.out_csv is None:
            args.out_csv = f"outputs/predictions/{args.session_prefix}_predictions.csv"

        infer_session(
            dlc_h5=args.dlc_h5, ann_csv=args.ann_csv,
            mat_path=args.combined_mat, session_prefix=args.session_prefix,
            model=model, ckpt=ckpt, device=device,
            switch_penalty=args.switch_penalty,
            min_bout_len=MIN_BOUT_LEN, out_csv=args.out_csv,
        )


if __name__ == "__main__":
    main()