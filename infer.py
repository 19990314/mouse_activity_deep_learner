#!/usr/bin/env python3
"""
Inference script for mouse open-field behavior classification
using a trained PyTorch TCN.

Supports two input modes:
  1. Single-session (original):
       --dlc_h5 FILE --ann_csv FILE --combined_mat FILE --session_prefix PREFIX
       --checkpoint FILE --out_csv FILE

  2. Multi-session via manifest:
       --manifest_csv FILE --combined_mat FILE --checkpoint FILE --out_dir DIR
     manifest.csv columns: session_prefix, dlc_h5, ann_csv
     Optional column: mat_path (overrides --combined_mat per session)
     Paths in manifest.csv are relative to the directory containing manifest.csv.

     Use --session_prefix to run only matching sessions from the manifest
     (prefix match or exact match). Omit to run all sessions.

Usage examples:
  # Single session (backward-compatible)
  python infer.py --dlc_h5 data/sc04_d3_of.h5 --ann_csv data/sc04_d3_of_ann.csv \
    --combined_mat data/combined.mat --session_prefix sc04_d3_of \
    --checkpoint model.pt --out_csv outputs/sc04_d3_of_predictions.csv

  # All sessions from manifest
  python infer.py --manifest_csv data/sessions/manifest.csv \
    --combined_mat data/combined.mat --checkpoint model.pt \
    --out_dir outputs/predictions/

  # Only sessions matching a prefix from manifest
  python infer.py --manifest_csv data/sessions/manifest.csv \
    --combined_mat data/combined.mat --checkpoint model.pt \
    --out_dir outputs/predictions/ --session_prefix sc04
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
    0: "turn",
    1: "forward",
    2: "still",
    3: "explore",
    4: "rear",
    5: "groom",
    -1: "unsigned",
}

# ============================================================
# TCN model (IDENTICAL to train.py)
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
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
        y = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        y = self.dropout(F.gelu(self.norm2(self.conv2(y))))
        return x + y


class TCN(nn.Module):
    def __init__(self, in_features, n_classes=6,
                 channels=128, levels=8,
                 kernel_size=5, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, channels, 1)
        blocks = []
        for i in range(levels):
            blocks.append(
                TCNBlock(channels,
                         kernel_size=kernel_size,
                         dilation=2 ** i,
                         dropout=dropout)
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(channels, n_classes, 1)

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)           # (B, K, T)
        return x.transpose(1, 2)       # (B, T, K)

# ============================================================
# DLC utilities
# ============================================================
def load_dlc_csv(path):
    df = pd.read_csv(path, header=[0, 1, 2])
    if df.columns[0][0] == "scorer":
        df = df.drop(columns=df.columns[0])
    return df


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
            limit_direction="both").to_numpy()
    return out


def velocity(xy):
    v = np.zeros_like(xy, dtype=np.float32)
    v[1:] = xy[1:] - xy[:-1]
    return v

# ============================================================
# MAT loader
# ============================================================
def load_kinematics(mat_path, session_prefix):
    m = loadmat(mat_path)
    allTrackData = m["allTrackData"]

    idx = None
    for i in range(allTrackData.shape[1]):
        name = np.squeeze(allTrackData[0, i]["name"]).item()
        if isinstance(name, str) and name.startswith(session_prefix):
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"Session {session_prefix} not found in MAT")

    entry = allTrackData[0, idx]
    speed = np.squeeze(entry["speed_pixels_per_frame"]).astype(np.float32)
    angvel = np.squeeze(entry["w"]).astype(np.float32)
    return speed, angvel

# ============================================================
# Feature construction (must match training)
# ============================================================
def build_features(dlc_df, ann_df, speed, angvel, conf_thr=0.6):
    T = min(len(dlc_df), len(ann_df), len(speed), len(angvel))
    dlc_df = dlc_df.iloc[:T]
    ann_df = ann_df.iloc[:T]
    speed = speed[:T]
    angvel = angvel[:T]

    bodyparts = sorted({c[1] for c in dlc_df.columns})
    bodyparts = [bp for bp in bodyparts if bp not in ("bodyparts", "coords")]

    feats = []
    conf_all = []

    for bp in bodyparts:
        x = dlc_df.xs((bp, "x"), level=(1, 2), axis=1).to_numpy().squeeze()
        y = dlc_df.xs((bp, "y"), level=(1, 2), axis=1).to_numpy().squeeze()
        p = dlc_df.xs((bp, "likelihood"),
                      level=(1, 2), axis=1).to_numpy().squeeze()

        xy = interpolate_low_conf(
            np.stack([x, y], axis=1), p, thr=conf_thr
        )
        v = velocity(xy)

        feats.append(xy)
        feats.append(v)
        conf_all.append(p.reshape(-1, 1))

    conf_all = np.concatenate(conf_all, axis=1)
    feats.append(conf_all.mean(axis=1, keepdims=True))
    feats.append(conf_all.min(axis=1, keepdims=True))

    feats.append(ann_df["syllable"].to_numpy().reshape(-1, 1))
    for j in range(4):
        feats.append(ann_df[f"latent_state {j}"].to_numpy().reshape(-1, 1))

    for col in ["centroid x", "centroid y", "heading"]:
        feats.append(ann_df[col].to_numpy().reshape(-1, 1))

    feats.append(speed.reshape(-1, 1))
    feats.append(angvel.reshape(-1, 1))

    X = np.concatenate(feats, axis=1).astype(np.float32)
    return X, T

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
def load_manifest(manifest_csv: str, session_prefix_filter: Optional[str] = None) -> List[Dict]:
    """
    Read manifest.csv. Expected columns: session_prefix, dlc_h5, ann_csv.
    Optional column: mat_path (per-session MAT override).

    Paths are resolved relative to the directory containing manifest.csv.

    If session_prefix_filter is given, only sessions whose session_prefix
    starts with (or exactly matches) the filter string are returned.
    """
    manifest_path = Path(manifest_csv)
    base_dir = manifest_path.parent

    df = pd.read_csv(manifest_path)
    required = {"session_prefix", "dlc_h5", "ann_csv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing required columns: {missing}")

    sessions = []
    for _, row in df.iterrows():
        prefix = str(row["session_prefix"]).strip()

        # Apply prefix filter if provided
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
        all_prefixes = df["session_prefix"].tolist()
        raise ValueError(
            f"No sessions matched prefix filter '{session_prefix_filter}'. "
            f"Available prefixes: {all_prefixes}"
        )

    return sessions


# ============================================================
# Single-session inference (core logic, reused by both modes)
# ============================================================
def infer_session(
    dlc_h5: str,
    ann_csv: str,
    mat_path: str,
    session_prefix: str,
    model: TCN,
    ckpt: dict,
    device: str,
    switch_penalty: float,
    min_bout_len: Dict[int, int],
    out_csv: str,
) -> str:
    """Run inference for one session, write out_csv, return the output path."""

    print(f"\n[Infer] session={session_prefix}")
    print(f"  dlc_h5:  {dlc_h5}")
    print(f"  ann_csv: {ann_csv}")
    print(f"  mat:     {mat_path}")

    # Load data
    dlc_df = load_dlc_h5(dlc_h5)
    ann_df = pd.read_csv(ann_csv)
    speed, angvel = load_kinematics(mat_path, session_prefix)

    # Build features
    X, T = build_features(dlc_df, ann_df, speed, angvel)

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

    # --- Single-session args (original) ---
    ap.add_argument("--dlc_h5", default=None,
                    help="Single-session DLC .h5 file.")
    ap.add_argument("--ann_csv", default=None,
                    help="Single-session annotation CSV.")
    ap.add_argument("--out_csv", default=None,
                    help="Output CSV path (single-session mode).")

    # --- Multi-session args ---
    ap.add_argument("--manifest_csv", default=None,
                    help="Path to manifest.csv for batch inference. "
                         "Overrides --dlc_h5/--ann_csv.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for batch predictions "
                         "(one CSV per session). Used with --manifest_csv.")

    # --- Shared ---
    ap.add_argument("--combined_mat", required=True,
                    help="Path to combined .mat with kinematics.")
    ap.add_argument("--session_prefix", default=None,
                    help="Session prefix for MAT lookup. "
                         "In single-session mode: required, exact session name. "
                         "In manifest mode: optional filter — only sessions whose "
                         "prefix starts with this string are processed. "
                         "Omit to process all sessions in the manifest.")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to trained .pt checkpoint.")

    # --- Decoding ---
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
        0: args.min_turn,
        1: args.min_forward,
        2: args.min_still,
        3: args.min_explore,
        4: args.min_rear,
        5: args.min_groom,
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

    # ===========================================
    # Route: manifest (batch) vs single-session
    # ===========================================
    if args.manifest_csv is not None:
        # --- Batch mode ---
        if args.out_dir is None:
            args.out_dir = "outputs/predictions"
            print(f"[Note] --out_dir not set, defaulting to {args.out_dir}")

        sessions = load_manifest(
            args.manifest_csv,
            session_prefix_filter=args.session_prefix,  # None = all
        )

        n_total = len(sessions)
        filter_msg = (f" (filtered by prefix '{args.session_prefix}')"
                      if args.session_prefix else " (all)")
        print(f"[Manifest] {n_total} session(s) to process{filter_msg}:")
        for s in sessions:
            print(f"  - {s['session_prefix']}")

        results = []
        errors = []

        for i, sess in enumerate(sessions, 1):
            prefix = sess["session_prefix"]
            mat_path = sess.get("mat_path", args.combined_mat)
            out_csv = os.path.join(args.out_dir, f"{prefix}_predictions.csv")

            try:
                path = infer_session(
                    dlc_h5=sess["dlc_h5"],
                    ann_csv=sess["ann_csv"],
                    mat_path=mat_path,
                    session_prefix=prefix,
                    model=model,
                    ckpt=ckpt,
                    device=device,
                    switch_penalty=args.switch_penalty,
                    min_bout_len=MIN_BOUT_LEN,
                    out_csv=out_csv,
                )
                results.append(path)
            except Exception as e:
                print(f"  [ERROR] {prefix}: {e}")
                errors.append((prefix, str(e)))

        # Summary
        print(f"\n{'='*50}")
        print(f"[Done] {len(results)}/{n_total} sessions succeeded.")
        if errors:
            print(f"[Errors] {len(errors)} session(s) failed:")
            for prefix, err in errors:
                print(f"  - {prefix}: {err}")
        print(f"Outputs in: {args.out_dir}")

    else:
        # --- Single-session mode (original behavior) ---
        if not all([args.dlc_h5, args.ann_csv, args.session_prefix]):
            raise ValueError(
                "Single-session mode requires: --dlc_h5, --ann_csv, --session_prefix. "
                "Alternatively, use --manifest_csv for batch inference."
            )
        if args.out_csv is None:
            args.out_csv = f"outputs/predictions/{args.session_prefix}_predictions.csv"
            print(f"[Note] --out_csv not set, defaulting to {args.out_csv}")

        infer_session(
            dlc_h5=args.dlc_h5,
            ann_csv=args.ann_csv,
            mat_path=args.combined_mat,
            session_prefix=args.session_prefix,
            model=model,
            ckpt=ckpt,
            device=device,
            switch_penalty=args.switch_penalty,
            min_bout_len=MIN_BOUT_LEN,
            out_csv=args.out_csv,
        )


if __name__ == "__main__":
    main()