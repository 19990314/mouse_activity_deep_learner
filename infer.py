#!/usr/bin/env python3
"""
Inference script for mouse open-field behavior classification
using a trained PyTorch TCN.

Outputs per-frame predictions with temporal smoothing.
"""

import argparse
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
        # x: (B, T, F)
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
# MAT loader (session prefix match)
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

    # MoSeq + latent
    feats.append(ann_df["syllable"].to_numpy().reshape(-1, 1))
    for j in range(4):
        feats.append(ann_df[f"latent_state {j}"].to_numpy().reshape(-1, 1))

    # centroid + heading
    for col in ["centroid x", "centroid y", "heading"]:
        feats.append(ann_df[col].to_numpy().reshape(-1, 1))

    # kinematics
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

def enforce_min_duration(states, min_len):
    s = states.copy()
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i < min_len.get(s[i], 1):
            if i > 0:
                s[i:j] = s[i-1]
        i = j
    return s

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlc_csv", required=True)
    ap.add_argument("--ann_csv", required=True)
    ap.add_argument("--combined_mat", required=True)
    ap.add_argument("--session_prefix", default="sc04_d1_of")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--switch_penalty", type=float, default=2.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    dlc_df = load_dlc_csv(args.dlc_csv)
    ann_df = pd.read_csv(args.ann_csv)
    speed, angvel = load_kinematics(args.combined_mat, args.session_prefix)

    # Build features
    X, T = build_features(dlc_df, ann_df, speed, angvel)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TCN(
        in_features=ckpt["in_features"],
        channels=ckpt["channels"],
        levels=ckpt["levels"],
        kernel_size=ckpt["kernel_size"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # Normalize using training stats
    X = (X - ckpt["norm_mean"]) / ckpt["norm_std"]

    # Inference
    with torch.no_grad():
        logits = model(torch.from_numpy(X).unsqueeze(0).to(device))
    logp = F.log_softmax(logits.squeeze(0), dim=-1).cpu().numpy()

    pred_raw = logp.argmax(axis=1)
    pred_vit = viterbi(logp, args.switch_penalty)
    pred_final = enforce_min_duration(
        pred_vit,
        {0:3, 1:3, 2:5, 3:5, 4:5, 5:8}
    )

    # Output
    out = ann_df.iloc[:T].copy()
    out["tcn_pred"] = pred_raw
    out["tcn_pred_label"] = out["tcn_pred"].map(STATE_MAP)
    out["tcn_final"] = pred_final
    out["tcn_final_label"] = out["tcn_final"].map(STATE_MAP)
    out.to_csv(args.out_csv, index=False)

    print(f"[OK] saved {args.out_csv}")

if __name__ == "__main__":
    main()
