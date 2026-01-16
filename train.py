#!/usr/bin/env python3
# It answers:
# Given DLC pose + kinematics + MoSeq features, how should the model map them to human-labeled behavior states?
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

# ----------------------------
# State mapping
# ----------------------------
STATE_MAP = {0:"turn", 1:"forward", 2:"still", 3:"explore", 4:"rear", 5:"groom", -1:"unsigned"}
VALID_CLASSES = [0, 1, 2, 3, 4, 5]

# ----------------------------
# Model (same as train.py)
# ----------------------------
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
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
    def __init__(self, in_features, n_classes=6, channels=128, levels=8, kernel_size=5, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, channels, kernel_size=1)
        blocks = []
        for i in range(levels):
            blocks.append(TCNBlock(channels, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(channels, n_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.blocks(x)
        logits = self.out_proj(x)      # (B, K, T)
        return logits.transpose(1, 2)  # (B, T, K)

# ----------------------------
# Data utilities (same logic as train.py)
# ----------------------------
def load_dlc_csv(dlc_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(dlc_csv_path, header=[0, 1, 2])
    if df.columns[0][0] == "scorer" and df.columns[0][1] == "bodyparts":
        df = df.drop(columns=df.columns[0])
    return df

def load_dlc_h5(path):
    """
    Load DeepLabCut .h5 output as pandas DataFrame.
    Columns are a MultiIndex: (scorer, bodypart, coord).
    """
    df = pd.read_hdf(path)

    # Some DLC versions include an index level name; normalize if needed
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
    allTrackData = m["allTrackData"]  # (1, N)
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
        names = [np.squeeze(allTrackData[0, i]["name"]).item() for i in range(allTrackData.shape[1])]
        raise ValueError(f"Could not find session '{session_prefix}' in MAT. Example names:\n" + "\n".join(map(str, names[:10])))

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
    dlc_conf_thr: float = 0.6,
    smooth_label_win: int = 7,
    include_moseq: bool = True,
    include_latents: bool = True,
    include_centroid_heading: bool = True,
):
    # Align lengths
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

    # Labels
    y = ann_df[label_col].to_numpy(dtype=np.int64)[:T]
    if smooth_label_win is not None and smooth_label_win >= 3:
        y = window_majority_labels(y, win=smooth_label_win, ignore_label=-1)

    mask = np.isin(y, VALID_CLASSES)
    return X, y, mask, feat_names

def compute_norm_stats(X: np.ndarray, eps: float = 1e-6):
    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    std = X.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std

# ----------------------------
# Dataset: sequence chunks
# ----------------------------
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
            torch.from_numpy(self.X[a:b]),      # (L, F)
            torch.from_numpy(self.y[a:b]),      # (L,)
            torch.from_numpy(self.mask[a:b]),   # (L,)
        )

def masked_ce_loss(logits, y, mask):
    # logits: (B, T, K)
    B, T, K = logits.shape
    logits2 = logits.reshape(B*T, K)
    y2 = y.reshape(B*T)
    m2 = mask.reshape(B*T)
    if m2.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits2[m2], y2[m2])

def temporal_tv_penalty(logits, weight=0.02):
    diff = logits[:, 1:] - logits[:, :-1]
    return weight * diff.abs().mean()

# ----------------------------
# Train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlc_h5", required=True)
    ap.add_argument("--ann_csv", required=True)
    ap.add_argument("--combined_mat", required=True)
    ap.add_argument("--session_prefix", default="sc04_d1_of")
    ap.add_argument("--label_col", default="human_labeled_state")
    ap.add_argument("--out_ckpt", required=True)

    ap.add_argument("--dlc_conf_thr", type=float, default=0.6)
    ap.add_argument("--smooth_label_win", type=int, default=9)

    ap.add_argument("--seq_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tv_weight", type=float, default=0.04)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    dlc_df = load_dlc_h5(args.dlc_h5)
    ann_df = pd.read_csv(args.ann_csv)

    speed, w, fps, mat_name = load_kinematics_from_combined_mat(args.combined_mat, args.session_prefix)
    print(f"[MAT] matched: {mat_name} | fps={fps} | speed_len={len(speed)} | w_len={len(w)}")

    X, y, mask, feat_names = build_feature_matrix(
        dlc_df, ann_df, speed, w,
        label_col=args.label_col,
        dlc_conf_thr=args.dlc_conf_thr,
        smooth_label_win=args.smooth_label_win,
    )
    print(f"[Data] T={len(X)} F={X.shape[1]} valid_frames={mask.sum()}")
    print("Label counts (raw):")
    print(pd.Series(ann_df["human_labeled_state"][:-1]).value_counts(dropna=False).sort_index())

    print("Label counts (used for training):")
    print(pd.Series(y[mask]).value_counts().sort_index())

    # Time split (no leakage)
    T = len(X)
    split = int(0.8 * T)
    X_tr, y_tr, m_tr = X[:split], y[:split], mask[:split]
    X_va, y_va, m_va = X[split:], y[split:], mask[split:]

    # Normalize using TRAIN stats; apply to both
    norm_mean, norm_std = compute_norm_stats(X_tr)
    X_tr = (X_tr - norm_mean) / norm_std
    X_va = (X_va - norm_mean) / norm_std

    train_ds = SequenceDataset(X_tr, y_tr, m_tr, seq_len=args.seq_len, stride=args.stride)
    val_ds   = SequenceDataset(X_va, y_va, m_va, seq_len=args.seq_len, stride=args.stride)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TCN(
        in_features=X.shape[1],
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

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xb)
                loss = masked_ce_loss(logits, yb, mb) + temporal_tv_penalty(logits, weight=args.tv_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        val_loss = eval_val()
        print(f"epoch {ep:02d} | val_loss=8{val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best:
            best = val_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "in_features": int(X.shape[1]),
                "channels": int(args.channels),
                "levels": int(args.levels),
                "kernel_size": int(args.kernel_size),
                "dropout": float(args.dropout),
                "feature_names": feat_names,
                "norm_mean": norm_mean.astype(np.float32),
                "norm_std": norm_std.astype(np.float32),
                "label_col": args.label_col,
                "session_prefix": args.session_prefix,
                "mat_name": mat_name,
                "fps": float(fps),
                "smooth_label_win": int(args.smooth_label_win),
                "dlc_conf_thr": float(args.dlc_conf_thr),
            }
            torch.save(ckpt, args.out_ckpt)
            print(f"[CKPT] saved best -> {args.out_ckpt}")

    print(f"[Done] best val_loss={best:.4f}")

if __name__ == "__main__":
    main()
