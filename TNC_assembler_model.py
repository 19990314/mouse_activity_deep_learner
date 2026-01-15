import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_ce_loss(logits, y, mask):
    """
    logits: (B, T, K)
    y: (B, T)
    mask: (B, T) bool
    """
    B, T, K = logits.shape
    logits2 = logits.reshape(B*T, K)
    y2 = y.reshape(B*T)
    m2 = mask.reshape(B*T)

    if m2.sum() == 0:
        return logits.sum() * 0.0

    return F.cross_entropy(logits2[m2], y2[m2])

def temporal_tv_penalty(logits, weight=0.02):
    # logits: (B, T, K)
    diff = logits[:, 1:] - logits[:, :-1]
    return weight * diff.abs().mean()



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
        # x: (B, C, T)
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = F.gelu(y)
        y = self.dropout(y)

        return x + y  # residual

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
        logits = self.out_proj(x)  # (B, K, T)
        return logits.transpose(1, 2)  # (B, T, K)






import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from data_and_features import load_dlc_csv, load_annotations_csv, build_feature_matrix
from tcn import TCN

def train_one_session(
    dlc_csv, ann_csv,
    label_col="human_labeled_state",
    seq_len=256, stride=128,
    epochs=20, lr=2e-4,
    smooth_label_win=7,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dlc_df = load_dlc_csv(dlc_csv)
    ann_df = load_annotations_csv(ann_csv)

    X, y, mask, feat_names = build_feature_matrix(
        dlc_df, ann_df,
        label_col=label_col,
        smooth_label_win=smooth_label_win,
        speed=None, angvel=None,  # add later if desired
    )

    # Split train/val by time to avoid leakage
    T = len(X)
    split = int(0.8 * T)
    X_tr, y_tr, m_tr = X[:split], y[:split], mask[:split]
    X_va, y_va, m_va = X[split:], y[split:], mask[split:]

    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from sequence_dataset import SequenceDataset, masked_ce_loss, temporal_tv_penalty  # if you split files

    train_ds = SequenceDataset(X_tr, y_tr, m_tr, seq_len=seq_len, stride=stride)
    val_ds   = SequenceDataset(X_va, y_va, m_va, seq_len=seq_len, stride=stride)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = TCN(in_features=X.shape[1], n_classes=6, channels=128, levels=8, kernel_size=5, dropout=0.1).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    def eval_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb, mb in val_dl:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                logits = model(xb)
                loss = masked_ce_loss(logits, yb, mb)
                losses.append(loss.item())
        return float(sum(losses) / max(1, len(losses)))

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                logits = model(xb)
                loss = masked_ce_loss(logits, yb, mb) + temporal_tv_penalty(logits, weight=0.02)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        va = eval_loss()
        print(f"epoch {ep:02d} | val_loss={va:.4f}")

    return model, feat_names
