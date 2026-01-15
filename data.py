import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, X, y, mask, seq_len=256, stride=128):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mask = mask.astype(bool)

        self.seq_len = int(seq_len)
        self.indices = []
        T = len(X)
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            # keep windows that have at least some valid labels
            if self.mask[start:end].any():
                self.indices.append((start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        a, b = self.indices[i]
        x = torch.from_numpy(self.X[a:b])           # (L, F)
        y = torch.from_numpy(self.y[a:b])           # (L,)
        m = torch.from_numpy(self.mask[a:b])        # (L,)
        return x, y, m
