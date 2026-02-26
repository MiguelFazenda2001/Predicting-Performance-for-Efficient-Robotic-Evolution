import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

"""
Just because train_loader and val_loader are already created with the tcn class
"""
class EpisodeDataset(Dataset):
    def __init__(self, X, y, M):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.M = torch.tensor(M)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]    