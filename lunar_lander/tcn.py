import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              padding=padding,
                              dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]]
        return F.relu(self.norm(x))
    
class EpisodeTCN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_dim, 64, dilation=1),
            TCNBlock(64, 64, dilation=2),
            TCNBlock(64, 128, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)    

class EpisodeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]    