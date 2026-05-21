import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

"""
dropout = 0.1
"""

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, kernel=3, dropout=0.1):
        super().__init__()

        padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel,
                                padding=padding,
                                dilation=dilation)
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel,
                                padding=padding,
                                dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)

        self.dropout = nn.Dropout(dropout)

        #if channels don't match, use 1x1 conv to match them
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]
        out = F.relu(self.norm1(out))

        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = F.relu(self.norm2(out))

        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)

        return F.relu(out + res)

class ResidualEpisodeTCN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.tcn = nn.Sequential(
            ResidualTCNBlock(input_dim, 64, dilation=1),
            ResidualTCNBlock(64, 64, dilation=2),
            ResidualTCNBlock(64, 128, dilation=4),
        )
        
        #self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.success_head = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1),
                                        nn.Sigmoid())
        
        self.duration_head = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1))

    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(1)  # (B, 1, T)
        x = x * mask  # zero out masked positions
        sum_x = x.sum(dim=2)  # sum over time
        count = mask.sum(dim=2)  # count of valid positions
        return sum_x / (count + 1e-8)  # avoid division by zero    

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.tcn(x)
        x = self.masked_mean(x,mask)
        s = self.success_head(x)
        d = self.duration_head(x)
        return torch.cat([s, d], dim=1)


class EpisodeDataset(Dataset):
    def __init__(self, X, y, M):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.M = torch.tensor(M)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]    