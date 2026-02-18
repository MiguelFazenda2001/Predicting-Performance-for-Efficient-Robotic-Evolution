import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class EpisodeTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # project features → transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # IMPORTANT
            norm_first=True     # more stable
        )

        self.pos_enc = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # attention pooling (better than mean)
        self.attn_pool = nn.Linear(d_model, 1)

        # heads
        self.success_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.duration_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # -------------------------
    # attention pooling
    # -------------------------
    def masked_attention_pool(self, x, mask):
        # x: (B,T,D)
        # mask: (B,T)

        scores = self.attn_pool(x).squeeze(-1)  # (B,T)

        scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return pooled

    # -------------------------
    def forward(self, x, mask):
        """
        x: (B,T,F)
        mask: (B,T)  (1 = valid, 0 = padding)
        """

        x = self.input_proj(x)

        x = self.pos_enc(x)

        # Transformer expects True = PAD
        key_padding_mask = (mask == 0)

        x = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        x = self.masked_attention_pool(x, mask)

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