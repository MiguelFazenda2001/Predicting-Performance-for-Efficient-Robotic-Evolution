import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Scaled Positional Encoding (more stable)
# =====================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=501):
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

        # learnable scaling (VERY important)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.pe[:, :x.size(1)]


# =====================================================
# Episode Transformer v2
# =====================================================
class EpisodeTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        # FULL transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # learned query token (decoder input)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    # -------------------------------------------------
    def forward(self, x, mask):
        """
        x: (B,T,F)
        mask: (B,T)
        """

        B = x.size(0)

        x = self.input_proj(x)
        x = self.pos_enc(x)

        # True = padding
        src_key_padding_mask = (mask == 0)

        # encoder memory
        memory = self.transformer.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # decoder query
        query = self.query_token.expand(B, -1, -1)

        out = self.transformer.decoder(
            tgt=query,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask
        )

        out = out[:, 0]  # query output

        out = self.norm(out)

        return self.head(out)

class EpisodeDataset(Dataset):
    def __init__(self, X, y, M):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.M = torch.tensor(M)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]    