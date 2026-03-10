import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Scaled Positional Encoding (more stable)
# =====================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
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
        num_episodes=2,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.2,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_episodes = num_episodes

        # project input features
        self.input_proj = nn.Linear(input_dim, d_model)

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # episode embeddings (like BERT token type embeddings)
        self.episode_embedding = nn.Embedding(num_episodes, d_model)

        """
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        """
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )


        self.norm = nn.LayerNorm(d_model)

        # prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    # -------------------------------------------------
    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)

    # -------------------------------------------------
    def forward(self, x, mask):
        """
        x: (B,E,T,F)
        mask: (B,E,T)
        """

        B, E, T, F = x.shape

        # ----------------------------------------
        # project features
        # ----------------------------------------
        x = self.input_proj(x)

        # ----------------------------------------
        # add episode embeddings
        # ----------------------------------------
        episode_ids = torch.arange(E, device=x.device)
        episode_ids = episode_ids.view(1, E, 1).expand(B, E, T)

        ep_embed = self.episode_embedding(episode_ids)

        x = x + ep_embed

        # ----------------------------------------
        # flatten episodes
        # ----------------------------------------
        x = x.reshape(B, E*T, self.d_model)
        mask = mask.reshape(B, E*T)

        # positional encoding
        x = self.pos_enc(x)

        padding_mask = (mask == 0)

        # transformer
        memory = self.transformer.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        query = x[:, :E, :]  # use first token of each episode as query

        out = self.transformer.decoder(
            tgt=query,  # use first token of each episode as query
            memory=memory,
            memory_key_padding_mask=padding_mask
        )

        # pooling
        pooled = self.masked_mean(memory, mask)

        pooled = self.norm(pooled)

        return self.head(pooled)

class EpisodeDataset(Dataset):
    def __init__(self, X, y, M):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.M = torch.tensor(M)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]    