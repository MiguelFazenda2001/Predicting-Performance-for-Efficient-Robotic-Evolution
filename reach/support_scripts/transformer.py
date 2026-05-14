import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =====================================================
# Positional Encoding (scaled for stability)
# =====================================================
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1503):
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
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.pe[:, :x.size(1)]


# =====================================================
# Hierarchical Episode Transformer
# =====================================================
class EpisodeTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        num_episodes=5,
        max_time=512,
        d_model=256,
        nhead=8,
        num_temporal_layers=3,
        num_episode_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_episodes = num_episodes

        # -----------------------------------------
        # Input projection
        # -----------------------------------------
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(0.1)

        # -----------------------------------------
        # Embeddings (Option 3)
        # -----------------------------------------
        self.time_embedding = nn.Embedding(max_time, d_model)
        self.episode_embedding = nn.Embedding(num_episodes, d_model)

        # -----------------------------------------
        # CLS tokens
        # -----------------------------------------
        self.temporal_cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.episode_cls = nn.Parameter(torch.randn(1, 1, d_model))

        # -----------------------------------------
        # Temporal Transformer (inside episode)
        # -----------------------------------------
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_temporal_layers
        )

        # -----------------------------------------
        # Episode Transformer (across episodes)
        # -----------------------------------------
        episode_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.episode_transformer = nn.TransformerEncoder(
            episode_layer,
            num_layers=num_episode_layers
        )

        # -----------------------------------------
        # Output head
        # -----------------------------------------
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    # -------------------------------------------------
    def forward(self, x, mask):
        """
        x: (B,E,T,F)
        mask: (B,E,T)
        """

        B, E, T, F = x.shape

        # -----------------------------------------
        # Input projection
        # -----------------------------------------
        x = self.input_proj(x)
        #x = self.input_dropout(x)

        # -----------------------------------------
        # Add embeddings (Option 3)
        # -----------------------------------------
        time_ids = torch.arange(T, device=x.device)
        time_ids = time_ids.view(1, 1, T).expand(B, E, T)

        episode_ids = torch.arange(E, device=x.device)
        episode_ids = episode_ids.view(1, E, 1).expand(B, E, T)

        time_embed = self.time_embedding(time_ids)
        episode_embed = self.episode_embedding(episode_ids)

        x = x + time_embed + episode_embed

        # -----------------------------------------
        # Temporal transformer (per episode)
        # -----------------------------------------
        x = x.reshape(B * E, T, self.d_model)
        mask = mask.reshape(B * E, T)

        cls = self.temporal_cls.expand(B * E, -1, -1)

        x = torch.cat([cls, x], dim=1)

        mask = torch.cat([
            torch.ones(B * E, 1, device=x.device),
            mask
        ], dim=1)

        padding_mask = mask == 0

        temporal_out = self.temporal_transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        episode_repr = temporal_out[:, 0]  # CLS token

        # -----------------------------------------
        # Episode transformer (Option 4)
        # -----------------------------------------
        episode_repr = episode_repr.view(B, E, self.d_model)

        cls_ep = self.episode_cls.expand(B, -1, -1)

        episode_repr = torch.cat([cls_ep, episode_repr], dim=1)

        episode_out = self.episode_transformer(episode_repr)

        pooled = episode_out[:, 0]

        pooled = self.norm(pooled)

        return self.head(pooled)


# =====================================================
# Dataset
# =====================================================
class EpisodeDataset(Dataset):

    def __init__(self, X, y, M):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]