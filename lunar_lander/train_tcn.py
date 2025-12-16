import ast
import numpy as np
import pandas as pd
from tcn import EpisodeTCN, EpisodeDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


DATA_CSV = "data/dataset.csv"
MAX_LEN = 500

def parse_array(s):
    return np.array(ast.literal_eval(s), dtype=np.float32)

def pad_or_truncate(x, max_len):
    if len(x) >= max_len:
        return x[:max_len]
    pad = np.zeros((max_len - len(x), x.shape[1]), dtype=np.float32)
    return np.vstack([x, pad])

def evaluate(model, loader):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            p = model(xb)
            preds.append(p.cpu().numpy())
            ys.append(yb.cpu().numpy())

    y_true = np.vstack(ys)
    y_pred = np.vstack(preds)

    return {
        "mse_success": mean_squared_error(y_true[:,0], y_pred[:,0]),
        "mse_duration": mean_squared_error(y_true[:,1], y_pred[:,1]),
        "mae_success": mean_absolute_error(y_true[:,0], y_pred[:,0]),
        "mae_duration": mean_absolute_error(y_true[:,1], y_pred[:,1]),
        "r2_success": r2_score(y_true[:,0], y_pred[:,0]),
        "r2_duration": r2_score(y_true[:,1], y_pred[:,1]),
    }

# Load dataset
df = pd.read_csv(DATA_CSV)
df["obs"] = df["observations"].apply(parse_array)
df["acts"] = df["actions"].apply(parse_array)

# Prepare input and output arrays
X_list, y_list = [], []

for _, row in df.iterrows():
    X = np.concatenate([row["obs"], row["acts"]], axis=1)
    y = np.array([
        row["success_rate"],
        row["avg_duration"]
    ], dtype=np.float32)

    X_list.append(X)
    y_list.append(y)

X = np.stack([pad_or_truncate(x, MAX_LEN) for x in X_list])
y = np.stack(y_list)

# Normalize inputs
mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-6
X = (X - mean) / std

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_loader = DataLoader(EpisodeDataset(X_train, y_train), batch_size=64, shuffle=True, pin_memory=True)

val_loader = DataLoader(EpisodeDataset(X_val, y_val), batch_size=64, shuffle=False, pin_memory=True)

model = EpisodeTCN(input_dim=X.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    total = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item()

    print(f"Epoch {epoch}: {total / len(train_loader):.4f}")

metrics = evaluate(model, val_loader)
print(metrics)
torch.save(model.state_dict(), "models/tcn_model.pth")
