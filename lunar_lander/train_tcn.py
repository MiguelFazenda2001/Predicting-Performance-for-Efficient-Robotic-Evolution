import ast
import numpy as np
import pandas as pd
from tcn import EpisodeTCN, EpisodeDataset
from tqdm import tqdm

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

def data_processing():
    chunksize = 50_000  # tune for your RAM (50k–200k usually good)

    X_chunks = []
    y_chunks = []

    for chunk in pd.read_csv(DATA_CSV, chunksize=chunksize):
        # Parse arrays
        obs = chunk["observations"].apply(parse_array).to_numpy()
        acts = chunk["actions"].apply(parse_array).to_numpy()

        # Determine feature dimension once
        feat_dim = obs[0].shape[1] + acts[0].shape[1]

        # Allocate padded batch
        X_batch = np.zeros((len(chunk), MAX_LEN, feat_dim), dtype=np.float32)

        for i, (o, a) in enumerate(zip(obs, acts)):
            seq = np.concatenate([o, a], axis=1)
            length = min(len(seq), MAX_LEN)
            X_batch[i, :length] = seq[:length]

        y_batch = chunk[["success_rate", "avg_duration"]].to_numpy(dtype=np.float32)

        X_chunks.append(X_batch)
        y_chunks.append(y_batch)

    # Concatenate all chunks
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)

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

    train_loader = DataLoader(EpisodeDataset(X_train, y_train), batch_size=128, shuffle=True, pin_memory=True)

    val_loader = DataLoader(EpisodeDataset(X_val, y_val), batch_size=128, shuffle=False, pin_memory=True)

    return train_loader, val_loader, X


def train_tcn(data_loader, X, epochs=50):
    model = EpisodeTCN(input_dim=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    

    for epoch in range(epochs):
        model.train()
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for xb, yb in loop:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            # Update progress bar with current loss
            loop.set_postfix(loss=loss.item())

        metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: val_mse_success={metrics['mse_success']:.4f}, val_mse_duration={metrics['mse_duration']:.4f}")

    torch.save(model.state_dict(), "models/tcn_model.pth")
    
    return model    


if __name__ == "__main__":
    train_loader, val_loader, X = data_processing()
    model = train_tcn(train_loader, X)
    metrics = evaluate(model, val_loader)
    print(metrics)

