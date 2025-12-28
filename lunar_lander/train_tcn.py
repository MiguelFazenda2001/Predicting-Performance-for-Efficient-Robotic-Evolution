import ast
import numpy as np
import pandas as pd
from tcn import EpisodeTCN, EpisodeDataset
from tqdm import tqdm
import os
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


H5_PATH = "data/episodes.h5"
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

def data_processing_h5(h5_path=H5_PATH, max_len=MAX_LEN):
    X_list = []
    y_list = []

    with h5py.File(h5_path, "r") as h5:
        for evo in h5.values():
            for gen in evo.values():
                for genome in gen.values():
                    if "success_rate" not in genome.attrs:
                        continue

                    y = np.array(
                        [genome.attrs["success_rate"],
                         genome.attrs["avg_duration"]],
                        dtype=np.float32
                    )

                    for ep in genome.values():
                        obs = ep["observations"][:]
                        acts = ep["actions"][:]

                        seq = np.concatenate([obs, acts], axis=1)

                        # pad / truncate
                        if len(seq) >= max_len:
                            seq = seq[:max_len]
                        else:
                            pad = np.zeros((max_len - len(seq), seq.shape[1]), dtype=np.float32)
                            seq = np.vstack([seq, pad])

                        X_list.append(seq)
                        y_list.append(y)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # Normalize
    mean = X.mean(axis=(0,1), keepdims=True)
    std = X.std(axis=(0,1), keepdims=True) + 1e-6
    X = (X - mean) / std

    os.makedirs("models", exist_ok=True)
    np.save("models/x_mean.npy", mean)
    np.save("models/x_std.npy", std)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    train_loader = DataLoader(
        EpisodeDataset(X_train, y_train),
        batch_size=128,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        EpisodeDataset(X_val, y_val),
        batch_size=128,
        shuffle=False,
        pin_memory=True
    )

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
    train_loader, val_loader, X = data_processing_h5()
    model = train_tcn(train_loader, X)
    metrics = evaluate(model, val_loader)
    print(metrics)

