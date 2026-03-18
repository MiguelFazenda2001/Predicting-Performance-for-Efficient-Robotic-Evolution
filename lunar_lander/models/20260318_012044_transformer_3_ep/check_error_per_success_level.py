import numpy as np
from transformer import EpisodeTransformer
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime
import os
import torch.nn.functional as F

SAVE_PATH = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def evaluate(model, loader, w_success, w_duration, random_baseline=False):
    model.eval()
    ys, preds = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            p = model(xb, mb)

            loss_success = F.mse_loss(p[:, 0], yb[:, 0])
            loss_duration = F.mse_loss(p[:, 1], yb[:, 1])
            loss = (
                w_success * loss_success +
                w_duration * loss_duration
            )

            total_loss += loss.item()
            n_batches += 1

            preds.append(p.cpu().numpy())
            ys.append(yb.cpu().numpy())

    y_true = np.vstack(ys)
    y_pred = np.vstack(preds)

    val_loss = total_loss / n_batches

    metrics = {
        "val_loss": val_loss,
        "mse_success": mean_squared_error(y_true[:,0], y_pred[:,0]),
        "mse_duration": mean_squared_error(y_true[:,1], y_pred[:,1]),
        "mae_success": mean_absolute_error(y_true[:,0], y_pred[:,0]),
        "mae_duration": mean_absolute_error(y_true[:,1], y_pred[:,1]),
        "r2_success": r2_score(y_true[:,0], y_pred[:,0]),
        "r2_duration": r2_score(y_true[:,1], y_pred[:,1]),
    }


    # unique success-rate values
    unique_success = np.unique(y_true[:, 0])

    mse_per_success = {}

    print("\nMSE per success rate:")
    for s in unique_success:
        mask = y_true[:, 0] == s

        mse_s = mean_squared_error(
            y_true[mask, 0],
            y_pred[mask, 0]
        )

        mse_per_success[s] = mse_s
        print(f"Success {s:.3f} -> MSE: {mse_s:.6f}")


    return metrics

def load_data():
    if not os.path.exists(f"../../data/3_ep/train_loader.pth") or not os.path.exists(f"../../data/3_ep/val_loader.pth") or not os.path.exists(f"../../data/3_ep/input_dim.json"):
        return data_processing.data_processing_h5(save_path=save_path)

    train_loader = torch.load(f"../../data/3_ep/train_loader.pth", weights_only=False)
    val_loader = torch.load(f"../../data/3_ep/val_loader.pth", weights_only=False)
    with open(f"../../data/3_ep/input_dim.json", "r") as f:
        input_dim = json.load(f)

    return train_loader, val_loader, input_dim

if __name__ == "__main__":
    # Load validation data
    _, val_loader, input_dim = load_data()

    model = EpisodeTransformer(input_dim=input_dim)
    model.load_state_dict(torch.load("best_validation_transformer_model.pth"))
    model.to(device)

    # Evaluate the model
    w_success = 1.0
    w_duration = 0.1

    metrics = evaluate(model, val_loader, w_success, w_duration, random_baseline=True)

    print("\nFinal Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")