import numpy as np
from transformer import EpisodeTransformer
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from DataProcessingH5_v2 import DataProcessingH5
import json
from datetime import datetime
import os
import shutil
import torch.nn.functional as F
import math

"""
Weigthed MSE Loss (success = 1, duration = 1.0)
50 epochs
"""

#data_processing = DataProcessingH5() #Padding needs mask so tcn doesnt avgPool with zeros

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
            loss = w_success * loss_success + w_duration * loss_duration

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

    if random_baseline:
        n = len(y_true)

        # RANDOM SUCCESS RATE (DISCRETE)
        success_values = np.arange(0.0, 1.01, 0.1)
        rand_success = np.random.choice(success_values, size=n)

        # RANDOM DURATION (MATCH SCALE)
        dur_min, dur_max = y_true[:,1].min(), y_true[:,1].max()
        rand_duration = np.random.uniform(dur_min, dur_max, size=n)

        y_rand = np.stack([rand_success, rand_duration], axis=1)

        metrics.update({
            "rand_mse_success": mean_squared_error(y_true[:,0], y_rand[:,0]),
            "rand_mse_duration": mean_squared_error(y_true[:,1], y_rand[:,1]),
            "rand_mae_success": mean_absolute_error(y_true[:,0], y_rand[:,0]),
            "rand_mae_duration": mean_absolute_error(y_true[:,1], y_rand[:,1]),
            "rand_r2_success": r2_score(y_true[:,0], y_rand[:,0]),
            "rand_r2_duration": r2_score(y_true[:,1], y_rand[:,1]),
        })

    return metrics

def train_transformer(
    train_loader,
    val_loader,
    input_dim,
    epochs=50,
    save_path="model.pth"
):

    model = EpisodeTransformer(input_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    w_success = 1.0
    w_duration = 0.1

    train_losses = []
    val_losses = []
    mse_success = []
    mse_duration = []

    best_val = float("inf")

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for xb, yb, mb in loop:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad()

            pred = model(xb, mb)

            loss_s = F.mse_loss(pred[:, 0], yb[:, 0])
            loss_d = F.mse_loss(pred[:, 1], yb[:, 1])

            loss = w_success * loss_s + w_duration * loss_d

            loss.backward()

            # IMPORTANT for transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        metrics = evaluate(model, val_loader,
                           w_success, w_duration)

        print(
            f"Epoch {epoch+1} | "
            f"Val Success MSE: {metrics['mse_success']:.4f} | "
            f"Val Duration MSE: {metrics['mse_duration']:.4f}"
        )

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(metrics["val_loss"])
        mse_success.append(metrics["mse_success"])
        mse_duration.append(metrics["mse_duration"])

        # save best model
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            torch.save(model.state_dict(), f"{save_path}/best_validation_transformer_model.pth")

    torch.save(model.state_dict(), f"{save_path}/best_training_transformer_model.pth")        

    return model, val_losses, train_losses, mse_success, mse_duration

def log_metrics(metrics, path="results/metrics_log.jsonl"):
    """
    Append metrics to a JSONL file (one JSON object per line).
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        **metrics
    }

    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def plot_training_curves(val_losses, losses, mse_success, mse_duration, save_path=SAVE_PATH, model_name=None):
    import matplotlib.pyplot as plt

    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, mse_success, label="Validation MSE Success", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Success")
    plt.title("Validation MSE Success Curve")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, mse_duration, label="Validation MSE Duration", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Duration")
    plt.title("Validation MSE Duration Curve")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_training_curves.png")
    plt.close()


def load_data(save_path=SAVE_PATH):
    if not os.path.exists(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/train_loader.pth") or not os.path.exists(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/val_loader.pth") or not os.path.exists(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/input_dim.json"):
        return data_processing.data_processing_h5(save_path=save_path)

    train_loader = torch.load(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/train_loader.pth", weights_only=False)
    val_loader = torch.load(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/val_loader.pth", weights_only=False)
    with open(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/input_dim.json", "r") as f:
        input_dim = json.load(f)

    try:
        os.makedirs(save_path, exist_ok=True)
        shutil.copy(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/x_mean.npy", f"{save_path}/x_mean.npy")
        shutil.copy(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/x_std.npy", f"{save_path}/x_std.npy")
        shutil.copy(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/y_mean.npy", f"{save_path}/y_mean.npy")
        shutil.copy(f"/mnt/DATA/miguelfazenda/reach/dataset/dataloaders/3_ep/y_std.npy", f"{save_path}/y_std.npy")
    except FileNotFoundError:
        print("Warning: x_mean.npy or y_mean.npy not found in data directory.")

    return train_loader, val_loader, input_dim

if __name__ == "__main__":
    model_folder = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_path = f"models/{model_folder}_transformer_3_ep"

    train_loader, val_loader, input_dim = load_data(save_path=save_path)

    episode_model, val_losses, losses, mse_success, mse_duration = train_transformer(train_loader, val_loader, input_dim, save_path=save_path)
    plot_training_curves(val_losses, losses, mse_success, mse_duration, save_path=save_path, model_name="transformer")
    #metrics = evaluate(episode_model, val_loader)
    #print(metrics)

