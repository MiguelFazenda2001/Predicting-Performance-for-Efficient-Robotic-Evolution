import numpy as np
from tcn import ResidualEpisodeTCN
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DataProcessingH5 import DataProcessingH5
import json
from datetime import datetime
import os
import shutil
import torch.nn.functional as F

"""
Weigthed MSE Loss (success = 1, duration = 0.1)
75 epochs
"""

data_processing = DataProcessingH5() #Padding needs mask so tcn doesnt avgPool with zeros

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

def train_residualtcn(data_loader, input_dim, val_loader, epochs=75, save_path=SAVE_PATH):
    model = ResidualEpisodeTCN(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    w_success = 1.0
    w_duration = 0.1

    val_losses = []
    losses = []
    mse_success = []
    mse_duration = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        episode_loss = 0.0

        for xb, yb, mb in loop:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb, mb)
            loss_success = F.mse_loss(pred[:, 0], yb[:, 0])
            loss_duration = F.mse_loss(pred[:, 1], yb[:, 1])

            loss = (
                w_success * loss_success +
                w_duration * loss_duration
            )

            loss.backward()
            optimizer.step()

            episode_loss += loss.item()

            # Update progress bar with current loss
            loop.set_postfix(loss=loss.item())

        metrics = evaluate(model, val_loader, w_success=w_success, w_duration=w_duration)
        print(f"Epoch {epoch+1}: val_mse_success={metrics['mse_success']:.4f}, val_mse_duration={metrics['mse_duration']:.4f}")

        val_losses.append(metrics['val_loss'])
        losses.append(episode_loss / len(data_loader))
        mse_success.append(metrics['mse_success'])
        mse_duration.append(metrics['mse_duration'])
    torch.save(model.state_dict(), f"{save_path}/residual_tcn_model.pth")
    
    return model, val_losses, losses, mse_success, mse_duration


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
    if not os.path.exists(f"data/train_loader.pth") or not os.path.exists(f"data/val_loader.pth") or not os.path.exists(f"data/input_dim.json"):
        return data_processing.data_processing_h5(save_path=save_path)

    train_loader = torch.load(f"data/train_loader.pth", weights_only=False)
    val_loader = torch.load(f"data/val_loader.pth", weights_only=False)
    with open(f"data/input_dim.json", "r") as f:
        input_dim = json.load(f)

    try:
        os.makedirs(save_path, exist_ok=True)
        shutil.copy(f"data/x_mean.npy", f"{save_path}/x_mean.npy")
        shutil.copy(f"data/x_std.npy", f"{save_path}/x_std.npy")
    except FileNotFoundError:
        print("Warning: x_mean.npy or y_mean.npy not found in data directory.")

    return train_loader, val_loader, input_dim

if __name__ == "__main__":
    model_folder = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_path = f"models/{model_folder}_tcn"

    train_loader, val_loader, input_dim = load_data(save_path=save_path)

    save_path = f"models/{model_folder}_tcn"
    episode_model, val_losses, losses, mse_success, mse_duration = train_residualtcn(train_loader, input_dim, val_loader, save_path=save_path)
    plot_training_curves(val_losses, losses, mse_success, mse_duration, save_path=save_path, model_name="residualtcn")
    #metrics = evaluate(episode_model, val_loader)
    #print(metrics)

