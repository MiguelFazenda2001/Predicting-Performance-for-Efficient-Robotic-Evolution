import numpy as np
from tcn import EpisodeTCN, ResidualEpisodeTCN
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DataProcessingH5 import DataProcessingH5
import json
from datetime import datetime

data_processing = DataProcessingH5() #Padding needs mask so tcn doesnt avgPool with zeros

SAVE_PATH = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def evaluate(model, loader, random_baseline=False):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            p = model(xb)
            preds.append(p.cpu().numpy())
            ys.append(yb.cpu().numpy())

    y_true = np.vstack(ys)
    y_pred = np.vstack(preds)

    metrics = {
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
    

def train_episodetcn(data_loader, input_dim, val_loader, epochs=50, save_path=SAVE_PATH):
    model = EpisodeTCN(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    mse_success = []
    mse_duration = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for xb, yb, mb in loop:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            # Update progress bar with current loss
            loop.set_postfix(loss=loss.item())

        metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: val_mse_success={metrics['mse_success']:.4f}, val_mse_duration={metrics['mse_duration']:.4f}")

        losses.append(loss.item())
        mse_success.append(metrics['mse_success'])
        mse_duration.append(metrics['mse_duration'])
    torch.save(model.state_dict(), f"{save_path}/tcn_model.pth")
    
    return model, losses, mse_success, mse_duration

def train_residualtcn(data_loader, input_dim, val_loader, epochs=50, save_path=SAVE_PATH):
    model = ResidualEpisodeTCN(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    mse_success = []
    mse_duration = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for xb, yb, mb in loop:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            # Update progress bar with current loss
            loop.set_postfix(loss=loss.item())

        metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: val_mse_success={metrics['mse_success']:.4f}, val_mse_duration={metrics['mse_duration']:.4f}")

        losses.append(loss.item())
        mse_success.append(metrics['mse_success'])
        mse_duration.append(metrics['mse_duration'])
    torch.save(model.state_dict(), f"{save_path}/tcn_model.pth")
    
    return model, losses, mse_success, mse_duration


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

def plot_training_curves(losses, mse_success, mse_duration, save_path=SAVE_PATH):
    import matplotlib.pyplot as plt

    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label="Training Loss")
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
    plt.savefig(f"{save_path}/training_curves.png")
    plt.close()

if __name__ == "__main__":
    model_folder = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_path = f"models/{model_folder}_episodetcn"
    train_loader, val_loader, input_dim = data_processing.data_processing_h5(save_path=save_path)
    episode_model, losses, mse_success, mse_duration = train_episodetcn(train_loader, input_dim, val_loader, save_path=save_path)
    plot_training_curves(losses, mse_success, mse_duration, save_path=save_path)
    metrics = evaluate(episode_model, val_loader)
    print(metrics)

    save_path = f"models/{model_folder}_residualtcn"
    episode_model, losses, mse_success, mse_duration = train_residualtcn(train_loader, input_dim, val_loader, save_path=save_path)
    plot_training_curves(losses, mse_success, mse_duration, save_path=save_path)
    metrics = evaluate(episode_model, val_loader)
    print(metrics)
    #log_metrics(metrics, path="results/metrics_log.jsonl")

