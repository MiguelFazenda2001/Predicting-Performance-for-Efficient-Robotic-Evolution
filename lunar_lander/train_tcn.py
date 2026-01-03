import numpy as np
from tcn import EpisodeTCN
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DataProcessingH5 import DataProcessingH5
import json
from datetime import datetime

data_processing = DataProcessingH5()

SAVE_PATH = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def evaluate(model, loader, random_baseline=False):
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
    

def train_tcn(data_loader, X, val_loader, epochs=50, save_path=SAVE_PATH):
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

    torch.save(model.state_dict(), f"{save_path}/tcn_model.pth")
    
    return model    


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

if __name__ == "__main__":
    model_folder = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_path = f"models/{model_folder}_tcn"
    train_loader, val_loader, X = data_processing.data_processing_h5(save_path=save_path)
    model = train_tcn(train_loader, X, val_loader, save_path=save_path)
    metrics = evaluate(model, val_loader)
    print(metrics)
    log_metrics(metrics, path="results/metrics_log.jsonl")

