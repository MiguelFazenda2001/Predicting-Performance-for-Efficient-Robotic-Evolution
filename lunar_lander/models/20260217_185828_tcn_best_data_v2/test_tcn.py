import numpy as np
import h5py
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tcn import ResidualEpisodeTCN, EpisodeDataset
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
H5_PATH = "../../data/test_episodes.h5"
DEFAULT_PATH = ""#"models/20260217_185828_tcn_best_data_v2"
MODEL_PATH = f"{DEFAULT_PATH}residual_tcn_model.pth"
X_MEAN_PATH = f"{DEFAULT_PATH}x_mean.npy"
X_STD_PATH = f"{DEFAULT_PATH}x_std.npy"
Y_MEAN_PATH = f"{DEFAULT_PATH}y_mean.npy"
Y_STD_PATH = f"{DEFAULT_PATH}y_std.npy"

MAX_LEN = 500
SAMPLES_PER_SUCCESS = 0# 0 means use all samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- HELPERS ----------------
def pad_or_truncate(seq, max_len):
    out = np.zeros((max_len, seq.shape[1]), dtype=np.float32)
    length = min(len(seq), max_len)
    out[:length] = seq[:length]
    return out


# ---------------- DATA PROCESSING (H5) ----------------
def data_processing_h5():
    X_list = []
    y_list = []
    M_list = []

    # collect episodes grouped by success_rate
    success_to_samples = {}

    with h5py.File(H5_PATH, "r") as h5:
        for evo in h5.values():
            for gen in evo.values():
                for genome in gen.values():

                    if "success_rate" not in genome.attrs:
                        continue

                    success_rate = float(genome.attrs["success_rate"])
                    avg_duration = float(genome.attrs["avg_duration"])
                    y = np.array([success_rate, avg_duration], dtype=np.float32)

                    for ep in genome.values():
                        obs = ep["observations"][:]
                        acts = ep["actions"][:]

                        seq = np.concatenate([obs, acts], axis=1)

                        mask = np.ones(MAX_LEN, dtype=np.float32)

                        length = len(seq)

                        # pad / truncate
                        if length >= MAX_LEN:
                            seq = seq[:MAX_LEN]
                            mask = mask[:MAX_LEN]
                        else:
                            pad = np.zeros((MAX_LEN - length, seq.shape[1]), dtype=np.float32)
                            seq = np.vstack([seq, pad])
                            mask[length:] = 0.0
                            
                        if SAMPLES_PER_SUCCESS != 0:
                            success_to_samples.setdefault(success_rate, []).append(
                                (seq, y, mask)
                            )
                        else:
                            X_list.append(seq)
                            y_list.append(y)
                            M_list.append(mask)    

    if SAMPLES_PER_SUCCESS != 0:
        # ---- SAMPLE EXACTLY n PER SUCCESS RATE ----
        for rate in sorted(success_to_samples.keys()):
            samples = success_to_samples[rate]

            if len(samples) < SAMPLES_PER_SUCCESS:
                raise ValueError(
                    f"Not enough episodes for success_rate={rate}: "
                    f"needed {SAMPLES_PER_SUCCESS}, found {len(samples)}"
                )

            chosen = np.random.choice(
                len(samples), SAMPLES_PER_SUCCESS, replace=False
            )

            for idx in chosen:
                X_list.append(samples[idx][0])
                y_list.append(samples[idx][1])
                M_list.append(samples[idx][2])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    M = np.asarray(M_list, dtype=np.float32)

    # ---- NORMALIZE ----
    mean = np.load(X_MEAN_PATH)
    std = np.load(X_STD_PATH)

    mask_exp = M[..., None]  # (B,T,1)

    count = mask_exp.sum(axis=(0,1))

    X = (X - mean) / std
    X *= mask_exp 

    y_mean = np.load(Y_MEAN_PATH)
    y_std = np.load(Y_STD_PATH)

    y[:,1] = (y[:,1] - y_mean) / y_std

    data_loader = DataLoader(
        EpisodeDataset(X, y, M),
        batch_size=128,
        shuffle=False,   # keep order for inspection
        pin_memory=True
    )

    return data_loader, X, y, M


def save_line_comparison_png(y_true, y_pred, save_path="prediction_vs_gt.png"):
    """
    Saves a single PNG with two vertically stacked plots:
    - Top: Success Rate
    - Bottom: Average Duration
    """

    assert y_true.shape == y_pred.shape, "Shapes must match"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x = np.arange(len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    titles = ["Success Rate", "Average Duration"]

    for i in range(2):
        axes[i].plot(x, y_true[:, i], label="Ground Truth", linewidth=2)
        axes[i].plot(x, y_pred[:, i], linestyle="--", label="Prediction", linewidth=2)

        axes[i].set_ylabel(titles[i])
        axes[i].set_title(f"{titles[i]}: Ground Truth vs Prediction")
        axes[i].legend()
        axes[i].grid(True)

    axes[1].set_xlabel("Sample Index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved stacked comparison plot to: {save_path}")


# ---------------- LOAD DATA ----------------
data_loader, X, y_true, M = data_processing_h5()


# ---------------- LOAD MODEL ----------------
model = ResidualEpisodeTCN(input_dim=X.shape[2]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ---------------- INFERENCE ----------------
ys, preds = [], []

with torch.no_grad():
    for xb, yb, mb in data_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        p = model(xb, mb)
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

print("\n=== TCN Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print(len(y_true), "samples evaluated.")

"""
# ---------------- OUTPUT ----------------
np.set_printoptions(precision=3, suppress=True)

print("\n=== TCN Predictions (H5 | 5 per success rate) ===\n")
print("Ground truth:")
print(y_true)

print("\nPredictions:")
print(y_pred)


# ---------------- PER-SUCCESS-RATE VIEW ----------------
for rate in np.unique(y_true[:, 0]):
    idxs = np.where(y_true[:, 0] == rate)[0]
    print(f"\nSuccess rate {rate}:")
    for i in idxs:
        print(
            f"  true={y_true[i]}  pred={y_pred[i]}"
        )

save_line_comparison_png(
    y_true,
    y_pred,
    save_path=f"{DEFAULT_PATH}/tcn_line_comparison.png"
)
"""