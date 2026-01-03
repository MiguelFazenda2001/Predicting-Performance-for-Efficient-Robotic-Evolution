import numpy as np
import h5py
import torch

from tcn import EpisodeTCN, EpisodeDataset
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
H5_PATH = "data/episodes.h5"
MODEL_PATH = "models/tcn_model.pth"
MEAN_PATH = "models/x_mean.npy"
STD_PATH = "models/x_std.npy"

MAX_LEN = 500
SAMPLES_PER_SUCCESS = 5

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
                        seq = pad_or_truncate(seq, MAX_LEN)

                        success_to_samples.setdefault(success_rate, []).append(
                            (seq, y)
                        )

    # ---- SAMPLE EXACTLY 5 PER SUCCESS RATE ----
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

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # ---- NORMALIZE ----
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    X = (X - mean) / std

    data_loader = DataLoader(
        EpisodeDataset(X, y),
        batch_size=128,
        shuffle=False,   # keep order for inspection
        pin_memory=True
    )

    return data_loader, X, y


# ---------------- LOAD DATA ----------------
data_loader, X, y_true = data_processing_h5()


# ---------------- LOAD MODEL ----------------
model = EpisodeTCN(input_dim=X.shape[2]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ---------------- INFERENCE ----------------
ys, preds = [], []

with torch.no_grad():
    for xb, yb in data_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        p = model(xb)
        preds.append(p.cpu().numpy())
        ys.append(yb.cpu().numpy())

y_true = np.vstack(ys)
y_pred = np.vstack(preds)


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