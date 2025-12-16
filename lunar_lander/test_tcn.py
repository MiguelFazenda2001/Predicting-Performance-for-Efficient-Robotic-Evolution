import ast
import random
import numpy as np
import pandas as pd
import torch

from tcn import EpisodeTCN, EpisodeDataset
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
DATA_CSV = "data/dataset.csv"
MODEL_PATH = "models/tcn_model.pth"
MEAN_PATH = "models/x_mean.npy"
STD_PATH = "models/x_std.npy"
MAX_LEN = 500
NUM_SAMPLES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- HELPERS ----------------
def parse_array(s):
    return np.array(ast.literal_eval(s), dtype=np.float32)


def pad_or_truncate(x, max_len):
    out = np.zeros((max_len, x.shape[1]), dtype=np.float32)
    length = min(len(x), max_len)
    out[:length] = x[:length]
    return out

def data_processing():
    df = pd.read_csv(DATA_CSV)
    sampled_df = df.sample(n=NUM_SAMPLES, random_state=None)


    obs = sampled_df["observations"].apply(parse_array).to_numpy()
    acts = sampled_df["actions"].apply(parse_array).to_numpy()

    # Determine feature dimension once
    feat_dim = obs[0].shape[1] + acts[0].shape[1]

    # Allocate padded batch
    X = np.zeros((len(sampled_df), MAX_LEN, feat_dim), dtype=np.float32)

    for i, (o, a) in enumerate(zip(obs, acts)):
        seq = np.concatenate([o, a], axis=1)
        length = min(len(seq), MAX_LEN)
        X[i, :length] = seq[:length]

    y = sampled_df[["success_rate", "avg_duration"]].to_numpy(dtype=np.float32)

    # Normalize inputs
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)

    X = (X - mean) / std

    data_loader = DataLoader(EpisodeDataset(X, y), batch_size=128, shuffle=True, pin_memory=True)

    return data_loader, X

# ---------------- LOAD NORMALIZATION ----------------
mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)


# ---------------- LOAD MODEL ----------------
# infer input_dim from normalization stats


data_loader, X = data_processing()

model = EpisodeTCN(input_dim=X.shape[2]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ---------------- LOAD & SAMPLE DATA ----------------

print("\n=== TCN Episode Predictions ===\n")

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

np.set_printoptions(precision=3, suppress=True)

print(y_true)
print("\n--- Predictions ---\n")
print(y_pred)