import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tcn import EpisodeDataset

H5_PATH = "/mnt/DATA/miguelfazenda/25_evo_episodes.h5"
SAVE_PATH = "../data"
MAX_LEN = 500


class DataProcessingH5:
    def __init__(self):
        pass

    def data_processing_h5(self,save_path=SAVE_PATH, h5_path=H5_PATH, max_len=MAX_LEN):
        rate_to_locations = {}

        with h5py.File(h5_path, "r") as h5:
            for evo in h5.values():
                for gen in evo.values():
                    for genome in gen.values():

                        if "success_rate" not in genome.attrs:
                            continue

                        success = float(genome.attrs["success_rate"])

                        episodes = list(genome.values())

                        pair_count = len(episodes) - 1

                        for pair_idx in range(pair_count):
                            rate_to_locations.setdefault(success, []).append(
                                (genome.name, pair_idx)
                            )

        min_count = min(len(v) for v in rate_to_locations.values())

        rng = np.random.default_rng(42)

        selected = {}
        for rate, locs in rate_to_locations.items():
            idx = rng.choice(len(locs), min_count, replace=False)
            selected[rate] = [locs[i] for i in idx]

        print("Balanced samples per success_rate:", min_count)

        # ---------- PASS 2: load only selected ----------
        X_list, y_list, mask_list = [], [], []

        with h5py.File(h5_path, "r") as h5:

            genome_cache = {}

            for rate, locs in selected.items():

                for genome_name, pair_idx in locs:

                    genome = h5[genome_name]

                    y = np.array([
                        genome.attrs["success_rate"],
                        genome.attrs["avg_duration"]
                    ], dtype=np.float32)

                    episodes = list(genome.values())

                    ep1 = episodes[pair_idx]
                    ep2 = episodes[pair_idx+1]

                    obs1 = ep1["observations"][:]
                    acts1 = ep1["actions"][:]
                    seq1 = np.concatenate([obs1, acts1], axis=1)

                    obs2 = ep2["observations"][:]
                    acts2 = ep2["actions"][:]
                    seq2 = np.concatenate([obs2, acts2], axis=1)

                    mask = np.ones(max_len, dtype=np.float32)

                    length = min(len(seq1), len(seq2))

                    if length >= max_len:
                        seq1 = seq1[:max_len]
                        seq2 = seq2[:max_len]
                    else:
                        pad1 = np.zeros((max_len - len(seq1), seq1.shape[1]), dtype=np.float32)
                        pad2 = np.zeros((max_len - len(seq2), seq2.shape[1]), dtype=np.float32)

                        seq1 = np.vstack([seq1, pad1])
                        seq2 = np.vstack([seq2, pad2])

                        mask[length:] = 0.0

                    combined = np.concatenate([seq1, seq2], axis=1)  # (500, 20)

                    X_list.append(combined)
                    y_list.append(y)
                    mask_list.append(mask)

        print("Finished loading balanced dataset")

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
        M = np.asarray(mask_list, dtype=np.float32)

        print("Balanced counts:")
        for r in np.unique(y[:,0]):
            print(r, np.sum(y[:,0] == r))

        X_train, X_val, y_train, y_val, M_train, M_val = train_test_split(
            X, y, M, test_size=0.2, random_state=42, shuffle=True, stratify=y[:, 0]
        )

        X_train, X_val = self.__normalize(X_train, M_train, X_val, M_val, save_path)

        y_train, y_val = self.__normalize_targets(y_train, y_val, save_path)    

        train_loader = DataLoader(
            EpisodeDataset(X_train, y_train, M_train),
            batch_size=128,
            shuffle=True,
            pin_memory=True
        )

        val_loader = DataLoader(
            EpisodeDataset(X_val, y_val, M_val),
            batch_size=128,
            shuffle=False,
            pin_memory=True
        )

        return train_loader, val_loader, X.shape[2]

    def denormalize_targets(self, y_norm, save_path=SAVE_PATH):
        mean = np.load(f"{save_path}/y_mean.npy")
        std = np.load(f"{save_path}/y_std.npy")

        y_denorm = y_norm.copy()
        y_denorm[:,1] = y_denorm[:,1] * std + mean

        return y_denorm  
    
    @staticmethod
    def __balance_by_exact_success_rate(X, y, M, seed=42):
        """
        Balance dataset so each success_rate value (y[:,0]) appears equally often.
        """
        rng = np.random.default_rng(seed)

        success = y[:, 0]

        # unique success rates (e.g. 0.0, 0.1, ..., 1.0)
        unique_rates = np.unique(success)

        # collect indices per success_rate
        rate_to_indices = {}
        for idx, rate in enumerate(success):
            rate_to_indices.setdefault(rate, []).append(idx)

        # smallest count
        min_count = min(len(idxs) for idxs in rate_to_indices.values())

        # undersample
        selected_indices = []
        for rate, idxs in rate_to_indices.items():
            selected_indices.extend(
                rng.choice(idxs, min_count, replace=False)
            )

        selected_indices = np.array(selected_indices)

        return X[selected_indices], y[selected_indices], M[selected_indices]
    
    @staticmethod
    def __normalize(X_train, M_train, X_val, M_val, save_path):
        # Normalize
        mask_exp = M_train[..., None]  # (B,T,1)

        count = mask_exp.sum(axis=(0,1,2))

        mean = (X_train * mask_exp).sum(axis=(0,1,2)) / count

        var = ((X_train - mean)**2 * mask_exp).sum(axis=(0,1,2)) / count

        std = np.sqrt(var) + 1e-6

        X_train = (X_train - mean) / std
        X_train *= mask_exp

        mask_exp_val = M_val[..., None]  # (B,T,1)
        X_val = (X_val - mean) / std
        X_val *= mask_exp_val

        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/x_mean.npy", mean)
        np.save(f"{save_path}/x_std.npy", std)

        return X_train, X_val
    
    @staticmethod
    def __normalize_targets(y_train, y_val, save_path):
        mean = y_train[:,1].mean(axis=0)
        std = y_train[:,1].std(axis=0) + 1e-6

        y_train[:,1] = (y_train[:,1] - mean) / std
        y_val[:,1] = (y_val[:,1] - mean) / std

        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/y_mean.npy", mean)
        np.save(f"{save_path}/y_std.npy", std)

        return y_train, y_val