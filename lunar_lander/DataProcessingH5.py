import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tcn import EpisodeDataset

H5_PATH = "data/episodes.h5"
SAVE_PATH = "models"
MAX_LEN = 500


class DataProcessingH5:
    def __init__(self):
        pass

    def data_processing_h5(self,save_path=SAVE_PATH, h5_path=H5_PATH, max_len=MAX_LEN):
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
        print("Total counts per success rate:")
        for r in np.unique(y[:,0]):
            print(r, np.sum(y[:,0] == r))


        X, y = self.__balance_by_exact_success_rate(X, y)
        print("Balanced counts per success rate:")
        for r in np.unique(y[:,0]):
            print(r, np.sum(y[:,0] == r))

        # Normalize
        mean = X.mean(axis=(0,1), keepdims=True)
        std = X.std(axis=(0,1), keepdims=True) + 1e-6
        X = (X - mean) / std

        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/x_mean.npy", mean)
        np.save(f"{save_path}/x_std.npy", std)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y[:, 0]
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
    
    @staticmethod
    def __balance_by_exact_success_rate(X, y, seed=42):
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

        return X[selected_indices], y[selected_indices]