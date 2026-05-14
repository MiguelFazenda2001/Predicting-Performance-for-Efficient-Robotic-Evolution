import h5py
import pandas as pd

H5_PATH = "/mnt/DATA/miguelfazenda/reach/dataset/raw/train_episodes.h5"

def main():
    success_rates = []

    with h5py.File(H5_PATH, "r") as h5:
        def visitor(name, obj):
            # We only want genome-level groups
            if isinstance(obj, h5py.Group) and name.count("/") == 2:
                if "success_rate" in obj.attrs:
                    success_rates.append(obj.attrs["success_rate"])
                    print(f"Genome {name.split('/')[-1]}: success_rate = {obj.attrs['success_rate']:.2f}")

        h5.visititems(visitor)

    # Convert to DataFrame
    df = pd.DataFrame({"success_rate": success_rates})

    counts = (
        df["success_rate"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["success_rate", "count"]

    print("\nSuccess rate distribution (per genome):\n")
    print(counts.to_string(index=False))

    print("\nSummary:")
    print(f"Total genomes: {len(df)}")
    print(f"Unique success rates: {counts.shape[0]}")

if __name__ == "__main__":
    main()