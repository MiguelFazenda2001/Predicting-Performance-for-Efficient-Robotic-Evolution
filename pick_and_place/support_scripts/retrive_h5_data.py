import h5py

src_path = "/mnt/DATA/miguelfazenda/pickplace/dataset/raw/train_episodes.h5"
dst_path = "/mnt/DATA/miguelfazenda/pickplace/dataset/raw/train_episodes_fixed.h5"

with h5py.File(src_path, "r", swmr=True) as src, \
     h5py.File(dst_path, "w") as dst:

    for evo_name in src.keys():
        try:
            print("Copying", evo_name)
            src.copy(evo_name, dst)
        except Exception as e:
            print("Skipping corrupted part:", evo_name)
            print(e)

print("Recovery complete")