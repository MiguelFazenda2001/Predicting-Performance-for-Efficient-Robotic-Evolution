import h5py

src = "/mnt/DATA/miguelfazenda/pickplace/dataset/raw/train_episodes_hf_6.h5"
dst = "/mnt/DATA/miguelfazenda/pickplace/dataset/raw/train_episodes_hf.h5"

with h5py.File(dst, "a") as dst_file:
    with h5py.File(src, "r") as src_file:
        for key in src_file.keys():
            print(f"Copying group: {key}")
            src_file.copy(key, dst_file, name='evo_300')