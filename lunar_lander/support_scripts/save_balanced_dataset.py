from DataProcessingH5_v2_3_ep import DataProcessingH5
import torch
import json

data_processing = DataProcessingH5() 

SAVE_PATH = "../data"

if __name__ == "__main__":
    train_loader, val_loader, input_dim = data_processing.data_processing_h5(save_path=SAVE_PATH)

    torch.save(train_loader, f"{SAVE_PATH}/train_loader.pth")
    torch.save(val_loader, f"{SAVE_PATH}/val_loader.pth")

    with open(f"{SAVE_PATH}/input_dim.json", "w") as f:
        json.dump(input_dim, f)

