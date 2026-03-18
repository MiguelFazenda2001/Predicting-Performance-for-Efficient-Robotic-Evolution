import torch
from torch.utils.data import DataLoader

SAVE_PATH = "../data"
NEW_BATCH_SIZE = 64


def rebuild_dataloaders(save_path, new_batch_size):
    # Load the original dataloaders
    train_loader_old = torch.load(f"{save_path}/train_loader.pth", weights_only=False)
    val_loader_old = torch.load(f"{save_path}/val_loader.pth", weights_only=False)

    # Extract datasets from the old loaders
    train_dataset = train_loader_old.dataset
    val_dataset = val_loader_old.dataset

    # Create new dataloaders with a different batch size
    train_loader_new = DataLoader(
        train_dataset,
        batch_size=new_batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader_new = DataLoader(
        val_dataset,
        batch_size=new_batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader_new, val_loader_new


if __name__ == "__main__":
    train_loader, val_loader = rebuild_dataloaders(SAVE_PATH, NEW_BATCH_SIZE)

    # Optional: save the new loaders
    torch.save(train_loader, f"{SAVE_PATH}/train_loader_bs{NEW_BATCH_SIZE}.pth")
    torch.save(val_loader, f"{SAVE_PATH}/val_loader_bs{NEW_BATCH_SIZE}.pth")

    print("New DataLoaders created successfully.")
    print(f"Train batch size: {train_loader.batch_size}")
    print(f"Validation batch size: {val_loader.batch_size}")