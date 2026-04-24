from pathlib import Path
import torch
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, Subset, DataLoader, random_split



class UConnGrayscaleDataset(Dataset):
    """
    UConn voter center grayscale dataset loaded from a .pth file.

    The .pth file must contain:
        data          — Tensor of shape (N, 1, H, W) or (N, H, W)
        binary_labels — Tensor of shape (N,) with integer class labels

    Args:
        pth_path: Full path to the .pth file to load.
        transform:  Optional transform applied to each image tensor.
    """

    def __init__(self, pth_path: str, transform=None):
        self.transform = transform
        raw = torch.load(pth_path, weights_only=False)
        self.data: torch.Tensor = raw["data"]
        self.labels: torch.Tensor = raw["binary_labels"].long()

        # Ensure shape is (N, C, H, W)
        if self.data.ndim == 3:
            self.data = self.data.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image = self.data[idx].float()
        label = int(self.labels[idx].item())

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_uconn_grayscale_dataloaders(
    batch_size: int = 32,
    test_size: int = 200,
    root: str = "./uconn_voter_center_v2_2/FINALDATASETV3/",
    transform=None,
    val_transform=None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    seed: int = 42,
    variant: str = "Combined_Grayscale"
):
    """
    Build train / val / test DataLoaders for the UConn grayscale preprint dataset.

    - train + test: split from ``preprint/train_{variant}.pth``
    - val:          loaded directly from ``preprint/val_{variant}.pth``

    Each loader returns batches of ``(image, label, sample_id)`` via IndexedDataset.
    """
    root_path = Path(root).resolve()
    train_pth = root_path / "preprint" / f"train_{variant}.pth"
    val_pth   = root_path / "preprint" / f"val_{variant}.pth"

    if not train_pth.exists():
        raise FileNotFoundError(f"Missing train file: {train_pth}")
    if not val_pth.exists():
        raise FileNotFoundError(f"Missing val file: {val_pth}")

    # Load twice so train and test can have independent transforms
    train_full = UConnGrayscaleDataset(str(train_pth), transform=transform)
    test_full  = UConnGrayscaleDataset(str(train_pth), transform=val_transform)
    val_data   = UConnGrayscaleDataset(str(val_pth),   transform=val_transform)

    n_total = len(train_full)
    if test_size >= n_total:
        raise ValueError(
            f"test_size ({test_size}) must be smaller than train file size ({n_total})"
        )

    train_size = n_total - test_size
    print(f"Train size: {train_size}, Train Shape: {train_full.data[:train_size].shape}")
    print(f"Val size:   {len(val_data)}, Val Shape: {val_data.data.shape}")
    print(f"Test size:  {test_size}, Test Shape: {test_full.data[train_size:].shape}")
    print(f"Root filepath: {root_path}")

    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = all_indices[:train_size]
    test_indices  = all_indices[train_size:]

    train_data = Subset(train_full, train_indices)
    test_data  = Subset(test_full,  test_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader


# Load the pre-print and post-print data, and print out their shapes and unique classes to verify they are loaded correctly. Then, map the post-print data back to its corresponding pre-print data using the original indices.
def load_and_inspect_data(path: str = "D:/uconn_dataset/"):
    preprint_data = torch.load(f"{path}/preprint/train_Combined_Grayscale.pth", weights_only=False)
    postprint_data = torch.load(f"{path}/postprint/train_Combined_Grayscale.pth", weights_only=False)
    print("========================")
    print("Reading: PRE-PRINT data")
    print(preprint_data.keys())
    print(f"Pre-print data shape: {preprint_data['data'].shape}")
    print(f"Unique classes in pre-print data: {preprint_data['binary_labels'].unique(return_counts=True)}")
    print("========================")
    print("Reading: POST-PRINT data")
    print(postprint_data.keys())
    print(f"Post-print data shape: {postprint_data['data'].shape}")
    print(f"Unique classes in post-print data: {postprint_data['binary_labels'].unique(return_counts=True)}")
    print("========================")
    # plot random pre print samples with their corresponding classes
    random_indices = random.sample(range(preprint_data['data'].shape[0]), 5)
    for idx in random_indices:
        plt.imshow(preprint_data['data'][idx].cpu().squeeze(), cmap='gray')
        plt.title(f"Pre-print Class: {preprint_data['binary_labels'][idx].cpu().item()}")
        plt.axis('off')
        plt.show()





