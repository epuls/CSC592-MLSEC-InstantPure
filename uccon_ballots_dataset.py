from pathlib import Path
import torch
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, Subset, DataLoader


from pathlib import Path
import torch
from torch.utils.data import Dataset


class UConnDataset(Dataset):
    """
    UConn ballot dataset loaded from a .pth file.
      - stores only CPU tensors
      - __getitem__ returns (image, label)
      - image is a CPU float tensor in C,H,W layout
      - transform handles resize/channel/normalization
    """

    def __init__(self, pth_path: str, transform=None, target_transform=None):
        self.pth_path = Path(pth_path)
        self.transform = transform
        self.target_transform = target_transform

        if not self.pth_path.exists():
            raise FileNotFoundError(f"Missing UConn dataset file: {self.pth_path}")

        raw = torch.load(str(self.pth_path), map_location="cpu", weights_only=False)

        self.data = raw["data"].detach().cpu()
        self.targets = raw["binary_labels"].detach().long().cpu()

        if len(self.data) != len(self.targets):
            raise RuntimeError(
                f"Data/label length mismatch: {len(self.data)} vs {len(self.targets)}"
            )

        # Normalize storage layout to N,C,H,W.
        if self.data.ndim == 3:
            # N,H,W -> N,1,H,W
            self.data = self.data.unsqueeze(1)
        elif self.data.ndim == 4:
            if self.data.shape[1] in (1, 3):
                # already N,C,H,W
                pass
            elif self.data.shape[-1] in (1, 3):
                # N,H,W,C -> N,C,H,W
                self.data = self.data.permute(0, 3, 1, 2).contiguous()
            else:
                raise RuntimeError(f"Unexpected image tensor shape: {self.data.shape}")
        else:
            raise RuntimeError(f"Unexpected image tensor shape: {self.data.shape}")

        self.classes = ["0", "1"]
        self.class_to_idx = {"0": 0, "1": 1}
        self.samples = list(range(len(self.data)))

    def __getitem__(self, index: int):
        image = self.data[index].clone().float()
        target = int(self.targets[index].item())

        # Match torchvision ToTensor behavior: output should be [0, 1].
        if image.max() > 1.0:
            image = image / 255.0

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image.contiguous(), target

    def __len__(self):
        return len(self.data)


def _load_or_create_split(
    split_file: Path,
    n_total: int,
    test_size: int,
    seed: int,
) -> tuple:
    """
    Return ``(main_indices, test_indices)``.

    If *split_file* exists, load test indices from it (one integer per line).
    Otherwise randomly sample *test_size* indices, write them to *split_file*
    for reproducibility, then derive the complementary main indices.
    """
    if split_file.exists():
        test_indices = [int(x) for x in split_file.read_text().splitlines() if x.strip()]
        if len(test_indices) != test_size:
            print(
                f"Warning: split file contains {len(test_indices)} indices "
                f"but test_size={test_size}. Using file as-is."
            )
        print(f"Loaded test split from {split_file} ({len(test_indices)} samples)")
    else:
        generator = torch.Generator().manual_seed(seed)
        shuffled = torch.randperm(n_total, generator=generator).tolist()
        test_indices = shuffled[:test_size]
        split_file.write_text("\n".join(str(i) for i in test_indices))
        print(f"Created test split → {split_file} ({len(test_indices)} samples)")

    test_set = set(test_indices)
    main_indices = [i for i in range(n_total) if i not in test_set]
    return main_indices, test_indices


def get_uconn_dataloaders(
    batch_size: int = 32,
    test_size: int = 200,
    test_source: str = "val",
    print_option: str = "preprint",
    root: str = "./uconn_voter_center_v2_2/FINALDATASETV3/",
    variant: str = "Combined_Grayscale",
    transform=None,
    val_transform=None,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    seed: int = 42,
):
    """
    Build train / val / test DataLoaders for the UConn voter-center dataset.

    Args:
        test_source: ``"train"`` — test indices are carved from
                     ``preprint/train_{variant}.pth``; the remaining samples
                     form the training set and ``preprint/val_{variant}.pth``
                     is used as-is for validation.

                     ``"val"``   — test indices are carved from
                     ``preprint/val_{variant}.pth``; the remaining val samples
                     stay as the validation set and the full train file is used
                     for training.

        variant:     Filename stem, e.g. ``"Combined_Grayscale"`` or
                     ``"Combined_RGB"``.  A file named
                     ``split_{variant}.txt`` in *root* stores the chosen test
                     indices for reproducibility; it is created automatically
                     on the first run if absent.
    """
    if test_source not in ("train", "val"):
        raise ValueError(f"test_source must be 'train' or 'val', got {test_source!r}")

    root_path  = Path(root).resolve()
    train_pth  = root_path / print_option / f"train_{variant}.pth"
    val_pth    = root_path / print_option / f"val_{variant}.pth"
    split_file = root_path / f"split_{variant}.txt"

    if not train_pth.exists():
        raise FileNotFoundError(f"Missing train file: {train_pth}")
    if not val_pth.exists():
        raise FileNotFoundError(f"Missing val file: {val_pth}")

    if test_source == "train":
        # Load train file twice so train/test can carry independent transforms
        source_for_train = UConnDataset(str(train_pth), transform=transform)
        source_for_test  = UConnDataset(str(train_pth), transform=val_transform)
        val_data         = UConnDataset(str(val_pth),   transform=val_transform)

        n_source = len(source_for_train)
        if test_size >= n_source:
            raise ValueError(f"test_size ({test_size}) >= train file size ({n_source})")

        train_indices, test_indices = _load_or_create_split(
            split_file, n_source, test_size, seed
        )

        train_data = Subset(source_for_train, train_indices)
        test_data  = Subset(source_for_test,  test_indices)

    else:  # test_source == "val"
        train_data = UConnDataset(str(train_pth), transform=transform)
        source_val = UConnDataset(str(val_pth),   transform=val_transform)

        n_source = len(source_val)
        if test_size >= n_source:
            raise ValueError(f"test_size ({test_size}) >= val file size ({n_source})")

        val_indices, test_indices = _load_or_create_split(
            split_file, n_source, test_size, seed
        )

        val_data  = Subset(source_val, val_indices)
        test_data = Subset(source_val, test_indices)

    n_channels = (
        source_for_train.data.shape[1]
        if test_source == "train"
        else source_val.data.shape[1]
    )
    print(f"Variant:    {variant}  |  channels: {n_channels}")
    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")
    print(f"Root:       {root_path}")

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





