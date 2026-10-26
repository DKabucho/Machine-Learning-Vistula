from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATCH_SIZE = 128
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
SPLIT_SEED = 42


def create_dataloaders(
    data_dir: str | Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    official_train = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=download,
    )
    official_test = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=download,
    )

    train_dataset, validation_dataset = random_split(
        official_train,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(official_test, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader


def main() -> None:
    train_loader, validation_loader, test_loader = create_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")


if __name__ == "__main__":
    main()
