""" This script takes the input datasets and produces the data loaders for training and validation. """

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


DATA_ROOT = Path("data")
SPLITS_DIR = DATA_ROOT / "splits"

INPUT_SIZE = 224

DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
}

# --- Transform the data --- #
def _train_transform():
    return transforms.Compose([
        transforms.Resize(224),           # 32x32 -> 224x224, required
        transforms.RandomHorizontalFlip(), # cheap, standard, essentially free
        transforms.ToTensor(),             # PIL -> tensor, required
    ])

def _eval_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

def _load_splits(dataset):
    splits_path = SPLITS_DIR / f"{dataset}_splits.json"

    with open(splits_path) as f:
        splits = json.load(f)
    
    return splits


def build_dataloaders(dataset, label_fraction, batch_size, num_workers):
    splits = _load_splits(dataset)
    frac_key = str(label_fraction)

    DatasetClass = DATASETS[dataset]
    train_pool_aug = DatasetClass(root=DATA_ROOT, train=True, download=False, transform=_train_transform())
    train_pool_eval = DatasetClass(root=DATA_ROOT, train=True, download=False, transform=_eval_transform())
    test_set = DatasetClass(root=DATA_ROOT, train=False, download=False, transform=_eval_transform())

    val_idx = splits["val_idx"]
    train_idx = splits["labeled_idx"][frac_key]

    if set(val_idx) & set(train_idx):
        raise RuntimeError(
            f"Splits file is corrupt: val_idx and labeled_idx[{frac_key}] overlap. "
            f"Re-run data_setup.py to regenerate."
        )
    
    train_ds = Subset(train_pool_aug, train_idx)
    val_ds = Subset(train_pool_eval, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,       # avoid tiny final batches that destabilize BN
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    # Quick check that everything wires up correctly for one config
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset="cifar10",
        label_fraction=0.1,
        batch_size=32,
        num_workers=0,  # 0 for the smoke test to avoid multiprocessing overhead
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Pull one batch and check shapes
    x, y = next(iter(train_loader))
    print(f"Train batch x: {x.shape}, dtype={x.dtype}, range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"Train batch y: {y.shape}, dtype={y.dtype}, unique={y.unique().tolist()}")
    
    # Expected for cifar10/0.1:
    #   Train batches: ~140  (4500 examples / 32 batch size, drop_last=True)
    #   Val batches:    157  (5000 / 32, plus partial)
    #   Test batches:   313  (10000 / 32, plus partial)
    #   Train batch x: torch.Size([32, 3, 224, 224]), range=[0.000, 1.000]
    #   Train batch y: torch.Size([32]), dtype=torch.int64