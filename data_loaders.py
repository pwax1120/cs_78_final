""" This script takes the input datasets and produces the data loaders for training and validation. """

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EuroSAT


DATA_ROOT = Path("data")
SPLITS_DIR = DATA_ROOT / "splits"

INPUT_SIZE = 224

DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "eurosat": EuroSAT,
}

# Datasets without a native train/test split; test is carved from the pool via test_idx
NO_NATIVE_TEST = {"eurosat"}


# --- Transform the data --- #
def _train_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),  # CIFAR 32x32 / EuroSAT 64x64 -> 224x224
        transforms.RandomHorizontalFlip(),            # cheap, standard, essentially free
        transforms.ToTensor(),                        # PIL -> tensor, required
    ])

def _eval_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])


def _load_splits(dataset):
    splits_path = SPLITS_DIR / f"{dataset}_splits.json"

    with open(splits_path) as f:
        splits = json.load(f)

    return splits


def _make_pool(DatasetClass, dataset, train, transform):
    """Construct a dataset, handling datasets that lack a `train` argument."""
    if dataset in NO_NATIVE_TEST:
        return DatasetClass(root=DATA_ROOT, download=False, transform=transform)
    return DatasetClass(root=DATA_ROOT, train=train, download=False, transform=transform)


def build_dataloaders(dataset, label_fraction, batch_size, num_workers):
    splits = _load_splits(dataset)
    frac_key = str(label_fraction)

    DatasetClass = DATASETS[dataset]
    has_native_test = dataset not in NO_NATIVE_TEST

    train_pool_aug = _make_pool(DatasetClass, dataset, train=True, transform=_train_transform())
    train_pool_eval = _make_pool(DatasetClass, dataset, train=True, transform=_eval_transform())

    val_idx = splits["val_idx"]
    train_idx = splits["labeled_idx"][frac_key]

    if set(val_idx) & set(train_idx):
        raise RuntimeError(
            f"Splits file is corrupt: val_idx and labeled_idx[{frac_key}] overlap. "
            f"Re-run data_setup.py to regenerate."
        )

    train_ds = Subset(train_pool_aug, train_idx)
    val_ds = Subset(train_pool_eval, val_idx)

    # --- Test set: native for CIFAR, carved from the pool for EuroSAT --- #
    if has_native_test:
        test_ds = _make_pool(DatasetClass, dataset, train=False, transform=_eval_transform())
    else:
        test_idx = splits["test_idx"]
        if set(test_idx) & (set(val_idx) | set(train_idx)):
            raise RuntimeError(
                f"Splits file is corrupt: test_idx overlaps val_idx or labeled_idx[{frac_key}]. "
                f"Re-run data_setup.py to regenerate."
            )
        test_pool_eval = _make_pool(DatasetClass, dataset, train=False, transform=_eval_transform())
        test_ds = Subset(test_pool_eval, test_idx)

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
        test_ds,
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
    for ds in ("cifar10", "eurosat"):
        print(f"\n=== {ds} ===")
        train_loader, val_loader, test_loader = build_dataloaders(
            dataset=ds,
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
    #
    # Expected for eurosat/0.1 (test ≈ 20% of 27k, val/train from the rest):
    #   Test batches:  ~169  (≈5400 / 32, plus partial)