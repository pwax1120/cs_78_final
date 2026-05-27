from __future__ import annotations
""" One-time script to generate deterministic, stratified split index files."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from __future__ import annotations

# --- PROTOCOL CONSTANTS --- #
VAL_SEED = 20260524
LABEL_SEED = 20260525
VAL_FRAC = .1
LABEL_FRACTIONS = [.01, .1, 1.0]

DATA_ROOT = Path("data")
SPLITS_DIR = DATA_ROOT / "splits"

DATASETS = {
    "cifar10": (CIFAR10, 10),
    "cifar100": (CIFAR100, 100),
}

# Private function to get the indices of each class label as a list
def _indices_by_class(targets):
    by_class = {}
    for i, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(i)
    
    return by_class

# For a given value fraction return the targets sorted by list
def build_val_indices(targets, val_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    by_class = _indices_by_class(targets)

    val_idx = []
    for y, class_indices in by_class.items():
        class_indices = list(class_indices)
        rng.shuffle(class_indices)
        n_val = int(round(val_frac * len(class_indices)))
        val_idx.extend(class_indices[:n_val])

    
    return sorted(val_idx)

# For each fraction return the nested prefix of train_pool
def build_labeled_indices(targets, fractions: list[float], val_idx: list[int], seed: int):
    rng = np.random.RandomState(seed)
    val_set = set(val_idx)
    by_class = _indices_by_class(targets)

    # Remove the val indices from each class's pool, then shuffle deterministically
    for y in by_class:
        by_class[y] = [i for i in by_class[y] if i not in val_set]
    
    out = {}
    for frac in sorted(fractions):
        idxs = []
        for y, class_indices in by_class.items():
            n = int(np.ceil(frac * len(class_indices)))
            idxs.extend(class_indices[:n])
        
        out[str(frac)] = sorted(idxs)

    return out

#NOTE: I used claude code here to teach me how to use the assert errors and clean up the UI interactions
def verify_splits(splits: dict, n_classes: int, targets) -> None:
    val_idx = splits["val_idx"]
    labeled_idx = splits["labeled_idx"]
    val_set = set(val_idx)

    for f, idxs in labeled_idx.items():
        overlap = val_set & set(idxs)
        assert not overlap, f"val_idx overlaps labeled_idx[{f}]: {len(overlap)} indices"

    sorted_fracs = sorted(labeled_idx.keys(), key=float)
    for f1, f2 in zip(sorted_fracs, sorted_fracs[1:]):
        s1, s2 = set(labeled_idx[f1]), set(labeled_idx[f2])
        assert s1.issubset(s2), f"labeled_idx[{f1}] is not a subset of labeled_idx[{f2}]"

    val_classes = {int(targets[i]) for i in val_idx}
    assert len(val_classes) == n_classes, (
        f"val has {len(val_classes)} classes, expected {n_classes}"
    )

    smallest = sorted_fracs[0]
    smallest_classes = {int(targets[i]) for i in labeled_idx[smallest]}
    assert len(smallest_classes) == n_classes, (
        f"labeled_idx[{smallest}] has {len(smallest_classes)} classes, "
        f"expected {n_classes}"
    )

    print("  All verification checks passed.")

#NOTE: I wrote the logic for this function myself but used Claude code to insert print statements for better debugging
def build_splits_for(dataset_name: str) -> None:
    DatasetClass, n_classes = DATASETS[dataset_name]

    print(f"\n[{dataset_name}] downloading / loading train_pool ...")
    train_pool = DatasetClass(root=DATA_ROOT, train=True, download=True)
    targets = train_pool.targets   # list[int] for CIFAR
    print(f"[{dataset_name}] train_pool size: {len(targets)} ({n_classes} classes)")

    print(f"[{dataset_name}] building val indices (frac={VAL_FRAC}, seed={VAL_SEED}) ...")
    val_idx = build_val_indices(targets, VAL_FRAC, VAL_SEED)
    print(f"[{dataset_name}] val size: {len(val_idx)}")

    print(f"[{dataset_name}] building labeled indices for {LABEL_FRACTIONS} ...")
    labeled_idx = build_labeled_indices(targets, LABEL_FRACTIONS, val_idx, LABEL_SEED)
    for f, idxs in labeled_idx.items():
        print(f"[{dataset_name}]   fraction {f}: {len(idxs)} examples")

    splits = {
        "dataset": dataset_name,
        "n_classes": n_classes,
        "val_seed": VAL_SEED,
        "label_seed": LABEL_SEED,
        "val_frac": VAL_FRAC,
        "label_fractions": LABEL_FRACTIONS,
        "val_idx": val_idx,
        "labeled_idx": labeled_idx,
    }

    print(f"[{dataset_name}] verifying splits ...")
    verify_splits(splits, n_classes, targets)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPLITS_DIR / f"{dataset_name}_splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[{dataset_name}] wrote {out_path} ({size_kb:.1f} KB)")

#NOTE: I used Claude code in this functino to help me understand how to use inline parsing better
# to make it easier to interact with the datasets. 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        required=True,
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        build_splits_for(ds)


if __name__ == "__main__":
    main()
