from __future__ import annotations
""" One-time script to generate deterministic, stratified split index files."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, EuroSAT


# --- PROTOCOL CONSTANTS --- #
VAL_SEED = 20260524
LABEL_SEED = 20260525
VAL_FRAC = .1
LABEL_FRACTIONS = [.01, .1, 1.0]
TEST_FRAC = .2
TEST_SEED = 20260526

DATA_ROOT = Path("data")
SPLITS_DIR = DATA_ROOT / "splits"

DATASETS = {
    "cifar10": (CIFAR10, 10),
    "cifar100": (CIFAR100, 100),
    "eurosat": (EuroSAT, 10),
}
# For datasets with no built-in test split, carve one out deterministically
def build_test_indices(targets, test_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    by_class = _indices_by_class(targets)

    test_idx = []
    for y, class_indices in by_class.items():
        class_indices = list(class_indices)
        rng.shuffle(class_indices)
        n_test = int(round(test_frac * len(class_indices)))
        test_idx.extend(class_indices[:n_test])

    return sorted(test_idx)

# Private function to get the indices of each class label as a list
def _indices_by_class(targets):
    by_class = {}
    for i, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(i)
    
    return by_class

def build_val_indices(targets, val_frac: float, seed: int, exclude=None):
    rng = np.random.RandomState(seed)
    by_class = _indices_by_class(targets)
    exclude = exclude or set()

    val_idx = []
    for y, class_indices in by_class.items():
        class_indices = [i for i in class_indices if i not in exclude]
        rng.shuffle(class_indices)
        n_val = int(round(val_frac * len(class_indices)))
        val_idx.extend(class_indices[:n_val])

    return sorted(val_idx)

def build_labeled_indices(targets, fractions, val_idx, seed, exclude=None):
    rng = np.random.RandomState(seed)
    val_set = set(val_idx)
    exclude = (exclude or set()) | val_set
    by_class = _indices_by_class(targets)

    for y in by_class:
        by_class[y] = [i for i in by_class[y] if i not in exclude]

    out = {}
    for frac in sorted(fractions):
        idxs = []
        for y, class_indices in by_class.items():
            n = int(np.ceil(frac * len(class_indices)))
            idxs.extend(class_indices[:n])
        out[str(frac)] = sorted(idxs)

    return out

#NOTE: I used claude code here to teach me how to use the assert errors and clean up the UI interactions
def verify_splits(splits, n_classes, targets):
    val_idx = splits["val_idx"]
    labeled_idx = splits["labeled_idx"]
    test_idx = splits.get("test_idx", [])
    val_set = set(val_idx)
    test_set = set(test_idx)

    # test must not overlap val or any labeled subset
    assert not (test_set & val_set), "test_idx overlaps val_idx"
    for f, idxs in labeled_idx.items():
        assert not (test_set & set(idxs)), f"test_idx overlaps labeled_idx[{f}]"
        assert not (val_set & set(idxs)), f"val_idx overlaps labeled_idx[{f}]"

    sorted_fracs = sorted(labeled_idx.keys(), key=float)
    for f1, f2 in zip(sorted_fracs, sorted_fracs[1:]):
        s1, s2 = set(labeled_idx[f1]), set(labeled_idx[f2])
        assert s1.issubset(s2), f"labeled_idx[{f1}] is not a subset of labeled_idx[{f2}]"

    # class coverage
    val_classes = {int(targets[i]) for i in val_idx}
    assert len(val_classes) == n_classes, (
        f"val has {len(val_classes)} classes, expected {n_classes}"
    )

    if test_idx:
        test_classes = {int(targets[i]) for i in test_idx}
        assert len(test_classes) == n_classes, (
            f"test has {len(test_classes)} classes, expected {n_classes}"
        )

    smallest = sorted_fracs[0]
    smallest_classes = {int(targets[i]) for i in labeled_idx[smallest]}
    assert len(smallest_classes) == n_classes, (
        f"labeled_idx[{smallest}] has {len(smallest_classes)} classes, expected {n_classes}"
    )

    print("  All verification checks passed.")

#NOTE: I wrote the logic for this function myself but used Claude code to insert print statements for better debugging
def build_splits_for(dataset_name: str) -> None:
    DatasetClass, n_classes = DATASETS[dataset_name]
    needs_test_split = dataset_name == "eurosat"

    print(f"\n[{dataset_name}] downloading / loading train_pool ...")
    if dataset_name == "eurosat":
        train_pool = DatasetClass(root=DATA_ROOT, download=True)
        targets = [s[1] for s in train_pool.samples]
    else:
        train_pool = DatasetClass(root=DATA_ROOT, train=True, download=True)
        targets = train_pool.targets
    print(f"[{dataset_name}] train_pool size: {len(targets)} ({n_classes} classes)")

    test_idx = []
    if needs_test_split:
        print(f"[{dataset_name}] building test indices (frac={TEST_FRAC}, seed={TEST_SEED}) ...")
        test_idx = build_test_indices(targets, TEST_FRAC, TEST_SEED)
        print(f"[{dataset_name}] test size: {len(test_idx)}")
    exclude = set(test_idx)

    print(f"[{dataset_name}] building val indices (frac={VAL_FRAC}, seed={VAL_SEED}) ...")
    val_idx = build_val_indices(targets, VAL_FRAC, VAL_SEED, exclude=exclude)
    print(f"[{dataset_name}] val size: {len(val_idx)}")

    print(f"[{dataset_name}] building labeled indices for {LABEL_FRACTIONS} ...")
    labeled_idx = build_labeled_indices(targets, LABEL_FRACTIONS, val_idx, LABEL_SEED, exclude=exclude)
    for f, idxs in labeled_idx.items():
        print(f"[{dataset_name}]   fraction {f}: {len(idxs)} examples")

    splits = {
        "dataset": dataset_name,
        "n_classes": n_classes,
        "val_seed": VAL_SEED,
        "label_seed": LABEL_SEED,
        "test_seed": TEST_SEED if needs_test_split else None,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC if needs_test_split else None,
        "label_fractions": LABEL_FRACTIONS,
        "val_idx": val_idx,
        "test_idx": test_idx,
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
