""" Finetuning model for the SimCLR pretrained ResNet-50x1 encoder. 
This is a simple wrapper around the load_pretrained_resnet50x1 function in load_pretrained.py, 
which allows you to easily load the pretrained weights and then fine-tune on a downstream classification task by 
replacing the final fully connected layer and optionally freezing the encoder layers. """

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_pretrained import load_pretrained_resnet50x1
from data_loaders import build_dataloaders

# This creates a dataclass for each training run allowing the script to only need one clean input into it
@dataclass
class FinetuneConfig:

    # Dataset Specifics
    dataset: str # the chosen dataset for the run
    num_classes: int # the number of outputs, allows the build function to accurately add the last layer
    label_fraction: float # the fraction of labels to use for training, should be one of [0.01, 0.1, 1.0]

    # Training Mode
    mode: str # either "finetune" or "linear_eval", determines whether the encoder layers are frozen or not during training

    # Optimizer - to hold consistent training values from the SimCLR paper
    lr: float = 0.0
    weight_decay: float = 0.0
    momentum: float = 0.0
    optimizer: str = "sgd" 
    epochs: int = 100
    batch_size: int = 256
    scheduler: str = "cosine" 
    warmup_epochs: int = 0

    # Reproducibility
    seed: int = 42 
    device: str = "cuda"
    num_workers: int = 4


    # I/O - should help make training more clear and help to debug
    output_dir: str = "./checkpoints/finetune_outputs" # where to save the model checkpoints and training logs
    run_name: str = "" # a name for the run, used for naming the output files

# Function to set specific seeds so training can be compared and allows for reproducibility
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _build_model(config: FinetuneConfig):
    freeze_encoder = (config.mode == "linear_eval") # if we're doing linear evaluation, we want to freeze the encoder layers
    return load_pretrained_resnet50x1(num_classes=config.num_classes, freeze_encoder=freeze_encoder) 

def _build_optimizer(model, config):
    trainable = [ p for p in model.parameters() if p.requires_grad ]

    return torch.optim.SGD(
        trainable,
        lr = config.lr,
        momentum = config.momentum,
        weight_decay = config.weight_decay,
    )

def _build_scheduler(optimizer, config, steps_per_epoch):
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.epochs * steps_per_epoch,
        )
    if config.scheduler == "none":
        return None
    raise ValueError(f"Unsupported scheduler type: {config.scheduler}")
  

def _train_one_epoch(model, data_loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(torch.device("cuda"), non_blocking=True)
        targets = targets.to(torch.device("cuda"), non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, device, criterion):
    """One pass over a val/test set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total

def _save_checkpoint(model, optimizer, config, metrics, path: Path) -> None:
    """Save a self-describing checkpoint bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
        "architecture": "resnet50x1_simclr_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, path)


def _default_run_name(config: FinetuneConfig) -> str:
    return (
        f"{config.dataset}"
        f"_{config.mode}"
        f"_frac{config.label_fraction}"
        f"_seed{config.seed}"
    )


def train(config: FinetuneConfig) -> Path:
    """Run end-to-end fine-tuning. Returns the path to the best checkpoint."""
    _set_seed(config.seed)
    device = "cuda"
    print(f"Device: {device}")
    print(f"Config: {asdict(config)}")

    # Build everything
    print("Loading model...", flush=True)
    model = _build_model(config).to(device)
    print("Model loaded", flush=True)

    print("Loading data...", flush=True)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset=config.dataset,
        label_fraction=config.label_fraction,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    print("Data loaded", flush=True)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config, len(train_loader))
    criterion = nn.CrossEntropyLoss()

    # Track history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    run_name = config.run_name or _default_run_name(config)
    best_path = Path(config.output_dir) / f"{run_name}_best.pt"

    # Loop
    for epoch in range(config.epochs):
        t0 = time.time()
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_acc = _evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:3d}/{config.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| {dt:.1f}s", flush=True
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(
                model, optimizer, config,
                {**history, "best_val_acc": best_val_acc, "best_epoch": epoch + 1},
                best_path,
            )

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint:   {best_path}")
    return best_path


if __name__ == "__main__":
    """Smoke test: minimal config that exercises every code path in ~30 seconds.
    
    This is NOT a meaningful experiment — it just verifies the training loop
    runs end-to-end. Real runs should be launched from a separate runner script
    with SimCLR-protocol hyperparameters.
    """
    config = FinetuneConfig(
        dataset="cifar10",
        num_classes=10,
        label_fraction=1,    
        mode="full",   
        lr=0.003125,            # scaled from 4096 lr to 256 following the SimCLR paper's linear eval protocol
        momentum=0.9,
        weight_decay=0.0,
        optimizer="sgd",
        epochs=60,             
        batch_size=256,
        scheduler="cosine",
        seed=42,
        device="auto",          
        num_workers=0,
    )
    train(config)
