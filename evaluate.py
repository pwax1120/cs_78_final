"""Evaluate a fine-tuned SimCLR checkpoint on its test set.

Usage:
    python evaluate.py --checkpoint <path_to_finetuned.pt>
    python evaluate.py --checkpoint <path> --output results/eval_<run_name>.json

Loads a checkpoint produced by model_finetune.py, runs it on the canonical
test split for its dataset, and writes a JSON results file with all the
metrics + provenance needed for the "anyone can verify" workflow.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from load_pretrained import load_pretrained_resnet50x1
from data_loaders import build_dataloaders


# --- Core evaluation -------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes):
    """Run model over loader. Returns dict of metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top5 = 0
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Top-1
        _, top1 = outputs.topk(1, dim=1)
        correct_top1 += (top1.squeeze(1) == targets).sum().item()

        # Top-5 (capped at num_classes for CIFAR-10)
        k = min(5, num_classes)
        _, topk = outputs.topk(k, dim=1)
        correct_top5 += topk.eq(targets.view(-1, 1)).any(dim=1).sum().item()

        # Per-class
        for c in range(num_classes):
            mask = (targets == c)
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (top1.squeeze(1)[mask] == c).sum().item()

        total += inputs.size(0)

    per_class_acc = (per_class_correct.float() / per_class_total.clamp(min=1).float()).tolist()

    return {
        "test_loss": total_loss / total,
        "test_acc_top1": correct_top1 / total,
        "test_acc_top5": correct_top5 / total,
        "per_class_accuracy": per_class_acc,
        "per_class_correct": per_class_correct.tolist(),
        "per_class_total": per_class_total.tolist(),
        "num_test_examples": total,
    }


# --- Checkpoint loading ----------------------------------------------------

def load_finetuned_model(checkpoint_path, device):
    """Reconstruct the fine-tuned model from a saved checkpoint.

    Returns (model, checkpoint_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    num_classes = config["num_classes"]

    # Rebuild architecture, then load weights
    model = load_pretrained_resnet50x1(num_classes=num_classes, freeze_encoder=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    return model, ckpt


def _resolve_device(spec):
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Public entry point ----------------------------------------------------

def evaluate(checkpoint_path: str, output_path: str = None,
             batch_size: int = 128, num_workers: int = 2,
             device_spec: str = "auto") -> dict:
    """Evaluate a fine-tuned checkpoint. Returns the results dict.

    If output_path is provided (or auto-derived), also writes JSON to disk.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _resolve_device(device_spec)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load model + provenance
    model, ckpt = load_finetuned_model(checkpoint_path, device)
    config = ckpt["config"]
    dataset = config["dataset"]
    num_classes = config["num_classes"]

    print(f"Dataset: {dataset} ({num_classes} classes)")
    print(f"Training mode: {config['mode']}, label_fraction={config['label_fraction']}")
    print(f"Trained with seed={config['seed']}, epochs={config['epochs']}")

    # Build test loader (we ignore train/val here)
    _, _, test_loader = build_dataloaders(
        dataset=dataset,
        label_fraction=config["label_fraction"],  # any valid fraction works for test
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Run evaluation
    print(f"Running evaluation on {len(test_loader.dataset)} test examples...")
    t0 = time.time()
    metrics = evaluate_model(model, test_loader, device, num_classes)
    dt = time.time() - t0

    print(f"\n--- Results ({dt:.1f}s) ---")
    print(f"Test top-1 accuracy: {metrics['test_acc_top1']:.4f}")
    print(f"Test top-5 accuracy: {metrics['test_acc_top5']:.4f}")
    print(f"Test loss:           {metrics['test_loss']:.4f}")
    print(f"Per-class accuracy:  min={min(metrics['per_class_accuracy']):.3f}, "
          f"max={max(metrics['per_class_accuracy']):.3f}")

    # Assemble full results record
    results = {
        "checkpoint_path": str(checkpoint_path),
        "evaluation_metrics": metrics,
        "evaluation_time_seconds": dt,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluation_device": str(device),
        "training_config": config,
        "training_metrics": ckpt.get("metrics", {}),
        "architecture": ckpt.get("architecture", "unknown"),
        "training_git_commit": ckpt.get("git_commit", "unknown"),
        "training_timestamp": ckpt.get("timestamp", "unknown"),
    }

    # Write results
    if output_path is None:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_eval.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {output_path}")

    return results


# --- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned SimCLR checkpoint on its test set."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt checkpoint produced by model_finetune.py")
    parser.add_argument("--output", default=None,
                        help="Where to write the results JSON (default: alongside the checkpoint)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_spec=args.device,
    )


if __name__ == "__main__":
    main()