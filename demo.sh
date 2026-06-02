#!/usr/bin/env bash
# =============================================================================
# demo.sh — Demonstrate the SimCLR label-efficiency pipeline end to end.
#
# Walks through every stage of the project on a single small config so you can
# see the whole thing work in minutes rather than days:
#
#   1. environment / dependency check
#   2. pretrained-weight acquisition  (load_pretrained.py smoke test)
#   3. deterministic data setup        (data_setup.py)
#   4. dataloader sanity check         (data_loaders.py)
#   5. a short fine-tuning run          (model_finetune.py)
#   6. evaluation of the result        (evaluate.py)
#
# Usage:
#   ./demo.sh            # quick demo: cifar10, 1% labels, 2 epochs  (default)
#   ./demo.sh --full     # a realistic single run: 60 epochs, full fine-tune
#   ./demo.sh --dataset eurosat
#
# Re-runnable: stages that have already produced their output are skipped.
# =============================================================================

set -euo pipefail

# ---- defaults (overridable via flags) ---------------------------------------
DATASET="cifar10"
LABEL_FRACTION="0.01"
MODE="full"
EPOCHS=2
BATCH_SIZE=128
SEED=42
QUICK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)            QUICK=0; EPOCHS=60; LABEL_FRACTION="1.0"; shift ;;
    --dataset)         DATASET="$2"; shift 2 ;;
    --label-fraction)  LABEL_FRACTION="$2"; shift 2 ;;
    --mode)            MODE="$2"; shift 2 ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Pick num_classes from the dataset.
case "$DATASET" in
  cifar10|eurosat) NUM_CLASSES=10 ;;
  cifar100)        NUM_CLASSES=100 ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 1 ;;
esac

RUN_NAME="${DATASET}_${MODE}_frac${LABEL_FRACTION}_seed${SEED}"
CKPT="checkpoints/finetune_outputs/${RUN_NAME}_best.pt"

# Pretty section headers.
hr()   { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }
note() { printf '  \033[0;90m%s\033[0m\n' "$1"; }

# ---- 0. context -------------------------------------------------------------
hr "SimCLR label-efficiency demo"
note "dataset=$DATASET  mode=$MODE  label_fraction=$LABEL_FRACTION  epochs=$EPOCHS  seed=$SEED"
if [[ "$QUICK" -eq 1 ]]; then
  note "running in QUICK mode (small/fast). use --full for a realistic run."
fi

# ---- 1. environment check ---------------------------------------------------
hr "1/6  Environment check"
command -v python >/dev/null 2>&1 || { echo "python not found on PATH" >&2; exit 1; }
python - <<'PY'
import importlib, sys
required = ["torch", "torchvision", "numpy", "safetensors"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print(f"  MISSING packages: {missing}")
    print("  -> run: pip install -r requirements.txt")
    sys.exit(1)
import torch
dev = ("cuda" if torch.cuda.is_available()
       else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
       else "cpu")
print(f"  python      {sys.version.split()[0]}")
print(f"  torch       {torch.__version__}")
print(f"  device      {dev}")
PY

# ---- 2. pretrained weights --------------------------------------------------
hr "2/6  Pretrained SimCLR weights"
if [[ -f "checkpoints/simclr_v1_r50_1x_pytorch.safetensors" ]]; then
  note "cleaned weights already present — skipping download."
else
  note "fetching cleaned SimCLR v1 ResNet-50 weights (~102 MB) via load_pretrained.py ..."
  # The module's __main__ smoke test triggers the gdown download + a strict load.
  python load_pretrained.py
fi

# ---- 3. data setup (deterministic splits) -----------------------------------
hr "3/6  Data setup — deterministic stratified splits"
SPLIT_JSON="data/splits/${DATASET}_splits.json"
if [[ -f "$SPLIT_JSON" ]]; then
  note "$SPLIT_JSON already exists — skipping (splits are deterministic anyway)."
else
  note "downloading $DATASET and writing $SPLIT_JSON ..."
  python data_setup.py --dataset "$DATASET"
fi

# ---- 4. dataloader sanity check ---------------------------------------------
hr "4/6  DataLoader sanity check"
note "building train/val/test loaders and inspecting one batch ..."
python - "$DATASET" "$LABEL_FRACTION" <<'PY'
import sys
from data_loaders import build_dataloaders
dataset, frac = sys.argv[1], float(sys.argv[2])
train, val, test = build_dataloaders(dataset, label_fraction=frac,
                                     batch_size=32, num_workers=0)
print(f"  train batches: {len(train)}   val: {len(val)}   test: {len(test)}")
x, y = next(iter(train))
print(f"  batch x: {list(x.shape)} {x.dtype}  range=[{x.min():.3f}, {x.max():.3f}]")
print(f"  batch y: {y.dtype}  classes in batch: {sorted(set(y.tolist()))[:10]}")
PY

# ---- 5. fine-tuning ---------------------------------------------------------
hr "5/6  Fine-tuning ($MODE, ${EPOCHS} epochs)"
if [[ -f "$CKPT" ]]; then
  note "checkpoint $CKPT already exists — skipping training."
else
  note "training $RUN_NAME ... (best-val checkpoint -> $CKPT)"
  python - <<PY
from model_finetune import train, FinetuneConfig
cfg = FinetuneConfig(
    dataset="$DATASET",
    num_classes=$NUM_CLASSES,
    label_fraction=$LABEL_FRACTION,
    mode="$MODE",
    lr=0.003125,
    weight_decay=0.0,
    momentum=0.9,
    optimizer="sgd",
    epochs=$EPOCHS,
    batch_size=$BATCH_SIZE,
    scheduler="cosine",
    warmup_epochs=0,
    seed=$SEED,
    num_workers=0,
)
path = train(cfg)
print(f"  saved best checkpoint: {path}")
PY
fi

# ---- 6. evaluation ----------------------------------------------------------
hr "6/6  Evaluation"
if [[ -f "$CKPT" ]]; then
  note "evaluating $CKPT on the $DATASET test split ..."
  python evaluate.py --checkpoint "$CKPT"
  EVAL_JSON="${CKPT%.pt}_eval.json"
  if [[ -f "$EVAL_JSON" ]]; then
    note "results written to $EVAL_JSON"
    python - "$EVAL_JSON" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["evaluation_metrics"]
print(f"  test top-1: {m['test_acc_top1']*100:.2f}%   "
      f"top-5: {m['test_acc_top5']*100:.2f}%   "
      f"loss: {m['test_loss']:.4f}")
PY
  fi
else
  echo "  no checkpoint to evaluate (training stage produced nothing)." >&2
fi

hr "Demo complete"
note "to evaluate a whole sweep folder at once:"
note "    python evaluate.py --folder checkpoints/finetune_outputs/"
note "to launch the full SLURM sweep on Discovery:"
note "    sbatch run_finetune.sh"