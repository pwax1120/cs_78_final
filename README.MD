# SimCLR v1 Label-Efficiency Study (CS78 Final Project)

Fine-tunes a self-supervised SimCLR v1 ResNet-50 (1x) encoder that was pretrained on
ImageNet without labels on downstream image-classification datasets
(CIFAR-10 and EuroSAT) and measures how accuracy scales with the
amount of labeled data. Each dataset is run at multiple label fractions
(1%, 10%, 100%) in two regimes: linear evaluation with frozen encoders and full fine-tunings.

Everything is engineered to be reproducible: deterministic stratified splits stored as
JSON, fixed seeds, version-controlled code, and self-describing checkpoint bundles that
carry their own training config and git commit.

--- 
## Repository layout

```
CS78_Final_Project/
├── resnet_wider.py        # ResNet-50 architecture (width_mult variant; key names match the SimCLR ckpt)
├── clean_checkpoint.py    # ONE-TIME: raw HF safetensors -> clean, strict-loadable PyTorch safetensors
├── load_pretrained.py     # RUNTIME: download cleaned weights from Drive + build the model
├── data_setup.py          # ONE-TIME: download datasets + write deterministic split JSONs
├── data_loaders.py        # RUNTIME: build (train, val, test) DataLoaders from the split JSONs
├── model_finetune.py      # RUNTIME: FinetuneConfig + train(config) training loop (single config)
├── training.py            # RUNTIME: array driver — picks CONFIGS[run_id] for a SLURM job array
├── evaluate.py            # RUNTIME: evaluate one checkpoint or a whole folder, write *_eval.json
├── run_finetune.sh        # SLURM: submit the whole sweep as a job array
├── requirements.txt
├── finetune_outputs       # generated models for review
└── data/splits/*.json     # generated, committed (raw data and checkpoints are gitignored)
```

---

## Environment setup

### Locally (Mac / MPS)

```bash
conda create -n cs78 python=3.11 -y
conda activate cs78
pip install -r requirements.txt        
```

---

## One-time data setup

**(a) Generate the splits** (downloads the datasets on first run):

```bash
python data_setup.py --dataset all      # cifar10, cifar100, eurosat
# or one at a time:
python data_setup.py --dataset cifar10
python data_setup.py --dataset eurosat
```

This writes `data/splits/<dataset>_splits.json`. Splits are deterministic
(`VAL_SEED=20260524`, `LABEL_SEED=20260525`, `TEST_SEED=20260526`), so reruns reproduce
the exact same indices. CIFAR uses its native test set; EuroSAT has no native test set,
so a 20% test split is carved out deterministically (train 19,440 / val 2,160 /
test 5,400).

**(b) Sanity-check the loaders:**

```bash
python data_loaders.py    # smoke test over cifar10 + eurosat; prints batch counts/shapes
```

---

## Training

### Single run (local or one cluster job)

`model_finetune.py` runs one `FinetuneConfig` end-to-end and saves the best-by-val
checkpoint to `checkpoints/finetune_outputs/<run_name>_best.pt`. Edit the
`FinetuneConfig(...)` at the bottom of the file to choose what runs, then:

```bash
python model_finetune.py
```

Representative config (SimCLR fine-tune protocol, linearly scaled from batch 4096 → 256):

```python
FinetuneConfig(
    dataset="cifar10", num_classes=10, label_fraction=0.01, mode="full",
    lr=0.003125, weight_decay=0.0, momentum=0.9, optimizer="sgd",
    epochs=60, batch_size=256, scheduler="cosine", warmup_epochs=0, seed=42,
)
```

For **linear evaluation**, set `mode="linear_eval"` (freezes the encoder, trains only the
head). Use representation `h` (post-pooling, pre-projection-head), never `z = g(h)`.


## Evaluation

`evaluate.py` rebuilds the model from each checkpoint's stored config, runs the correct
test split automatically (it reads `dataset` out of the checkpoint, so a `cifar10_*`
checkpoint is tested on CIFAR-10 and a `eurosat_*` checkpoint on EuroSAT), and writes a
`<name>_eval.json` next to each checkpoint.

```bash
# one checkpoint
python evaluate.py --checkpoint checkpoints/finetune_outputs/cifar10_full_frac1.0_seed42_best.pt

# a whole folder (one bad checkpoint won't abort the batch)
python evaluate.py --folder checkpoints/finetune_outputs/
```

Each JSON contains `evaluation_metrics` (test loss, top-1, top-5, per-class accuracy),
the full `training_config`, the per-epoch `training_metrics` history, and provenance
(`architecture`, `training_git_commit`, timestamps). These JSONs feed the downstream
plotting of label-efficiency curves.

## Demo script
```
chmod +x demo.sh
./demo.sh
```