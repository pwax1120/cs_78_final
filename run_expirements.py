from model_finetune import train, FinetuneConfig

experiments = [
    FinetuneConfig(dataset="cifar10",  label_fraction=0.01, mode="linear_probe", lr=0.1, epochs=90, seed=42, ...),
    FinetuneConfig(dataset="cifar10",  label_fraction=0.10, mode="linear_probe", lr=0.1, epochs=90, seed=42, ...),
    FinetuneConfig(dataset="cifar10",  label_fraction=1.00, mode="linear_probe", lr=0.1, epochs=90, seed=42, ...),
    FinetuneConfig(dataset="cifar10",  label_fraction=0.01, mode="full",         lr=0.01, epochs=60, seed=42, ...),
    # ... etc
]

for config in experiments:
    print(f"\n{'='*60}\nRunning: {config.run_name or 'unnamed'}\n{'='*60}")
    checkpoint_path = train(config)
    print(f"Done. Checkpoint: {checkpoint_path}")