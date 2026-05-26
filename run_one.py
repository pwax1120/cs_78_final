from model_finetune import train, FinetuneConfig

config = FinetuneConfig(
    dataset="cifar10",
    num_classes=10,
    label_fraction=0.01,
    mode="full",
    lr=0.003125,
    weight_decay=0.0,
    momentum=0.9,
    optimizer="sgd",
    epochs=60,
    batch_size=256,
    scheduler="cosine",
    warmup_epochs=0,
    seed=42,
)
train(config)
