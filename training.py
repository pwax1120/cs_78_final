import argparse
from model_finetune import train, FinetuneConfig

CONFIGS = [
    '''
    #CIFAR-10 FULL FINETUNE (5 seeds x 3 label fractions x 2 modes = 30 runs)
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="cifar10", num_classes=10, label_fraction=1.0, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    '''
    #EURO-SAT FULL FINETUNE (Same structure as cifar-10)

    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="full", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.01, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=0.1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=0, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=1, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=2, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=3, device="auto", num_workers=0),
    FinetuneConfig(dataset="eurosat", num_classes=10, label_fraction=1, mode="linear_eval", lr=0.003125, momentum=0.9, weight_decay=0.0, optimizer="sgd", epochs=60, batch_size=256, scheduler="cosine", seed=4, device="auto", num_workers=0),

    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    args = parser.parse_args()
    train(CONFIGS[args.run_id])