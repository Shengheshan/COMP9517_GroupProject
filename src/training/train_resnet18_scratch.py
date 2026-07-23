from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import INatDataset
from models.resnet18_scratch import build_resnet18_scratch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    validation_loss = total_loss / total_samples
    validation_accuracy = total_correct / total_samples

    return validation_loss, validation_accuracy


def save_history(
    history: list[dict[str, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "learning_rate",
        "epoch_seconds",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a randomly initialized ResNet-18."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing train_mini and val.",
    )

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data_metadata/train.csv"),
    )

    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=Path("data_metadata/validation.csv"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints/resnet18_scratch_best.pth"),
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path(
            "results/training/resnet18_scratch_history.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = INatDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        transform=train_transform,
    )

    validation_dataset = INatDataset(
        csv_path=args.validation_csv,
        data_root=args.data_root,
        transform=evaluation_transform,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_resnet18_scratch(
        num_classes=500
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = StepLR(
        optimizer,
        step_size=10,
        gamma=0.1,
    )

    args.checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: list[dict[str, float]] = []
    best_validation_accuracy = -1.0

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {validation_loss:.4f} | "
            f"val acc: {validation_accuracy:.4f} | "
            f"time: {epoch_seconds:.1f}s"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "num_classes": 500,
                    "seed": args.seed,
                },
                args.checkpoint_path,
            )

            print(
                f"Saved best checkpoint to "
                f"{args.checkpoint_path}"
            )

        save_history(
            history=history,
            output_path=args.history_path,
        )

        scheduler.step()

    training_seconds = time.perf_counter() - training_start

    print("Training completed.")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )
    print(
        f"Total training time: "
        f"{training_seconds:.1f} seconds"
    )


if __name__ == "__main__":
    main()