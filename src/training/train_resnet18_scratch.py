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
from tqdm.auto import tqdm

from src.data.dataset import INatDataset
from src.models.resnet18_scratch import build_resnet18_scratch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} - Training",
        unit="batch",
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

        running_loss = total_loss / total_samples
        running_accuracy = total_correct / total_samples

        progress_bar.set_postfix(
            loss=f"{running_loss:.4f}",
            accuracy=f"{running_accuracy:.4f}",
        )

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Evaluate the model on the validation dataset."""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} - Validation",
        unit="batch",
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

        running_loss = total_loss / total_samples
        running_accuracy = total_correct / total_samples

        progress_bar.set_postfix(
            loss=f"{running_loss:.4f}",
            accuracy=f"{running_accuracy:.4f}",
        )

    validation_loss = total_loss / total_samples
    validation_accuracy = total_correct / total_samples

    return validation_loss, validation_accuracy


def load_existing_history(
    history_path: Path,
    start_epoch: int,
) -> list[dict[str, float | int]]:
    """
    Load history from previous epochs when resuming.

    Rows from epochs before start_epoch are preserved.
    """

    if not history_path.exists():
        return []

    history: list[dict[str, float | int]] = []

    with history_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            epoch = int(row["epoch"])

            if epoch >= start_epoch:
                continue

            history.append({
                "epoch": epoch,
                "train_loss": float(row["train_loss"]),
                "train_accuracy": float(
                    row["train_accuracy"]
                ),
                "validation_loss": float(
                    row["validation_loss"]
                ),
                "validation_accuracy": float(
                    row["validation_accuracy"]
                ),
                "learning_rate": float(
                    row["learning_rate"]
                ),
                "epoch_seconds": float(
                    row["epoch_seconds"]
                ),
            })

    return history


def save_history(
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Save the complete training history to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "learning_rate",
        "epoch_seconds",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a randomly initialized ResNet-18 "
            "for 500-class image classification."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Dataset root directory containing "
            "train_mini and val."
        ),
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
        default=5,
        help=(
            "Final epoch number. When resuming from "
            "epoch 5 with --epochs 10, epochs 6-10 run."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "checkpoints/resnet18_scratch_best.pth"
        ),
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path(
            "results/training/"
            "resnet18_scratch_history.csv"
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Checkpoint file used to continue training."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the full training process."""

    args = parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Scratch ResNet-18 Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Final epoch: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Resume checkpoint: {args.resume}")
    print("=" * 60)

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

    print(f"Training images: {len(train_dataset)}")
    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )
    print("=" * 60)

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
        num_classes=500,
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

    start_epoch = 1
    best_epoch = 0
    best_validation_accuracy = -1.0

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: "
                f"{args.resume}"
            )

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["epoch"])

        best_validation_accuracy = float(
            checkpoint["validation_accuracy"]
        )

        print()
        print("=" * 60)
        print(f"Resumed from: {args.resume}")
        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )
        print(f"Starting from epoch: {start_epoch}")
        print(
            f"Previous best validation accuracy: "
            f"{best_validation_accuracy:.4f}"
        )
        print("=" * 60)
        print()

    if start_epoch > args.epochs:
        print(
            f"Checkpoint is already at epoch "
            f"{start_epoch - 1}, but --epochs is "
            f"{args.epochs}."
        )
        print("No additional training is required.")
        return

    history = load_existing_history(
        history_path=args.history_path,
        start_epoch=start_epoch,
    )

    training_start = time.perf_counter()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_start = time.perf_counter()

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": (
                validation_accuracy
            ),
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        })

        save_history(
            history=history,
            output_path=args.history_path,
        )

        print()
        print("-" * 60)
        print(f"Epoch {epoch}/{args.epochs} completed")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )
        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )
        print(
            f"Epoch time: "
            f"{epoch_seconds / 60:.2f} minutes"
        )
        print(f"Learning rate: {current_lr:.6f}")

        scheduler.step()

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "scheduler_state_dict": (
                        scheduler.state_dict()
                    ),
                    "validation_accuracy": (
                        validation_accuracy
                    ),
                    "validation_loss": (
                        validation_loss
                    ),
                    "num_classes": 500,
                    "seed": args.seed,
                },
                args.checkpoint_path,
            )

            print(
                "New best model saved to: "
                f"{args.checkpoint_path}"
            )

        print("-" * 60)
        print()

    training_seconds = (
        time.perf_counter() - training_start
    )

    print("=" * 60)
    print("Training completed")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )
    print(
        f"Additional training time: "
        f"{training_seconds / 60:.2f} minutes"
    )
    print(
        f"Checkpoint: "
        f"{args.checkpoint_path}"
    )
    print(
        f"Training history: "
        f"{args.history_path}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()