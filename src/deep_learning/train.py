# Member 4: Transfer learning training pipeline
# This file trains and validates ImageNet-pretrained ResNet-18 using:
# 1. Frozen backbone with only the final classification layer trained
# 2. Full-network fine-tuning
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from src.data.dataset import INatDataset
from src.deep_learning.model import (
    create_resnet18,
    get_trainable_parameters,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_transforms():
    weights = ResNet18_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.7, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, evaluation_transform


def create_loaders(args):
    train_transform, evaluation_transform = create_transforms()

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

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    return train_loader, validation_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    mode,
):
    if mode == "frozen":
        model.eval()
        model.fc.train()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    all_labels = []
    all_predictions = []

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        with autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        top5_predictions = torch.topk(
            outputs,
            k=5,
            dim=1,
        ).indices

        top1_predictions = top5_predictions[:, 0]

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        top1_correct += (
            top1_predictions == labels
        ).sum().item()

        top5_correct += (
            top5_predictions == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

        all_labels.extend(
            labels.cpu().tolist()
        )
        all_predictions.extend(
            top1_predictions.cpu().tolist()
        )

    average_loss = total_loss / total_samples
    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        average_loss,
        top1_accuracy,
        top5_accuracy,
        macro_f1,
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    args,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "mode": args.mode,
        "num_classes": args.num_classes,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "arguments": vars(args),
    }

    torch.save(checkpoint, path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["frozen", "finetune"],
        required=True,
    )

    parser.add_argument(
        "--data-root",
        default=".",
    )

    parser.add_argument(
        "--train-csv",
        default="data_metadata/train.csv",
    )

    parser.add_argument(
        "--validation-csv",
        default="data_metadata/validation.csv",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    set_seed(args.seed)

    if args.learning_rate is None:
        if args.mode == "frozen":
            args.learning_rate = 1e-3
        else:
            args.learning_rate = 1e-4

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")

    train_loader, validation_loader = create_loaders(args)

    print(f"Training images: {len(train_loader.dataset)}")
    print(
        "Validation images:",
        len(validation_loader.dataset),
    )

    model = create_resnet18(
        num_classes=args.num_classes,
        mode=args.mode,
    ).to(device)

    trainable_parameters = get_trainable_parameters(model)

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Total parameters: {total_count:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = GradScaler(
        device=device.type,
        enabled=device.type == "cuda",
    )

    model_directory = Path("models")
    log_directory = Path("results/logs")

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_name = f"resnet18_{args.mode}"

    checkpoint_path = (
        model_directory
        / f"{experiment_name}_best.pth"
    )

    history_path = (
        log_directory
        / f"{experiment_name}_history.csv"
    )

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    total_training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mode=args.mode,
        )

        (
            validation_loss,
            validation_top1,
            validation_top5,
            validation_macro_f1,
        ) = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(validation_macro_f1)

        epoch_results = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_top1": validation_top1,
            "validation_top5": validation_top5,
            "validation_macro_f1": validation_macro_f1,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        }

        history.append(epoch_results)
        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f} | "
            f"val top1 {validation_top1:.4f} | "
            f"val top5 {validation_top5:.4f} | "
            f"val macro-F1 {validation_macro_f1:.4f} | "
            f"time {epoch_time:.1f}s"
        )

        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_results,
                args=args,
                path=checkpoint_path,
            )

            print(
                "Saved best checkpoint:",
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "Early stopping after",
                args.patience,
                "epochs without improvement.",
            )
            break

    total_training_time = (
        time.perf_counter()
        - total_training_start
    )

    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print(
        f"Total training time: "
        f"{total_training_time:.1f} seconds"
    )
    print(f"History saved to: {history_path}")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()