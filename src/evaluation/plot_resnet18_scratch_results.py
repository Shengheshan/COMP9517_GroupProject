from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training curves and summarize "
            "Scratch ResNet-18 experiment results."
        )
    )

    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            "results/training/"
            "resnet18_scratch_history.csv"
        ),
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "results/evaluation/"
            "resnet18_scratch_metrics.json"
        ),
    )

    parser.add_argument(
        "--inference-time",
        type=Path,
        default=Path(
            "results/evaluation/"
            "resnet18_scratch_inference_time.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/figures/resnet18_scratch"
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def create_loss_plot(
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training Loss",
        linewidth=2,
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    best_loss_index = (
        history["validation_loss"].idxmin()
    )

    best_loss_epoch = int(
        history.loc[best_loss_index, "epoch"]
    )

    best_validation_loss = float(
        history.loc[
            best_loss_index,
            "validation_loss",
        ]
    )

    plt.scatter(
        [best_loss_epoch],
        [best_validation_loss],
        s=60,
        label=(
            f"Minimum Validation Loss "
            f"(Epoch {best_loss_epoch})"
        ),
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "Scratch ResNet-18 Training and Validation Loss"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_accuracy_plot(
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    train_accuracy_percent = (
        history["train_accuracy"] * 100
    )

    validation_accuracy_percent = (
        history["validation_accuracy"] * 100
    )

    plt.plot(
        history["epoch"],
        train_accuracy_percent,
        label="Training Accuracy",
        linewidth=2,
    )

    plt.plot(
        history["epoch"],
        validation_accuracy_percent,
        label="Validation Accuracy",
        linewidth=2,
    )

    best_accuracy_index = (
        history["validation_accuracy"].idxmax()
    )

    best_epoch = int(
        history.loc[best_accuracy_index, "epoch"]
    )

    best_accuracy_percent = float(
        history.loc[
            best_accuracy_index,
            "validation_accuracy",
        ]
        * 100
    )

    plt.scatter(
        [best_epoch],
        [best_accuracy_percent],
        s=60,
        label=(
            f"Best Validation Accuracy "
            f"(Epoch {best_epoch}: "
            f"{best_accuracy_percent:.2f}%)"
        ),
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Scratch ResNet-18 Training and Validation Accuracy"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_summary_csv(
    summary: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "metric",
            "value",
        ])

        for key, value in summary.items():
            writer.writerow([
                key,
                value,
            ])


def main() -> None:
    args = parse_args()

    if not args.history.exists():
        raise FileNotFoundError(
            f"History file not found: {args.history}"
        )

    history = pd.read_csv(args.history)

    required_columns = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "learning_rate",
        "epoch_seconds",
    }

    missing_columns = (
        required_columns
        - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "History CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    history = history.sort_values(
        by="epoch"
    ).reset_index(drop=True)

    metrics = load_json(args.metrics)
    inference_time = load_json(
        args.inference_time
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loss_plot_path = (
        args.output_dir
        / "resnet18_scratch_loss_curve.png"
    )

    accuracy_plot_path = (
        args.output_dir
        / "resnet18_scratch_accuracy_curve.png"
    )

    create_loss_plot(
        history=history,
        output_path=loss_plot_path,
    )

    create_accuracy_plot(
        history=history,
        output_path=accuracy_plot_path,
    )

    best_accuracy_index = (
        history["validation_accuracy"].idxmax()
    )

    best_loss_index = (
        history["validation_loss"].idxmin()
    )

    best_epoch = int(
        history.loc[
            best_accuracy_index,
            "epoch",
        ]
    )

    best_validation_accuracy = float(
        history.loc[
            best_accuracy_index,
            "validation_accuracy",
        ]
    )

    minimum_validation_loss_epoch = int(
        history.loc[
            best_loss_index,
            "epoch",
        ]
    )

    minimum_validation_loss = float(
        history.loc[
            best_loss_index,
            "validation_loss",
        ]
    )

    final_epoch = int(
        history.iloc[-1]["epoch"]
    )

    final_train_accuracy = float(
        history.iloc[-1]["train_accuracy"]
    )

    final_validation_accuracy = float(
        history.iloc[-1][
            "validation_accuracy"
        ]
    )

    total_training_seconds = float(
        history["epoch_seconds"].sum()
    )

    total_training_minutes = (
        total_training_seconds / 60
    )

    total_training_hours = (
        total_training_seconds / 3600
    )

    generalization_gap = (
        final_train_accuracy
        - final_validation_accuracy
    )

    summary = {
        "experiment": "resnet18_scratch",
        "number_of_epochs_recorded": int(
            len(history)
        ),
        "final_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "best_validation_accuracy_percent": (
            best_validation_accuracy * 100
        ),
        "minimum_validation_loss_epoch": (
            minimum_validation_loss_epoch
        ),
        "minimum_validation_loss": (
            minimum_validation_loss
        ),
        "final_train_accuracy": (
            final_train_accuracy
        ),
        "final_train_accuracy_percent": (
            final_train_accuracy * 100
        ),
        "final_validation_accuracy": (
            final_validation_accuracy
        ),
        "final_validation_accuracy_percent": (
            final_validation_accuracy * 100
        ),
        "final_generalization_gap": (
            generalization_gap
        ),
        "final_generalization_gap_percent": (
            generalization_gap * 100
        ),
        "total_training_seconds": (
            total_training_seconds
        ),
        "total_training_minutes": (
            total_training_minutes
        ),
        "total_training_hours": (
            total_training_hours
        ),
        "test_top1_accuracy": metrics.get(
            "top1_accuracy"
        ),
        "test_top1_accuracy_percent": (
            metrics.get(
                "top1_accuracy",
                0,
            )
            * 100
        ),
        "test_top5_accuracy": metrics.get(
            "top5_accuracy"
        ),
        "test_top5_accuracy_percent": (
            metrics.get(
                "top5_accuracy",
                0,
            )
            * 100
        ),
        "test_macro_precision": metrics.get(
            "macro_precision"
        ),
        "test_macro_recall": metrics.get(
            "macro_recall"
        ),
        "test_macro_f1": metrics.get(
            "macro_f1"
        ),
        "checkpoint_epoch": (
            inference_time.get(
                "checkpoint_epoch"
            )
        ),
        "number_of_test_images": (
            inference_time.get(
                "number_of_test_images"
            )
        ),
        "total_inference_seconds": (
            inference_time.get(
                "total_inference_seconds"
            )
        ),
        "average_seconds_per_image": (
            inference_time.get(
                "average_seconds_per_image"
            )
        ),
        "images_per_second": (
            inference_time.get(
                "images_per_second"
            )
        ),
    }

    summary_json_path = (
        args.output_dir
        / "resnet18_scratch_summary.json"
    )

    summary_csv_path = (
        args.output_dir
        / "resnet18_scratch_summary.csv"
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    save_summary_csv(
        summary=summary,
        output_path=summary_csv_path,
    )

    print("=" * 60)
    print("Scratch ResNet-18 Result Summary")
    print("=" * 60)
    print(f"Recorded epochs: {len(history)}")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )
    print(
        f"Minimum validation loss: "
        f"{minimum_validation_loss:.4f} "
        f"(epoch {minimum_validation_loss_epoch})"
    )
    print(
        f"Total training time: "
        f"{total_training_hours:.2f} hours"
    )
    print(
        f"Test Top-1 accuracy: "
        f"{metrics.get('top1_accuracy', 0) * 100:.2f}%"
    )
    print(
        f"Test Top-5 accuracy: "
        f"{metrics.get('top5_accuracy', 0) * 100:.2f}%"
    )
    print(
        f"Test Macro F1: "
        f"{metrics.get('macro_f1', 0) * 100:.2f}%"
    )
    print(f"Loss curve: {loss_plot_path}")
    print(f"Accuracy curve: {accuracy_plot_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()