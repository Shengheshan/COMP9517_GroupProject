from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "results" / "logs"
EVAL_DIR = ROOT / "results" / "evaluation"
FIGURE_DIR = ROOT / "results" / "figures"/ "resnet18_frozen"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, candidates):
    """Find the first matching column name."""
    for name in candidates:
        if name in df.columns:
            return name

    raise KeyError(
        f"Cannot find any of these columns: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def plot_training_curves(mode):
    history_path = LOG_DIR / f"resnet18_{mode}_aligned_history.csv"
    df = pd.read_csv(history_path)

    epoch_col = find_column(df, ["epoch", "Epoch"])
    train_loss_col = find_column(df, ["train_loss", "training_loss"])
    val_loss_col = find_column(df, ["val_loss", "validation_loss"])
    train_acc_col = find_column(
        df,
        ["train_accuracy", "train_acc", "training_accuracy"]
    )
    val_top1_col = find_column(
        df,
        ["val_top1_accuracy", "val_top1", "validation_top1"]
    )
    val_f1_col = find_column(
        df,
        ["val_macro_f1", "validation_macro_f1", "macro_f1"]
    )

    epochs = df[epoch_col]

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df[train_loss_col], label="Training loss")
    plt.plot(epochs, df[val_loss_col], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"ResNet-18 {mode.capitalize()}: Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"resnet18_{mode}_aligned_loss.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Accuracy and Macro-F1 curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        df[train_acc_col] * 100,
        label="Training accuracy"
    )
    plt.plot(
        epochs,
        df[val_top1_col] * 100,
        label="Validation Top-1"
    )
    plt.plot(
        epochs,
        df[val_f1_col] * 100,
        label="Validation Macro-F1"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title(f"ResNet-18 {mode.capitalize()}: Accuracy and Macro-F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"resnet18_{mode}_aligned_scores.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def load_metrics(mode):
    metrics_path = (
        EVAL_DIR / f"resnet18_{mode}_aligned_metrics.json"
    )

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_model_comparison():
    frozen = load_metrics("frozen")
    finetune = load_metrics("finetune")

    metric_keys = [
        "top1_accuracy",
        "top5_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    labels = [
        "Top-1",
        "Top-5",
        "Macro\nPrecision",
        "Macro\nRecall",
        "Macro-F1",
    ]

    frozen_values = [
        frozen[key] * 100 for key in metric_keys
    ]
    finetune_values = [
        finetune[key] * 100 for key in metric_keys
    ]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 6))
    frozen_bars = plt.bar(
        x - width / 2,
        frozen_values,
        width,
        label="Frozen backbone"
    )
    finetune_bars = plt.bar(
        x + width / 2,
        finetune_values,
        width,
        label="Full fine-tuning"
    )

    plt.ylabel("Score (%)")
    plt.title("ResNet-18 Transfer Learning Test Results")
    plt.xticks(x, labels)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.bar_label(
        frozen_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8
    )
    plt.bar_label(
        finetune_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "resnet18_aligned_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def main():
    plot_training_curves("frozen")
    plot_training_curves("finetune")
    plot_model_comparison()

    print("Figures saved to:", FIGURE_DIR)

    for path in sorted(FIGURE_DIR.glob("resnet18_*aligned*.png")):
        print(path)


if __name__ == "__main__":
    main()