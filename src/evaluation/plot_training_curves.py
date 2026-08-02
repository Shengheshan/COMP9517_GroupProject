from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRATCH_HISTORY_PATH = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "resnet18_scratch_aligned_history.csv"
)

FROZEN_HISTORY_PATH = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "resnet18_frozen_aligned_history.csv"
)

FINETUNE_HISTORY_PATH = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "resnet18_finetune_aligned_history.csv"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_history(file_path: Path) -> pd.DataFrame:
    """Load one training-history CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Training history file not found: {file_path}"
        )

    history = pd.read_csv(file_path)

    if "epoch" not in history.columns:
        raise ValueError(
            f"The history file has no 'epoch' column: {file_path}"
        )

    return history


def prepare_scratch_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert Scratch ResNet-18 history columns into a common format.
    """
    required_columns = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    }

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            "Scratch history is missing columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = pd.DataFrame(
        {
            "epoch": history["epoch"],
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "validation_loss": history["validation_loss"],
            "validation_accuracy": history[
                "validation_accuracy"
            ],
            "validation_macro_f1": history[
                "validation_macro_f1"
            ],
        }
    )

    prepared["model"] = "Scratch ResNet-18"

    return prepared


def prepare_transfer_history(
    history: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Convert Frozen or Fine-tuned history columns into a common format.
    """
    required_columns = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_top1",
        "validation_macro_f1",
    }

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            f"{model_name} history is missing columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = pd.DataFrame(
        {
            "epoch": history["epoch"],
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "validation_loss": history["validation_loss"],
            "validation_accuracy": history[
                "validation_top1"
            ],
            "validation_macro_f1": history[
                "validation_macro_f1"
            ],
        }
    )

    prepared["model"] = model_name

    return prepared


def convert_fraction_to_percent(
    values: pd.Series,
) -> pd.Series:
    """
    Convert values from 0-1 fractions to percentages when necessary.
    """
    if values.dropna().empty:
        return values

    if values.max() <= 1.0:
        return values * 100

    return values


def plot_metric_comparison(
    histories: list[pd.DataFrame],
    metric_column: str,
    title: str,
    y_label: str,
    output_path: Path,
    convert_to_percent: bool = False,
) -> None:
    """Plot one training or validation metric for all models."""
    figure, axis = plt.subplots(figsize=(10, 6))

    for history in histories:
        values = history[metric_column]

        if convert_to_percent:
            values = convert_fraction_to_percent(values)

        axis.plot(
            history["epoch"],
            values,
            marker="o",
            markersize=3,
            label=history["model"].iloc[0],
        )

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.grid(
        visible=True,
        linestyle="--",
        alpha=0.4,
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scratch_raw = load_history(
        SCRATCH_HISTORY_PATH
    )

    frozen_raw = load_history(
        FROZEN_HISTORY_PATH
    )

    finetune_raw = load_history(
        FINETUNE_HISTORY_PATH
    )

    scratch_history = prepare_scratch_history(
        scratch_raw
    )

    frozen_history = prepare_transfer_history(
        frozen_raw,
        model_name="Frozen ResNet-18",
    )

    finetune_history = prepare_transfer_history(
        finetune_raw,
        model_name="Fine-tuned ResNet-18",
    )

    histories = [
        scratch_history,
        frozen_history,
        finetune_history,
    ]

    plot_metric_comparison(
        histories=histories,
        metric_column="train_loss",
        title="Training Loss Comparison",
        y_label="Training Loss",
        output_path=(
            FIGURES_DIR
            / "training_loss_comparison.png"
        ),
    )

    plot_metric_comparison(
        histories=histories,
        metric_column="validation_loss",
        title="Validation Loss Comparison",
        y_label="Validation Loss",
        output_path=(
            FIGURES_DIR
            / "validation_loss_comparison.png"
        ),
    )

    plot_metric_comparison(
        histories=histories,
        metric_column="train_accuracy",
        title="Training Accuracy Comparison",
        y_label="Training Accuracy (%)",
        output_path=(
            FIGURES_DIR
            / "training_accuracy_comparison.png"
        ),
        convert_to_percent=True,
    )

    plot_metric_comparison(
        histories=histories,
        metric_column="validation_accuracy",
        title="Validation Accuracy Comparison",
        y_label="Validation Accuracy (%)",
        output_path=(
            FIGURES_DIR
            / "validation_accuracy_comparison.png"
        ),
        convert_to_percent=True,
    )

    plot_metric_comparison(
        histories=histories,
        metric_column="validation_macro_f1",
        title="Validation Macro-F1 Comparison",
        y_label="Validation Macro-F1 (%)",
        output_path=(
            FIGURES_DIR
            / "validation_macro_f1_comparison.png"
        ),
        convert_to_percent=True,
    )


if __name__ == "__main__":
    main()