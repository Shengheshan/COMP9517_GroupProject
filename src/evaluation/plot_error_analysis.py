from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

WORST_CLASSES_PATH = (
    TABLES_DIR
    / "resnet18_finetune_worst_classes.csv"
)

CONFUSED_PAIRS_PATH = (
    TABLES_DIR
    / "resnet18_finetune_top_confused_pairs.csv"
)


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file and check that it exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required CSV file not found: {file_path}"
        )

    return pd.read_csv(file_path)


def plot_worst_classes(
    data: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot the classes with the lowest classification accuracy."""
    required_columns = {
        "true_species",
        "accuracy_percent",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Worst-classes CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    plot_data = (
        data.nsmallest(top_n, "accuracy_percent")
        .sort_values(
            by="accuracy_percent",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    bars = axis.barh(
        plot_data["true_species"],
        plot_data["accuracy_percent"],
    )

    axis.set_title(
        "Lowest-performing Species "
        "(Fine-tuned ResNet-18)"
    )
    axis.set_xlabel("Class Accuracy (%)")
    axis.set_ylabel("Species")
    axis.set_xlim(0, 100)

    for bar, value in zip(
        bars,
        plot_data["accuracy_percent"],
    ):
        axis.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
        )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved: {output_path}")


def plot_confused_pairs(
    data: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot the most frequent true-to-predicted class errors."""
    required_columns = {
        "true_species",
        "predicted_species",
        "error_count",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Confused-pairs CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    plot_data = (
        data.nlargest(top_n, "error_count")
        .copy()
        .sort_values(
            by="error_count",
            ascending=True,
        )
    )

    plot_data["pair_name"] = (
        plot_data["true_species"]
        + " → "
        + plot_data["predicted_species"]
    )

    figure, axis = plt.subplots(figsize=(12, 7))

    bars = axis.barh(
        plot_data["pair_name"],
        plot_data["error_count"],
    )

    axis.set_title(
        "Most Frequent Confused Species Pairs "
        "(Fine-tuned ResNet-18)"
    )
    axis.set_xlabel("Number of Misclassifications")
    axis.set_ylabel("True Species → Predicted Species")

    for bar, value in zip(
        bars,
        plot_data["error_count"],
    ):
        axis.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(int(value)),
            va="center",
        )

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

    worst_classes = load_csv(
        WORST_CLASSES_PATH
    )

    confused_pairs = load_csv(
        CONFUSED_PAIRS_PATH
    )

    plot_worst_classes(
        data=worst_classes,
        output_path=(
            FIGURES_DIR
            / "resnet18_finetune_worst_classes.png"
        ),
    )

    plot_confused_pairs(
        data=confused_pairs,
        output_path=(
            FIGURES_DIR
            / "resnet18_finetune_top_confused_pairs.png"
        ),
    )


if __name__ == "__main__":
    main()