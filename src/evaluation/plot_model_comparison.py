from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLE_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "model_comparison.csv"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_comparison_table(file_path: Path) -> pd.DataFrame:
    """Load the model comparison table."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Comparison table not found: {file_path}"
        )

    return pd.read_csv(file_path)


def add_value_labels(ax) -> None:
    """Add numeric labels above each bar."""
    for bar in ax.patches:
        height = bar.get_height()

        ax.annotate(
            f"{height:.2f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_single_metric(
    data: pd.DataFrame,
    metric_column: str,
    title: str,
    output_name: str,
) -> None:
    """Create and save one model-comparison bar chart."""
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        data["Model"],
        data[metric_column],
    )

    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel("Percentage (%)")
    axis.set_ylim(0, 100)

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    add_value_labels(axis)

    figure.tight_layout()

    output_path = FIGURES_DIR / output_name
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Figure saved to: {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_table = load_comparison_table(TABLE_PATH)

    required_columns = {
        "Model",
        "Top-1 Accuracy (%)",
        "Top-5 Accuracy (%)",
        "Macro F1 (%)",
    }

    missing_columns = (
        required_columns - set(comparison_table.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    plot_single_metric(
        data=comparison_table,
        metric_column="Top-1 Accuracy (%)",
        title="Top-1 Accuracy Comparison",
        output_name="top1_accuracy_comparison.png",
    )

    plot_single_metric(
        data=comparison_table,
        metric_column="Top-5 Accuracy (%)",
        title="Top-5 Accuracy Comparison",
        output_name="top5_accuracy_comparison.png",
    )

    plot_single_metric(
        data=comparison_table,
        metric_column="Macro F1 (%)",
        title="Macro F1 Comparison",
        output_name="macro_f1_comparison.png",
    )


if __name__ == "__main__":
    main()