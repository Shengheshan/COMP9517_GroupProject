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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "model_comparison_table.png"
)


def main() -> None:
    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Model comparison table not found: {TABLE_PATH}"
        )

    data = pd.read_csv(TABLE_PATH)

    numeric_columns = [
        column
        for column in data.columns
        if column != "Model"
    ]

    data[numeric_columns] = data[numeric_columns].round(2)

    formatted_data = data.copy()

    for column in numeric_columns:
        formatted_data[column] = formatted_data[column].map(
            lambda value: f"{value:.2f}"
        )

    figure, axis = plt.subplots(figsize=(14, 3.8))
    axis.axis("off")

    table = axis.table(
        cellText=formatted_data.values,
        colLabels=formatted_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for column_index in range(len(formatted_data.columns)):
        table[(0, column_index)].set_text_props(
            weight="bold"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()