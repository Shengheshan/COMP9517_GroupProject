import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = PROJECT_ROOT / "results" / "evaluation"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


TIMING_FILES = {
    "HOG + SVM": (
        EVALUATION_DIR
        / "hog_svm_timing.json"
    ),
    "Scratch ResNet-18": (
        EVALUATION_DIR
        / "resnet18_scratch_aligned_inference_time.json"
    ),
    "Frozen ResNet-18": (
        EVALUATION_DIR
        / "resnet18_frozen_aligned_timing.json"
    ),
    "Fine-tuned ResNet-18": (
        EVALUATION_DIR
        / "resnet18_finetune_aligned_timing.json"
    ),
}


def load_json(file_path: Path) -> dict[str, Any]:
    """Load one timing JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Timing file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_numeric_value(
    data: dict[str, Any],
    key: str,
) -> float | None:
    """Read one numeric value safely."""
    value = data.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_hog_timing(
    data: dict[str, Any],
) -> dict[str, float | str | None]:
    """Extract timing values from the nested HOG JSON format."""
    inference = data.get("inference", {})

    total_time = get_numeric_value(
        inference,
        "total_inference_seconds",
    )

    average_ms = get_numeric_value(
        inference,
        "milliseconds_per_image",
    )

    sample_count = get_numeric_value(
        inference,
        "number_of_test_images",
    )

    if average_ms is None:
        seconds_per_image = get_numeric_value(
            inference,
            "seconds_per_image",
        )

        if seconds_per_image is not None:
            average_ms = seconds_per_image * 1000

    if average_ms is None:
        if total_time is not None and sample_count:
            average_ms = (
                total_time / sample_count * 1000
            )

    return {
        "Model": "HOG + SVM",
        "Device": str(data.get("device", "Unknown")),
        "Sample Count": sample_count,
        "Total Inference Time (s)": total_time,
        "Average Time per Image (ms)": average_ms,
    }


def extract_scratch_timing(
    data: dict[str, Any],
) -> dict[str, float | str | None]:
    """Extract timing values from the Scratch ResNet JSON."""
    total_time = get_numeric_value(
        data,
        "total_inference_seconds",
    )

    average_seconds = get_numeric_value(
        data,
        "average_seconds_per_image",
    )

    sample_count = get_numeric_value(
        data,
        "number_of_test_images",
    )

    average_ms = (
        average_seconds * 1000
        if average_seconds is not None
        else None
    )

    if average_ms is None:
        if total_time is not None and sample_count:
            average_ms = (
                total_time / sample_count * 1000
            )

    return {
        "Model": "Scratch ResNet-18",
        "Device": str(data.get("device", "Unknown")),
        "Sample Count": sample_count,
        "Total Inference Time (s)": total_time,
        "Average Time per Image (ms)": average_ms,
    }


def extract_transfer_timing(
    data: dict[str, Any],
    model_name: str,
) -> dict[str, float | str | None]:
    """
    Extract timing values from Frozen or Fine-tuned ResNet JSON.

    These files record GPU forward-pass time only.
    """
    total_time = get_numeric_value(
        data,
        "total_gpu_forward_time_seconds",
    )

    average_seconds = get_numeric_value(
        data,
        "average_gpu_forward_time_seconds_per_image",
    )

    sample_count = get_numeric_value(
        data,
        "number_of_images",
    )

    average_ms = (
        average_seconds * 1000
        if average_seconds is not None
        else None
    )

    if average_ms is None:
        if total_time is not None and sample_count:
            average_ms = (
                total_time / sample_count * 1000
            )

    device = str(data.get("device", "Unknown"))

    gpu_name = data.get("gpu")

    if gpu_name:
        device = f"{device}: {gpu_name}"

    return {
        "Model": model_name,
        "Device": device,
        "Sample Count": sample_count,
        "Total Inference Time (s)": total_time,
        "Average Time per Image (ms)": average_ms,
    }


def extract_timing(
    model_name: str,
    data: dict[str, Any],
) -> dict[str, float | str | None]:
    """Select the correct parser for each model."""
    if model_name == "HOG + SVM":
        return extract_hog_timing(data)

    if model_name == "Scratch ResNet-18":
        return extract_scratch_timing(data)

    return extract_transfer_timing(
        data=data,
        model_name=model_name,
    )


def add_value_labels(
    axis,
    decimal_places: int,
) -> None:
    """Add numeric values above each bar."""
    for bar in axis.patches:
        height = bar.get_height()

        if pd.isna(height):
            continue

        axis.annotate(
            f"{height:.{decimal_places}f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_runtime_chart(
    data: pd.DataFrame,
    value_column: str,
    title: str,
    y_label: str,
    output_path: Path,
    decimal_places: int,
) -> None:
    """Create and save one runtime comparison chart."""
    valid_data = data.dropna(
        subset=[value_column]
    ).copy()

    if valid_data.empty:
        print(
            f"Skipped {output_path.name}: "
            f"no valid values found."
        )
        return

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        valid_data["Model"],
        valid_data[value_column],
    )

    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel(y_label)

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    add_value_labels(
        axis=axis,
        decimal_places=decimal_places,
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

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for model_name, timing_path in TIMING_FILES.items():
        timing_data = load_json(timing_path)

        rows.append(
            extract_timing(
                model_name=model_name,
                data=timing_data,
            )
        )

    runtime_table = pd.DataFrame(rows)

    output_columns = [
        "Model",
        "Device",
        "Sample Count",
        "Total Inference Time (s)",
        "Average Time per Image (ms)",
    ]

    runtime_table = runtime_table[output_columns]

    table_path = (
        TABLES_DIR
        / "runtime_comparison.csv"
    )

    runtime_table.to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRuntime comparison")
    print("------------------")
    print(
        runtime_table.round(4).to_string(
            index=False
        )
    )

    print(f"\nSaved: {table_path}")

    plot_runtime_chart(
        data=runtime_table,
        value_column="Total Inference Time (s)",
        title="Total Inference Time Comparison",
        y_label="Total Time (seconds)",
        output_path=(
            FIGURES_DIR
            / "total_inference_time_comparison.png"
        ),
        decimal_places=2,
    )

    plot_runtime_chart(
        data=runtime_table,
        value_column="Average Time per Image (ms)",
        title="Average Inference Time per Image",
        y_label="Time per Image (milliseconds)",
        output_path=(
            FIGURES_DIR
            / "average_inference_time_comparison.png"
        ),
        decimal_places=3,
    )


if __name__ == "__main__":
    main()