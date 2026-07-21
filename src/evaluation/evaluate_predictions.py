import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_metrics


REQUIRED_COLUMNS = {
    "relative_path",
    "true_label",
    "predicted_label",
    "top5_labels",
}


def parse_top5_labels(value: str) -> list[int]:
    parsed = ast.literal_eval(value)

    if not isinstance(parsed, list):
        raise ValueError(
            f"top5_labels must be a list, received: {value}"
        )

    if len(parsed) != 5:
        raise ValueError(
            f"Expected 5 labels, received {len(parsed)}: {value}"
        )

    return [int(label) for label in parsed]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="CSV file containing model predictions.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=500,
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)

    missing_columns = REQUIRED_COLUMNS - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    true_labels = predictions["true_label"].astype(int).tolist()
    predicted_labels = (
        predictions["predicted_label"].astype(int).tolist()
    )

    top5_predictions = [
        parse_top5_labels(value)
        for value in predictions["top5_labels"]
    ]

    metrics = calculate_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        top5_predictions=top5_predictions,
        num_classes=args.num_classes,
    )

    confusion_matrix = metrics.pop("confusion_matrix")

    experiment_name = args.predictions.stem

    metrics_path = (
        args.output_dir / f"{experiment_name}_metrics.json"
    )

    matrix_path = (
        args.output_dir
        / f"{experiment_name}_confusion_matrix.npy"
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    np.save(matrix_path, confusion_matrix)

    print(f"Experiment: {experiment_name}")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {matrix_path}")


if __name__ == "__main__":
    main()