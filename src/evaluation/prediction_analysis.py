import argparse
import ast
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "resnet18_finetune_aligned.csv"
)

DEFAULT_LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "data_metadata"
    / "label_to_category.json"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"


REQUIRED_COLUMNS = {
    "relative_path",
    "true_label",
    "predicted_label",
    "top5_labels",
    "top5_scores",
}


def parse_list(value: object) -> list:
    """
    Parse a Python-style list stored inside a CSV cell.

    Examples:
        "[12, 30, 8, 15, 91]"
        "[0.81, 0.10, 0.04, 0.03, 0.02]"
    """
    if isinstance(value, list):
        return value

    parsed = ast.literal_eval(str(value))

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list, received: {value}")

    return parsed


def load_label_mapping(file_path: Path) -> dict[int, str]:
    """
    Load the label-to-category mapping and extract only the species name.

    The JSON values may be dictionaries containing taxonomy information,
    for example:
        {
            "id": 9083,
            "species name": "Nymphaea alba",
            "kingdom": "Plantae",
            ...
        }
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Label mapping file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    label_mapping: dict[int, str] = {}

    for label, category_info in raw_mapping.items():
        if isinstance(category_info, dict):
            species_name = (
                category_info.get("species name")
                or category_info.get("species_name")
                or category_info.get("name")
                or f"Class {label}"
            )
        else:
            species_name = str(category_info)

        label_mapping[int(label)] = str(species_name)

    return label_mapping


def simplify_category_name(category: str) -> str:
    """Return a clean species name suitable for tables and reports."""
    return str(category).replace("_", " ").strip()


def prepare_predictions(
    predictions_path: Path,
    label_mapping: dict[int, str],
) -> pd.DataFrame:
    """Load and prepare the model prediction CSV."""
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}"
        )

    predictions = pd.read_csv(predictions_path)

    missing_columns = REQUIRED_COLUMNS - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    predictions["true_label"] = (
        predictions["true_label"].astype(int)
    )

    predictions["predicted_label"] = (
        predictions["predicted_label"].astype(int)
    )

    predictions["top5_labels_parsed"] = (
        predictions["top5_labels"].apply(parse_list)
    )

    predictions["top5_scores_parsed"] = (
        predictions["top5_scores"].apply(parse_list)
    )

    predictions["confidence"] = predictions[
        "top5_scores_parsed"
    ].apply(
        lambda scores: float(scores[0]) if scores else float("nan")
    )

    predictions["is_correct"] = (
        predictions["true_label"]
        == predictions["predicted_label"]
    )

    predictions["true_category"] = predictions[
        "true_label"
    ].map(label_mapping)

    predictions["predicted_category"] = predictions[
        "predicted_label"
    ].map(label_mapping)

    predictions["true_species"] = predictions[
        "true_category"
    ].fillna(
        predictions["true_label"].apply(
            lambda label: f"Class {label}"
        )
    ).apply(simplify_category_name)

    predictions["predicted_species"] = predictions[
        "predicted_category"
    ].fillna(
        predictions["predicted_label"].apply(
            lambda label: f"Class {label}"
        )
    ).apply(simplify_category_name)

    return predictions


def calculate_class_accuracy(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate sample count and accuracy for every class."""
    class_summary = (
        predictions.groupby(
            [
                "true_label",
                "true_species",
            ],
            dropna=False,
        )
        .agg(
            sample_count=("is_correct", "size"),
            correct_count=("is_correct", "sum"),
        )
        .reset_index()
    )

    class_summary["incorrect_count"] = (
        class_summary["sample_count"]
        - class_summary["correct_count"]
    )

    class_summary["accuracy_percent"] = (
        class_summary["correct_count"]
        / class_summary["sample_count"]
        * 100
    )

    return class_summary.sort_values(
        by=[
            "accuracy_percent",
            "sample_count",
        ],
        ascending=[
            True,
            False,
        ],
    )


def calculate_confused_pairs(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Count the most frequent true-to-predicted class errors."""
    errors = predictions.loc[
        ~predictions["is_correct"]
    ].copy()

    confused_pairs = (
        errors.groupby(
            [
                "true_label",
                "true_species",
                "predicted_label",
                "predicted_species",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="error_count")
    )

    return confused_pairs.sort_values(
        by="error_count",
        ascending=False,
    )


def select_error_examples(
    predictions: pd.DataFrame,
    number_of_examples: int,
) -> pd.DataFrame:
    """
    Select representative high-confidence incorrect predictions.
    """
    errors = predictions.loc[
        ~predictions["is_correct"]
    ].copy()

    errors = errors.sort_values(
        by="confidence",
        ascending=False,
    )

    selected_columns = [
        "relative_path",
        "true_label",
        "true_species",
        "predicted_label",
        "predicted_species",
        "confidence",
        "top5_labels",
        "top5_scores",
    ]

    return errors[selected_columns].head(number_of_examples)


def select_correct_examples(
    predictions: pd.DataFrame,
    number_of_examples: int,
) -> pd.DataFrame:
    """
    Select representative high-confidence correct predictions.
    """
    correct = predictions.loc[
        predictions["is_correct"]
    ].copy()

    correct = correct.sort_values(
        by="confidence",
        ascending=False,
    )

    selected_columns = [
        "relative_path",
        "true_label",
        "true_species",
        "predicted_label",
        "predicted_species",
        "confidence",
        "top5_labels",
        "top5_scores",
    ]

    return correct[selected_columns].head(number_of_examples)


def save_table(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save one analysis table as a CSV file."""
    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse prediction results, class accuracy, "
            "confused pairs and representative examples."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Prediction CSV file.",
    )

    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=DEFAULT_LABEL_MAPPING_PATH,
        help="JSON file mapping labels to species information.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV tables.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="resnet18_finetune",
        help="Prefix used for generated output files.",
    )

    parser.add_argument(
        "--top-classes",
        type=int,
        default=20,
        help="Number of worst-performing classes to save.",
    )

    parser.add_argument(
        "--top-pairs",
        type=int,
        default=20,
        help="Number of most confused class pairs to save.",
    )

    parser.add_argument(
        "--example-count",
        type=int,
        default=10,
        help="Number of correct and incorrect examples to save.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_mapping = load_label_mapping(
        args.label_mapping
    )

    predictions = prepare_predictions(
        predictions_path=args.predictions,
        label_mapping=label_mapping,
    )

    class_accuracy = calculate_class_accuracy(
        predictions
    )

    confused_pairs = calculate_confused_pairs(
        predictions
    )

    error_examples = select_error_examples(
        predictions,
        number_of_examples=args.example_count,
    )

    correct_examples = select_correct_examples(
        predictions,
        number_of_examples=args.example_count,
    )

    prefix = args.experiment_name

    save_table(
        class_accuracy,
        args.output_dir
        / f"{prefix}_class_accuracy.csv",
    )

    save_table(
        class_accuracy.head(args.top_classes),
        args.output_dir
        / f"{prefix}_worst_classes.csv",
    )

    save_table(
        confused_pairs.head(args.top_pairs),
        args.output_dir
        / f"{prefix}_top_confused_pairs.csv",
    )

    save_table(
        error_examples,
        args.output_dir
        / f"{prefix}_error_examples.csv",
    )

    save_table(
        correct_examples,
        args.output_dir
        / f"{prefix}_correct_examples.csv",
    )

    total_samples = len(predictions)
    correct_samples = int(
        predictions["is_correct"].sum()
    )
    incorrect_samples = total_samples - correct_samples

    print("\nPrediction analysis summary")
    print("---------------------------")
    print(f"Experiment: {prefix}")
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Incorrect samples: {incorrect_samples}")
    print(
        "Calculated accuracy: "
        f"{correct_samples / total_samples * 100:.2f}%"
    )

    print("\nMost confused pairs:")
    print(
        confused_pairs.head(10).to_string(
            index=False
        )
    )

    print("\nWorst-performing classes:")
    print(
        class_accuracy.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()