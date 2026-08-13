import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = PROJECT_ROOT / "results" / "evaluation"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


MODEL_FILES = {
    "HOG + SVM": "hog_svm_metrics.json",
    "Scratch ResNet-18": "resnet18_scratch_aligned_metrics.json",
    "Frozen ResNet-18": "resnet18_frozen_aligned_metrics.json",
    "Fine-tuned ResNet-18": "resnet18_finetune_aligned_metrics.json",
}


def load_metrics(file_path: Path) -> dict:
    """Load one model's evaluation metrics from a JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name, file_name in MODEL_FILES.items():
        metrics_path = EVALUATION_DIR / file_name
        metrics = load_metrics(metrics_path)

        rows.append(
            {
                "Model": model_name,
                "Top-1 Accuracy (%)": metrics["top1_accuracy"] * 100,
                "Top-5 Accuracy (%)": metrics["top5_accuracy"] * 100,
                "Macro Precision (%)": metrics["macro_precision"] * 100,
                "Macro Recall (%)": metrics["macro_recall"] * 100,
                "Macro F1 (%)": metrics["macro_f1"] * 100,
            }
        )

    comparison_table = pd.DataFrame(rows)

    output_path = TABLES_DIR / "model_comparison.csv"
    comparison_table.to_csv(output_path, index=False)

    print(comparison_table.round(2).to_string(index=False))
    print(f"\nComparison table saved to: {output_path}")


if __name__ == "__main__":
    main()