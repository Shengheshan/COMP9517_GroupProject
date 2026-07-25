import argparse
import json
import time
from pathlib import Path

import pandas as pd
from joblib import load

from src.traditional.feature_extraction import default_features_dir
from src.traditional.train import load_features, top_k_predictions


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=default_features_dir(),
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Suffix of the feature files produced by feature_extraction.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictions/hog_svm.csv"),
    )

    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""

    data = load_features(
        args.features_dir / f"{args.split}{suffix}.npz"
    )

    pipeline = load(args.model)

    start_time = time.perf_counter()
    scores = pipeline.decision_function(data["features"])
    inference_seconds = time.perf_counter() - start_time

    top_labels, top_scores = top_k_predictions(
        scores,
        pipeline.classes_,
    )

    predictions = pd.DataFrame(
        {
            "relative_path": data["relative_paths"],
            "true_label": data["labels"],
            "predicted_label": top_labels[:, 0],
            "top5_labels": [
                str(labels.tolist()) for labels in top_labels
            ],
            "top5_scores": [
                str([round(float(score), 6) for score in row])
                for row in top_scores
            ],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    num_images = len(predictions)

    timing = {
        "model": str(args.model),
        "split": args.split,
        "num_images": num_images,
        "classification_seconds": round(inference_seconds, 3),
        "classification_seconds_per_image": round(
            inference_seconds / num_images,
            6,
        ),
    }

    timing_path = args.output.with_name(
        f"{args.output.stem}_timing.json"
    )

    with timing_path.open("w", encoding="utf-8") as file:
        json.dump(timing, file, indent=2)

    print(f"Predictions saved to: {args.output}")
    print(f"Timing saved to: {timing_path}")


if __name__ == "__main__":
    main()
