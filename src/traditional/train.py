import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.evaluation.metrics import calculate_metrics
from src.traditional.feature_extraction import (
    default_features_dir,
    feature_config_from,
    load_config,
)


def load_features(features_path: Path) -> dict:
    archive = np.load(features_path, allow_pickle=False)

    # liblinear only accepts float64, so cache the converted array on disk.
    float64_path = features_path.with_name(
        f"{features_path.stem}_float64.npy"
    )

    if not float64_path.exists():
        np.save(
            float64_path,
            archive["features"].astype(np.float64),
        )

    return {
        "features": np.load(float64_path, mmap_mode="r"),
        "labels": archive["labels"],
        "relative_paths": archive["relative_paths"],
    }


def resolve_worker_count(
    requested: int,
    features: np.ndarray,
    memory_fraction: float = 0.5,
) -> int:
    cpu_count = os.cpu_count() or 1

    workers = cpu_count if requested in (-1, None) else requested
    workers = max(1, min(workers, cpu_count))

    # liblinear stores 16 bytes per non-zero value, plus one terminator per sample.
    per_worker_bytes = (
        int(np.count_nonzero(features)) + features.shape[0]
    ) * 16

    total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
        "SC_PHYS_PAGES"
    )

    affordable = int(total_memory * memory_fraction) // per_worker_bytes

    return max(1, min(workers, affordable))


def build_pipeline(classifier_config: dict) -> Pipeline:
    steps = []

    if classifier_config["standardize"]:
        steps.append(("scaler", StandardScaler()))

    binary_svm = LinearSVC(
        C=classifier_config["C"],
        max_iter=classifier_config["max_iter"],
        tol=classifier_config["tol"],
        random_state=classifier_config["random_state"],
        dual=classifier_config["dual"],
    )

    steps.append(
        (
            "svm",
            OneVsRestClassifier(
                binary_svm,
                n_jobs=classifier_config["n_jobs"],
                verbose=classifier_config["verbose"],
            ),
        )
    )

    return Pipeline(steps)


def top_k_predictions(
    scores: np.ndarray,
    classes: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])

    top_indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]

    top_labels = classes[top_indices]
    top_scores = np.take_along_axis(scores, top_indices, axis=1)

    return top_labels, top_scores


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hog_svm_baseline.yaml"),
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
        "--experiment-name",
        type=str,
        default=None,
        help="Overrides experiment_name from the config file.",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=None,
        help="Overrides the SVM regularisation strength from the config.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Overrides the number of parallel one-vs-rest workers.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/models"),
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
    )

    args = parser.parse_args()

    config = load_config(args.config)
    classifier_config = dict(config["classifier"])

    if args.C is not None:
        classifier_config["C"] = args.C

    if args.n_jobs is not None:
        classifier_config["n_jobs"] = args.n_jobs

    experiment_name = (
        args.experiment_name or config["experiment_name"]
    )

    suffix = f"_{args.tag}" if args.tag else ""

    train_data = load_features(
        args.features_dir / f"train{suffix}.npz"
    )

    validation_data = load_features(
        args.features_dir / f"validation{suffix}.npz"
    )

    num_classes = (
        int(
            max(
                train_data["labels"].max(),
                validation_data["labels"].max(),
            )
        )
        + 1
    )

    requested_workers = classifier_config["n_jobs"]

    classifier_config["n_jobs"] = resolve_worker_count(
        requested_workers,
        train_data["features"],
    )

    if classifier_config["n_jobs"] != requested_workers:
        print(
            f"Using {classifier_config['n_jobs']} workers "
            f"(requested {requested_workers}) to stay within memory."
        )

    pipeline = build_pipeline(classifier_config)

    start_time = time.perf_counter()
    pipeline.fit(train_data["features"], train_data["labels"])
    training_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    scores = pipeline.decision_function(
        validation_data["features"]
    )
    validation_seconds = time.perf_counter() - start_time

    top_labels, _ = top_k_predictions(
        scores,
        pipeline.classes_,
    )

    metrics = calculate_metrics(
        true_labels=validation_data["labels"].tolist(),
        predicted_labels=top_labels[:, 0].tolist(),
        top5_predictions=top_labels.tolist(),
        num_classes=num_classes,
    )

    metrics.pop("confusion_matrix")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / f"{experiment_name}.joblib"
    dump(pipeline, model_path)

    summary = {
        "experiment_name": experiment_name,
        "num_classes": num_classes,
        "num_train_images": int(train_data["features"].shape[0]),
        "num_validation_images": int(
            validation_data["features"].shape[0]
        ),
        "feature_dimension": int(train_data["features"].shape[1]),
        "features": feature_config_from(config),
        "classifier": classifier_config,
        "training_seconds": round(training_seconds, 3),
        "validation_inference_seconds": round(
            validation_seconds, 3
        ),
        "validation_metrics": metrics,
    }

    summary_path = (
        args.tables_dir / f"{experiment_name}_validation.json"
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Experiment: {experiment_name}")
    print(f"C: {classifier_config['C']}")
    print(f"Training time: {training_seconds:.1f} s")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print(f"Model saved to: {model_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
