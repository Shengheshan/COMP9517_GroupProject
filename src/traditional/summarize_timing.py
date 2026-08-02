import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from src.traditional.feature_extraction import default_features_dir


DEVICE = "cpu"

SELECTION_METRIC = "validation_macro_f1"

COUNT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def extraction_info_path(
    features_dir: Path,
    split: str,
    tag: str | None,
) -> Path:
    suffix = f"_{tag}" if tag else ""

    return features_dir / f"{split}{suffix}_info.json"


def load_extraction_info(
    features_dir: Path,
    split: str,
    tag: str | None,
) -> dict:
    if not features_dir.is_dir():
        raise SystemExit(
            f"Feature cache directory not found: {features_dir}\n"
            "Mount the dataset drive, or set COMP9517_FEATURES_DIR "
            "(or pass --features-dir) to the directory holding the "
            "*_info.json files written by "
            "src/traditional/feature_extraction.py."
        )

    info_path = extraction_info_path(features_dir, split, tag)

    if not info_path.is_file():
        raise SystemExit(
            f"Extraction metadata not found: {info_path}\n"
            "Re-run src/traditional/feature_extraction.py "
            f"--splits {split} --tag {tag}, or set "
            "COMP9517_FEATURES_DIR (or pass --features-dir) to the "
            "directory that holds it."
        )

    return load_json(info_path)


def cpu_brand() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            result = None

        if result is not None and result.stdout.strip():
            return result.stdout.strip()

    return platform.processor() or platform.machine() or "unknown CPU"


def total_memory_gb() -> int | None:
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )
    except (AttributeError, ValueError, OSError):
        return None

    return round(total_bytes / 1024**3)


def describe_hardware(cores: int) -> str:
    parts = [cpu_brand(), f"{cores} cores"]

    memory_gb = total_memory_gb()

    if memory_gb is not None:
        parts.append(f"{memory_gb} GB RAM")

    return ", ".join(parts)


def feature_config_from_metadata(metadata: dict) -> dict | None:
    if isinstance(metadata.get("features"), dict):
        return metadata["features"]

    # Older extraction metadata stored the blocks at the top level.
    if "hog" in metadata or "color" in metadata:
        return {
            "hog": metadata.get("hog"),
            "color": metadata.get("color"),
            "l2_normalize": metadata.get("l2_normalize", False),
        }

    return None


def resolve_feature_config(
    table: dict,
    train_info: dict,
    table_path: Path,
) -> dict:
    for metadata in (table, train_info):
        feature_config = feature_config_from_metadata(metadata)

        if feature_config is not None:
            return feature_config

    raise SystemExit(
        f"No feature description found in {table_path} or in the "
        "train extraction metadata, so the method label cannot be "
        "rebuilt. Re-run src/traditional/train.py to refresh the "
        "validation summary."
    )


def resolve_workers(requested: int | None, cores: int) -> int:
    if requested in (-1, None):
        return cores

    return requested


def method_label(feature_config: dict) -> str:
    parts = []

    hog_config = feature_config.get("hog")

    if hog_config is not None and hog_config.get("enabled", True):
        parts.append("HOG")

    color_config = feature_config.get("color")

    if color_config is not None and color_config.get(
        "enabled", True
    ):
        parts.append("spatial HSV colour histogram")

    parts.append("linear SVM (one-vs-rest)")

    return " + ".join(parts)


def pipeline_stages(feature_config: dict) -> list[str]:
    stages = ["JPEG decode"]

    hog_config = feature_config.get("hog") or {}
    color_config = feature_config.get("color") or {}

    image_size = hog_config.get("image_size") or color_config.get(
        "image_size"
    )

    if image_size is not None:
        stages.append(
            f"resize to {image_size[0]}x{image_size[1]}"
        )

    if hog_config and hog_config.get("enabled", True):
        stages.append("HOG descriptor")

    if color_config and color_config.get("enabled", True):
        stages.append("spatial HSV histogram")

    if feature_config.get("l2_normalize", False):
        stages.append("L2 normalisation")

    return stages


def sweep_tables(table_path: Path) -> list[Path]:
    # Sweep members share everything before the _C<value> part of the filename.
    prefix = table_path.stem.split("_C")[0]

    return sorted(
        table_path.parent.glob(f"{prefix}_C*_validation.json")
    )


def sweep_fit_seconds(tables: list[Path]) -> float:
    return sum(
        load_json(table_path)["training_seconds"]
        for table_path in tables
    )


def build_notes(
    feature_config: dict,
    validation_info: dict | None,
    workers: int,
    cores: int,
    tables: list[Path],
) -> list[str]:
    stages = pipeline_stages(feature_config)

    notes = [
        "Inference time covers the full pipeline: "
        f"{', '.join(stages)} and the linear SVM decision function.",
        "Feature extraction and one-vs-rest SVM fitting both ran "
        f"with {workers} joblib workers, so all timings are wall "
        f"clock on a fully loaded {cores}-core CPU rather than "
        "single-core time.",
    ]

    if validation_info is not None:
        notes.append(
            "Validation feature extraction "
            f"({validation_info['num_images']} images, "
            f"{validation_info['extraction_seconds']} s) supported "
            "the C sweep and is excluded from training time, which "
            "reports the cost of producing the final model only."
        )

    if len(tables) > 1:
        count = COUNT_WORDS.get(len(tables), str(len(tables)))

        notes.append(
            f"The {count}-value C sweep for this configuration cost "
            f"roughly {round(sweep_fit_seconds(tables))} s of "
            "additional SVM fitting."
        )

    return notes


def build_summary(
    experiment_name: str,
    config_path: Path,
    features_dir: Path,
    tag: str | None,
    table_path: Path,
    classification_timing_path: Path,
) -> dict:
    train_info = load_extraction_info(features_dir, "train", tag)
    test_info = load_extraction_info(features_dir, "test", tag)

    validation_path = extraction_info_path(
        features_dir,
        "validation",
        tag,
    )

    validation_info = (
        load_json(validation_path)
        if validation_path.is_file()
        else None
    )

    table = load_json(table_path)
    classification = load_json(classification_timing_path)

    if classification["num_images"] != test_info["num_images"]:
        raise SystemExit(
            "Inference sources disagree on the image count: "
            f"{classification_timing_path} reports "
            f"{classification['num_images']} but "
            f"{extraction_info_path(features_dir, 'test', tag)} "
            f"reports {test_info['num_images']}. "
            "Re-run both steps on the same split."
        )

    feature_dimension = table.get(
        "feature_dimension",
        test_info["feature_dimension"],
    )

    if feature_dimension != test_info["feature_dimension"]:
        raise SystemExit(
            "Feature sources disagree on the dimension: "
            f"{table_path} reports {feature_dimension} but "
            f"{extraction_info_path(features_dir, 'test', tag)} "
            f"reports {test_info['feature_dimension']}. "
            "Check that --tag matches the table."
        )

    cores = os.cpu_count() or 1

    feature_config = resolve_feature_config(
        table,
        train_info,
        table_path,
    )

    workers = resolve_workers(
        table["classifier"]["n_jobs"],
        cores,
    )

    training_seconds = (
        train_info["extraction_seconds"] + table["training_seconds"]
    )

    inference_seconds = (
        test_info["extraction_seconds"]
        + classification["classification_seconds"]
    )

    num_test_images = test_info["num_images"]

    return {
        "experiment": experiment_name,
        "method": method_label(feature_config),
        "configuration": (
            f"{config_path} with C={table['classifier']['C']}"
        ),
        "selection_metric": SELECTION_METRIC,
        "device": DEVICE,
        "hardware": describe_hardware(cores),
        "parallel_workers": workers,
        "feature_dimension": feature_dimension,
        "training": {
            "number_of_train_images": train_info["num_images"],
            "feature_extraction_seconds": train_info[
                "extraction_seconds"
            ],
            "svm_fit_seconds": table["training_seconds"],
            "total_training_seconds": round(training_seconds, 3),
            "total_training_minutes": round(
                training_seconds / 60,
                3,
            ),
        },
        "inference": {
            "number_of_test_images": num_test_images,
            "feature_extraction_seconds": test_info[
                "extraction_seconds"
            ],
            "classification_seconds": classification[
                "classification_seconds"
            ],
            "total_inference_seconds": round(inference_seconds, 3),
            "seconds_per_image": round(
                inference_seconds / num_test_images,
                6,
            ),
            "milliseconds_per_image": round(
                1000 * inference_seconds / num_test_images,
                3,
            ),
            "images_per_second": round(
                num_test_images / inference_seconds,
                2,
            ),
        },
        "notes": build_notes(
            feature_config=feature_config,
            validation_info=validation_info,
            workers=workers,
            cores=cores,
            tables=sweep_tables(table_path),
        ),
        "sources": {
            "feature_extraction_seconds": (
                f"{{train,test}}{f'_{tag}' if tag else ''}_info.json "
                "in the feature cache directory named by "
                "COMP9517_FEATURES_DIR (git ignored; absolute path "
                "omitted because it is machine specific)"
            ),
            "svm_fit_seconds": str(table_path),
            "classification_seconds": str(
                classification_timing_path
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=default_features_dir(),
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="color",
        help="Suffix of the feature files produced by feature_extraction.",
    )

    parser.add_argument(
        "--table",
        type=Path,
        default=Path("results/tables/color_C0.3_validation.json"),
        help="Validation summary of the delivered configuration.",
    )

    parser.add_argument(
        "--classification-timing",
        type=Path,
        default=Path("results/predictions/hog_svm_timing.json"),
        help="Timing JSON written alongside the test predictions.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hog_color.yaml"),
        help="Config file named in the configuration field.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="hog_svm",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation/hog_svm_timing.json"),
    )

    args = parser.parse_args()

    summary = build_summary(
        experiment_name=args.experiment_name,
        config_path=args.config,
        features_dir=args.features_dir,
        tag=args.tag,
        table_path=args.table,
        classification_timing_path=args.classification_timing,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    training = summary["training"]
    inference = summary["inference"]

    print(f"Experiment: {summary['experiment']}")
    print(f"Hardware: {summary['hardware']}")

    print(
        f"Training: {training['total_training_seconds']} s "
        f"({training['total_training_minutes']} min)"
    )

    print(
        f"Inference: {inference['total_inference_seconds']} s for "
        f"{inference['number_of_test_images']} images, "
        f"{inference['images_per_second']} images/second"
    )

    print(f"Timing summary saved to: {args.output}")


if __name__ == "__main__":
    main()
