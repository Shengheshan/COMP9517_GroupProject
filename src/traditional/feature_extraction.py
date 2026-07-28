import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog

from src.data.dataset import load_dataframe


SPLIT_CSV_NAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def default_features_dir() -> Path:
    # Where extracted features are cached, set COMP9517_FEATURES_DIR to change it.
    return Path(
        os.environ.get(
            "COMP9517_FEATURES_DIR",
            "results/features",
        )
    )


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def feature_config_from(config: dict) -> dict:
    return {
        "hog": config.get("hog"),
        "color": config.get("color"),
        "l2_normalize": config.get("l2_normalize", False),
    }


def compute_hog_features(
    image: Image.Image,
    hog_config: dict,
) -> np.ndarray:
    grayscale = image.convert("L").resize(
        tuple(hog_config["image_size"]),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(grayscale, dtype=np.float32) / 255.0

    features = hog(
        array,
        orientations=hog_config["orientations"],
        pixels_per_cell=tuple(hog_config["pixels_per_cell"]),
        cells_per_block=tuple(hog_config["cells_per_block"]),
        block_norm=hog_config["block_norm"],
        transform_sqrt=hog_config["transform_sqrt"],
        feature_vector=True,
    )

    return features.astype(np.float32)


def compute_color_features(
    image: Image.Image,
    color_config: dict,
) -> np.ndarray:
    hsv_image = image.convert("HSV").resize(
        tuple(color_config["image_size"]),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(hsv_image)
    height, width = array.shape[:2]

    rows, columns = color_config["grid"]

    channel_bins = (
        color_config["hue_bins"],
        color_config["saturation_bins"],
        color_config["value_bins"],
    )

    histograms = []

    for row_index in range(rows):
        row_slice = slice(
            row_index * height // rows,
            (row_index + 1) * height // rows,
        )

        for column_index in range(columns):
            column_slice = slice(
                column_index * width // columns,
                (column_index + 1) * width // columns,
            )

            patch = array[row_slice, column_slice]

            for channel, bins in enumerate(channel_bins):
                histogram, _ = np.histogram(
                    patch[:, :, channel],
                    bins=bins,
                    range=(0, 256),
                )

                histogram = histogram.astype(np.float32)
                total = histogram.sum()

                if total > 0:
                    histogram /= total

                # Square root (Hellinger mapping) makes histograms easier for a linear SVM to compare.
                histograms.append(np.sqrt(histogram))

    return np.concatenate(histograms)


def unit_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0.0:
        return vector

    return vector / norm


def compute_descriptor(
    image_path: Path,
    feature_config: dict,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    parts = []

    hog_config = feature_config.get("hog")

    if hog_config is not None and hog_config.get("enabled", True):
        parts.append(compute_hog_features(image, hog_config))

    color_config = feature_config.get("color")

    if color_config is not None and color_config.get(
        "enabled", True
    ):
        parts.append(compute_color_features(image, color_config))

    if not parts:
        raise ValueError(
            "No feature block is enabled in the configuration."
        )

    if feature_config.get("l2_normalize", False):
        parts = [unit_normalize(part) for part in parts]

    return np.concatenate(parts).astype(np.float32)


def subset_dataframe(data, max_classes, max_images_per_class):
    if max_classes is not None:
        kept_labels = sorted(data["label"].unique())[:max_classes]
        data = data[data["label"].isin(kept_labels)]

    if max_images_per_class is not None:
        data = data.groupby("label", group_keys=False).head(
            max_images_per_class
        )

    return data.reset_index(drop=True)


def extract_split(
    csv_path: Path,
    data_root: Path,
    feature_config: dict,
    n_jobs: int,
    max_classes: int | None,
    max_images_per_class: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = load_dataframe(csv_path)

    data = subset_dataframe(
        data,
        max_classes,
        max_images_per_class,
    )

    image_paths = [
        data_root / relative_path
        for relative_path in data["relative_path"]
    ]

    start_time = time.perf_counter()

    features = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(compute_descriptor)(image_path, feature_config)
        for image_path in image_paths
    )

    elapsed_seconds = time.perf_counter() - start_time

    feature_matrix = np.vstack(features)
    labels = data["label"].to_numpy(dtype=np.int64)

    relative_paths = data["relative_path"].to_numpy(dtype=object)
    relative_paths = relative_paths.astype(str)

    return feature_matrix, labels, relative_paths, elapsed_seconds


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hog_svm_baseline.yaml"),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data_metadata"),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=sorted(SPLIT_CSV_NAMES),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_features_dir(),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Use only the first N labels. Intended for smoke tests.",
    )

    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Use only the first M images per label.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Suffix appended to output filenames.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    feature_config = feature_config_from(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""

    for split_name in args.splits:
        csv_path = args.metadata_dir / SPLIT_CSV_NAMES[split_name]

        (
            feature_matrix,
            labels,
            relative_paths,
            elapsed_seconds,
        ) = extract_split(
            csv_path=csv_path,
            data_root=args.data_root,
            feature_config=feature_config,
            n_jobs=args.n_jobs,
            max_classes=args.max_classes,
            max_images_per_class=args.max_images_per_class,
        )

        output_path = (
            args.output_dir / f"{split_name}{suffix}.npz"
        )

        np.savez(
            output_path,
            features=feature_matrix,
            labels=labels,
            relative_paths=relative_paths,
        )

        metadata = {
            "split": split_name,
            "num_images": int(feature_matrix.shape[0]),
            "num_classes": int(np.unique(labels).size),
            "feature_dimension": int(feature_matrix.shape[1]),
            "extraction_seconds": round(elapsed_seconds, 3),
            "seconds_per_image": round(
                elapsed_seconds / feature_matrix.shape[0],
                6,
            ),
            "features": feature_config,
        }

        metadata_path = (
            args.output_dir / f"{split_name}{suffix}_info.json"
        )

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print(
            f"{split_name}: {metadata['num_images']} images, "
            f"{metadata['num_classes']} classes, "
            f"dimension {metadata['feature_dimension']}, "
            f"{metadata['extraction_seconds']} s"
        )

        print(f"Features saved to: {output_path}")


if __name__ == "__main__":
    main()
