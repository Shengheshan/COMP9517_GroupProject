import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


NUM_CLASSES = 500
TRAIN_PER_CLASS = 40
VALIDATION_PER_CLASS = 10
TEST_PER_CLASS = 10
DEFAULT_SEED = 42


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_category_image_map(
    annotation_data: dict[str, Any],
) -> dict[int, list[str]]:
    """
    Convert iNaturalist COCO-style annotations into:

        category_id -> list of image relative paths
    """

    image_id_to_path = {
        int(image["id"]): image["file_name"]
        for image in annotation_data["images"]
    }

    category_to_images: dict[int, list[str]] = defaultdict(list)

    for annotation in annotation_data["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])

        if image_id not in image_id_to_path:
            raise ValueError(
                f"Image ID {image_id} appears in annotations "
                "but not in images."
            )

        category_to_images[category_id].append(
            image_id_to_path[image_id]
        )

    return dict(category_to_images)


def build_category_metadata(
    annotation_data: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """
    Read species/category information from the categories section.
    """

    category_metadata: dict[int, dict[str, Any]] = {}

    for category in annotation_data.get("categories", []):
        category_id = int(category["id"])
        category_metadata[category_id] = category

    return category_metadata


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "relative_path",
        "label",
        "category_id",
        "species_name",
        "split",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_split_counts(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    expected = {
        "train": TRAIN_PER_CLASS,
        "validation": VALIDATION_PER_CLASS,
        "test": TEST_PER_CLASS,
    }

    split_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }

    for split_name, rows in split_rows.items():
        counts: dict[int, int] = defaultdict(int)

        for row in rows:
            counts[int(row["label"])] += 1

        if len(counts) != NUM_CLASSES:
            raise ValueError(
                f"{split_name} contains {len(counts)} classes, "
                f"expected {NUM_CLASSES}."
            )

        for label in range(NUM_CLASSES):
            actual_count = counts[label]
            expected_count = expected[split_name]

            if actual_count != expected_count:
                raise ValueError(
                    f"Label {label} in {split_name} has "
                    f"{actual_count} images, expected {expected_count}."
                )


def check_no_path_overlap(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    train_paths = {
        row["relative_path"]
        for row in train_rows
    }
    validation_paths = {
        row["relative_path"]
        for row in validation_rows
    }
    test_paths = {
        row["relative_path"]
        for row in test_rows
    }

    if train_paths & validation_paths:
        raise ValueError(
            "Data leakage: train and validation paths overlap."
        )

    if train_paths & test_paths:
        raise ValueError(
            "Data leakage: train and test paths overlap."
        )

    if validation_paths & test_paths:
        raise ValueError(
            "Data leakage: validation and test paths overlap."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible iNaturalist-2021 "
            "train/validation/test splits."
        )
    )

    parser.add_argument(
        "--train-json",
        type=Path,
        required=True,
        help="Path to train_mini.json",
    )

    parser.add_argument(
        "--test-json",
        type=Path,
        required=True,
        help=(
            "Path to official val.json, which is used "
            "as the project test set."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_metadata"),
        help="Directory for generated CSV and JSON files.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for class selection and splitting.",
    )

    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading annotation files...")
    train_data = load_json(args.train_json)
    test_data = load_json(args.test_json)

    print("Building category-to-image mappings...")
    train_category_images = build_category_image_map(train_data)
    test_category_images = build_category_image_map(test_data)

    category_metadata = build_category_metadata(train_data)

    eligible_categories = sorted(
        category_id
        for category_id in train_category_images
        if (
            len(train_category_images[category_id])
            >= TRAIN_PER_CLASS + VALIDATION_PER_CLASS
            and len(test_category_images.get(category_id, []))
            >= TEST_PER_CLASS
        )
    )

    if len(eligible_categories) < NUM_CLASSES:
        raise ValueError(
            f"Only {len(eligible_categories)} eligible categories found; "
            f"at least {NUM_CLASSES} are required."
        )

    rng = random.Random(args.seed)

    selected_category_ids = sorted(
        rng.sample(eligible_categories, NUM_CLASSES)
    )

    category_to_label = {
        category_id: label
        for label, category_id in enumerate(selected_category_ids)
    }

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    label_to_category: dict[str, dict[str, Any]] = {}

    for category_id in selected_category_ids:
        label = category_to_label[category_id]

        metadata = category_metadata.get(category_id, {})
        species_name = metadata.get(
            "name",
            f"category_{category_id}",
        )

        category_train_images = list(
            train_category_images[category_id]
        )
        category_test_images = list(
            test_category_images[category_id]
        )

        # Stable but category-specific shuffling.
        category_rng = random.Random(args.seed + category_id)
        category_rng.shuffle(category_train_images)
        category_rng.shuffle(category_test_images)

        selected_train_images = category_train_images[
            :TRAIN_PER_CLASS
        ]

        selected_validation_images = category_train_images[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS + VALIDATION_PER_CLASS
        ]

        selected_test_images = category_test_images[
            :TEST_PER_CLASS
        ]

        for relative_path in selected_train_images:
            train_rows.append(
                {
                    "relative_path": relative_path,
                    "label": label,
                    "category_id": category_id,
                    "species_name": species_name,
                    "split": "train",
                }
            )

        for relative_path in selected_validation_images:
            validation_rows.append(
                {
                    "relative_path": relative_path,
                    "label": label,
                    "category_id": category_id,
                    "species_name": species_name,
                    "split": "validation",
                }
            )

        for relative_path in selected_test_images:
            test_rows.append(
                {
                    "relative_path": relative_path,
                    "label": label,
                    "category_id": category_id,
                    "species_name": species_name,
                    "split": "test",
                }
            )

        label_to_category[str(label)] = {
            "category_id": category_id,
            "species_name": species_name,
            "kingdom": metadata.get("kingdom"),
            "phylum": metadata.get("phylum"),
            "class": metadata.get("class"),
            "order": metadata.get("order"),
            "family": metadata.get("family"),
            "genus": metadata.get("genus"),
        }

        summary_rows.append(
            {
                "label": label,
                "category_id": category_id,
                "species_name": species_name,
                "train_count": len(selected_train_images),
                "validation_count": len(
                    selected_validation_images
                ),
                "test_count": len(selected_test_images),
            }
        )

    validate_split_counts(
        train_rows,
        validation_rows,
        test_rows,
    )

    check_no_path_overlap(
        train_rows,
        validation_rows,
        test_rows,
    )

    write_csv(output_dir / "train.csv", train_rows)
    write_csv(
        output_dir / "validation.csv",
        validation_rows,
    )
    write_csv(output_dir / "test.csv", test_rows)

    with (
        output_dir / "selected_classes.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "random_seed": args.seed,
                "num_classes": NUM_CLASSES,
                "original_category_ids": selected_category_ids,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (
        output_dir / "category_to_label.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                str(category_id): label
                for category_id, label
                in category_to_label.items()
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (
        output_dir / "label_to_category.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            label_to_category,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (
        output_dir / "dataset_info.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_name": "iNaturalist-2021",
                "split_version": "1.0",
                "random_seed": args.seed,
                "num_classes": NUM_CLASSES,
                "train_images_per_class": TRAIN_PER_CLASS,
                "validation_images_per_class": (
                    VALIDATION_PER_CLASS
                ),
                "test_images_per_class": TEST_PER_CLASS,
                "total_train_images": len(train_rows),
                "total_validation_images": len(
                    validation_rows
                ),
                "total_test_images": len(test_rows),
                "train_source": "official train_mini",
                "validation_source": (
                    "held out from official train_mini"
                ),
                "test_source": (
                    "official val split, used as project test set"
                ),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary_path = output_dir / "split_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label",
                "category_id",
                "species_name",
                "train_count",
                "validation_count",
                "test_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Split generation completed.")
    print(f"Selected classes: {NUM_CLASSES}")
    print(f"Training images: {len(train_rows)}")
    print(f"Validation images: {len(validation_rows)}")
    print(f"Test images: {len(test_rows)}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()