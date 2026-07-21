import argparse
from pathlib import Path

import pandas as pd


EXPECTED_COUNTS = {
    "train": 40,
    "validation": 10,
    "test": 10,
}


def validate_csv(
    csv_path: Path,
    data_root: Path,
    split_name: str,
) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    required_columns = {
        "relative_path",
        "label",
        "category_id",
        "species_name",
        "split",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    class_counts = data.groupby("label").size()

    if len(class_counts) != 500:
        raise ValueError(
            f"{split_name} contains {len(class_counts)} classes, "
            "expected 500."
        )

    expected_count = EXPECTED_COUNTS[split_name]

    invalid_counts = class_counts[
        class_counts != expected_count
    ]

    if not invalid_counts.empty:
        raise ValueError(
            f"{split_name} has incorrect class counts:\n"
            f"{invalid_counts}"
        )

    missing_paths = []

    for relative_path in data["relative_path"]:
        full_path = data_root / relative_path

        if not full_path.exists():
            missing_paths.append(str(full_path))

            if len(missing_paths) >= 10:
                break

    if missing_paths:
        raise FileNotFoundError(
            "Some images were not found:\n"
            + "\n".join(missing_paths)
        )

    print(
        f"{split_name}: OK — "
        f"{len(data)} images, "
        f"{len(class_counts)} classes"
    )

    return data


def main():
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    train_data = validate_csv(
        args.metadata_dir / "train.csv",
        args.data_root,
        "train",
    )

    validation_data = validate_csv(
        args.metadata_dir / "validation.csv",
        args.data_root,
        "validation",
    )

    test_data = validate_csv(
        args.metadata_dir / "test.csv",
        args.data_root,
        "test",
    )

    train_paths = set(train_data["relative_path"])
    validation_paths = set(
        validation_data["relative_path"]
    )
    test_paths = set(test_data["relative_path"])

    if train_paths & validation_paths:
        raise ValueError(
            "Train and validation overlap."
        )

    if train_paths & test_paths:
        raise ValueError(
            "Train and test overlap."
        )

    if validation_paths & test_paths:
        raise ValueError(
            "Validation and test overlap."
        )

    print("No overlap between splits.")
    print("Dataset validation completed successfully.")


if __name__ == "__main__":
    main()