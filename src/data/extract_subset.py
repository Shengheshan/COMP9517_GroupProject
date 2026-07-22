import argparse
import tarfile
from pathlib import Path

import pandas as pd


def load_required_paths(csv_paths: list[Path]) -> set[str]:
    """Read the relative image paths required by one or more CSV files."""
    required_paths: set[str] = set()

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        data = pd.read_csv(csv_path)

        if "relative_path" not in data.columns:
            raise ValueError(
                f"{csv_path} does not contain a relative_path column."
            )

        for path in data["relative_path"]:
            # tar archives always use forward slashes
            normalized_path = str(path).replace("\\", "/").lstrip("./")
            required_paths.add(normalized_path)

    return required_paths


def normalize_archive_name(name: str) -> str:
    """Normalize a member path stored inside a tar archive."""
    return name.replace("\\", "/").lstrip("./")


def extract_selected_files(
    archive_path: Path,
    required_paths: set[str],
    output_dir: Path,
) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    found_paths: set[str] = set()

    print(f"Opening archive: {archive_path}")
    print(f"Required images: {len(required_paths)}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue

            member_name = normalize_archive_name(member.name)

            if member_name not in required_paths:
                continue

            # Prevent unsafe paths such as ../../file
            destination = (output_dir / member_name).resolve()
            output_root = output_dir.resolve()

            if output_root not in destination.parents:
                raise ValueError(
                    f"Unsafe archive path detected: {member.name}"
                )

            archive.extract(member, path=output_dir)
            found_paths.add(member_name)

            extracted_count = len(found_paths)

            if extracted_count % 1000 == 0:
                print(
                    f"Extracted {extracted_count}/"
                    f"{len(required_paths)} images"
                )

    missing_paths = required_paths - found_paths

    print(f"Successfully extracted: {len(found_paths)}")

    if missing_paths:
        example_paths = sorted(missing_paths)[:10]

        raise FileNotFoundError(
            f"{len(missing_paths)} required images were not found "
            f"inside {archive_path}.\n"
            "First missing paths:\n"
            + "\n".join(example_paths)
        )

    print("All required images were extracted successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the selected COMP9517 dataset subset "
            "directly from the original iNaturalist archives."
        )
    )

    parser.add_argument(
        "--train-archive",
        type=Path,
        required=True,
        help="Path to train_mini.tar.gz",
    )

    parser.add_argument(
        "--val-archive",
        type=Path,
        required=True,
        help="Path to val.tar.gz",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data_metadata"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the selected subset will be extracted.",
    )

    args = parser.parse_args()

    train_paths = load_required_paths(
        [
            args.metadata_dir / "train.csv",
            args.metadata_dir / "validation.csv",
        ]
    )

    test_paths = load_required_paths(
        [args.metadata_dir / "test.csv"]
    )

    print("\n=== Extracting train and validation images ===")
    extract_selected_files(
        archive_path=args.train_archive,
        required_paths=train_paths,
        output_dir=args.output_dir,
    )

    print("\n=== Extracting test images ===")
    extract_selected_files(
        archive_path=args.val_archive,
        required_paths=test_paths,
        output_dir=args.output_dir,
    )

    print("\nDataset subset extraction completed.")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()