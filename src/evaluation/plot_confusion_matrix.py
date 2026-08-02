import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATRIX_PATH = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "resnet18_finetune_aligned_confusion_matrix.npy"
)

LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "data_metadata"
    / "label_to_category.json"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_species_names(file_path: Path) -> list[str]:
    """Load species names in label order."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Label mapping file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    species_names = []

    for label in range(len(raw_mapping)):
        category_info = raw_mapping[str(label)]

        if isinstance(category_info, dict):
            species_name = (
                category_info.get("species name")
                or category_info.get("species_name")
                or category_info.get("name")
                or f"Class {label}"
            )
        else:
            species_name = str(category_info)

        species_names.append(
            str(species_name).replace("_", " ").strip()
        )

    return species_names


def normalise_confusion_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """Normalise each row so values represent class recall."""
    row_sums = matrix.sum(axis=1, keepdims=True)

    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )


def plot_full_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the complete 500-class normalised confusion matrix."""
    normalised = normalise_confusion_matrix(matrix)

    figure, axis = plt.subplots(figsize=(12, 10))

    image = axis.imshow(
        normalised,
        interpolation="nearest",
        aspect="auto",
    )

    axis.set_title(
        "Normalised Confusion Matrix "
        "(Fine-tuned ResNet-18)"
    )
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")

    figure.colorbar(
        image,
        ax=axis,
        label="Proportion",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def select_most_confused_classes(
    matrix: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """
    Select classes with the largest numbers of classification errors.
    """
    class_errors = (
        matrix.sum(axis=1)
        - np.diag(matrix)
    )

    return np.argsort(class_errors)[-top_n:]


def plot_top_classes_confusion_matrix(
    matrix: np.ndarray,
    species_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Plot a readable matrix for the most error-prone classes.
    """
    selected_indices = select_most_confused_classes(
        matrix,
        top_n,
    )

    selected_matrix = matrix[
        np.ix_(selected_indices, selected_indices)
    ]

    normalised = normalise_confusion_matrix(
        selected_matrix
    )

    selected_names = [
        species_names[index]
        for index in selected_indices
    ]

    figure, axis = plt.subplots(figsize=(14, 12))

    image = axis.imshow(
        normalised,
        interpolation="nearest",
        aspect="auto",
    )

    axis.set_title(
        f"Confusion Matrix for {top_n} "
        "Most Error-prone Species"
    )
    axis.set_xlabel("Predicted Species")
    axis.set_ylabel("True Species")

    axis.set_xticks(
        np.arange(top_n)
    )
    axis.set_yticks(
        np.arange(top_n)
    )

    axis.set_xticklabels(
        selected_names,
        rotation=90,
        fontsize=8,
    )
    axis.set_yticklabels(
        selected_names,
        fontsize=8,
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Proportion",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Confusion matrix not found: {MATRIX_PATH}"
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_matrix = np.load(MATRIX_PATH)
    species_names = load_species_names(
        LABEL_MAPPING_PATH
    )

    if confusion_matrix.ndim != 2:
        raise ValueError(
            "Confusion matrix must be a two-dimensional array."
        )

    if confusion_matrix.shape[0] != len(species_names):
        raise ValueError(
            "The number of species names does not match "
            "the confusion-matrix dimensions."
        )

    plot_full_confusion_matrix(
        matrix=confusion_matrix,
        output_path=(
            FIGURES_DIR
            / "resnet18_finetune_confusion_matrix_full.png"
        ),
    )

    plot_top_classes_confusion_matrix(
        matrix=confusion_matrix,
        species_names=species_names,
        output_path=(
            FIGURES_DIR
            / "resnet18_finetune_confusion_matrix_top20.png"
        ),
        top_n=20,
    )


if __name__ == "__main__":
    main()