import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "cam_examples"
    / "cam_examples_manifest.csv"
)

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_finetune_aligned_best.pth"
)

DEFAULT_LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "data_metadata"
    / "label_to_category.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "gradcam"
)

REQUIRED_MANIFEST_COLUMNS = {
    "group",
    "filename",
    "true_label",
    "predicted_label",
}


class GradCAM:
    """Generate Grad-CAM maps for a selected convolutional layer."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs

        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(
        self,
        gradients: torch.Tensor,
    ) -> None:
        self.gradients = gradients

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Generate one Grad-CAM heatmap.

        Returns:
            heatmap:
                Normalised two-dimensional CAM array.
            predicted_class:
                Model Top-1 prediction.
            confidence:
                Top-1 softmax confidence.
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        predicted_class = int(
            probabilities.argmax(dim=1).item()
        )

        confidence = float(
            probabilities[0, predicted_class].item()
        )

        explained_class = (
            predicted_class
            if target_class is None
            else int(target_class)
        )

        logits[0, explained_class].backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * self.activations
        ).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam[0].detach().cpu().numpy()

        minimum = float(cam.min())
        maximum = float(cam.max())

        if maximum > minimum:
            cam = (
                (cam - minimum)
                / (maximum - minimum)
            )
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class, confidence

    def close(self) -> None:
        """Remove the registered hook."""
        self.forward_handle.remove()


def find_checkpoint(
    requested_path: Path,
) -> Path:
    """
    Find the checkpoint even if it was placed in a slightly different
    project directory.
    """
    if requested_path.exists():
        return requested_path

    possible_paths = [
        PROJECT_ROOT
        / "results"
        / "models"
        / requested_path.name,

        PROJECT_ROOT
        / requested_path.name,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    matches = list(
        PROJECT_ROOT.rglob(requested_path.name)
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Fine-tuned model checkpoint was not found.\n"
        f"Expected filename: {requested_path.name}\n"
        "Recommended location:\n"
        f"{PROJECT_ROOT / 'models' / requested_path.name}"
    )


def build_model(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
) -> nn.Module:
    """Create ResNet-18 and load the fine-tuned checkpoint."""
    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must be a dictionary."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def create_transforms() -> tuple[
    transforms.Compose,
    transforms.Compose,
]:
    """
    Reproduce the evaluation preprocessing used during testing.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1

    mean = weights.transforms().mean
    std = weights.transforms().std

    model_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    display_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    return model_transform, display_transform


def load_label_mapping(
    file_path: Path,
) -> dict[int, str]:
    """Load label IDs and extract species names."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Label mapping not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_mapping = json.load(file)

    mapping: dict[int, str] = {}

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

        mapping[int(label)] = str(
            species_name
        ).replace("_", " ")

    return mapping


def load_manifest(
    file_path: Path,
) -> pd.DataFrame:
    """Load and validate the CAM examples manifest."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {file_path}"
        )

    manifest = pd.read_csv(file_path)

    missing_columns = (
        REQUIRED_MANIFEST_COLUMNS
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    manifest["group"] = (
        manifest["group"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    manifest["true_label"] = (
        manifest["true_label"].astype(int)
    )

    manifest["predicted_label"] = (
        manifest["predicted_label"].astype(int)
    )

    return manifest


def resolve_image_path(
    manifest_path: Path,
    group: str,
    filename: str,
) -> Path:
    """Locate an image in correct/ or incorrect/."""
    image_path = (
        manifest_path.parent
        / group
        / filename
    )

    if image_path.exists():
        return image_path

    matches = list(
        manifest_path.parent.rglob(filename)
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Image not found: {filename}"
    )


def resize_heatmap(
    heatmap: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Resize CAM to the displayed image size."""
    image = Image.fromarray(
        np.uint8(heatmap * 255),
        mode="L",
    )

    image = image.resize(
        image_size,
        resample=Image.Resampling.BILINEAR,
    )

    return (
        np.asarray(image).astype(np.float32)
        / 255.0
    )


def colourise_heatmap(
    heatmap: np.ndarray,
) -> np.ndarray:
    """Convert a grayscale heatmap into RGB."""
    colour_map = plt.get_cmap("jet")
    coloured = colour_map(heatmap)[..., :3]

    return np.uint8(coloured * 255)


def create_overlay(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend the original image with the coloured heatmap."""
    blended = (
        (1.0 - alpha)
        * original.astype(np.float32)
        + alpha
        * heatmap.astype(np.float32)
    )

    return np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)


def save_panel(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
    group: str,
    true_name: str,
    recorded_prediction_name: str,
    actual_prediction_name: str,
    confidence: float,
) -> None:
    """Save a report-ready Original/Heatmap/Overlay panel."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")

    for axis in axes:
        axis.axis("off")

    status = (
        "Correct Prediction"
        if group == "correct"
        else "Misclassification"
    )

    title = (
        f"{status} | "
        f"Ground Truth: {true_name} | "
        f"Prediction: {actual_prediction_name} | "
        f"Confidence: {confidence * 100:.2f}%"
    )

    if (
        recorded_prediction_name
        != actual_prediction_name
    ):
        title += (
            " | Recorded Prediction: "
            f"{recorded_prediction_name}"
        )

    figure.suptitle(
        title,
        fontsize=11,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def process_example(
    row: pd.Series,
    manifest_path: Path,
    model: nn.Module,
    gradcam: GradCAM,
    model_transform: transforms.Compose,
    display_transform: transforms.Compose,
    label_mapping: dict[int, str],
    device: torch.device,
    output_dir: Path,
    alpha: float,
) -> dict[str, object]:
    """Generate Grad-CAM outputs for one manifest row."""
    group = str(row["group"])
    filename = str(row["filename"])

    true_label = int(row["true_label"])
    recorded_predicted_label = int(
        row["predicted_label"]
    )

    image_path = resolve_image_path(
        manifest_path=manifest_path,
        group=group,
        filename=filename,
    )

    original_pil = Image.open(
        image_path
    ).convert("RGB")

    display_pil = display_transform(
        original_pil
    )

    original_array = np.asarray(
        display_pil
    )

    input_tensor = model_transform(
        original_pil
    ).unsqueeze(0).to(device)

    # Explain the class recorded by the original evaluation.
    heatmap, actual_predicted_label, confidence = (
        gradcam.generate(
            input_tensor=input_tensor,
            target_class=recorded_predicted_label,
        )
    )

    resized_heatmap = resize_heatmap(
        heatmap=heatmap,
        image_size=display_pil.size,
    )

    colour_heatmap = colourise_heatmap(
        resized_heatmap
    )

    overlay = create_overlay(
        original=original_array,
        heatmap=colour_heatmap,
        alpha=alpha,
    )

    true_name = label_mapping.get(
        true_label,
        f"Class {true_label}",
    )

    recorded_prediction_name = label_mapping.get(
        recorded_predicted_label,
        f"Class {recorded_predicted_label}",
    )

    actual_prediction_name = label_mapping.get(
        actual_predicted_label,
        f"Class {actual_predicted_label}",
    )

    group_output_dir = output_dir / group
    group_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = Path(filename).stem

    panel_path = (
        group_output_dir
        / f"{stem}_gradcam_panel.png"
    )

    overlay_path = (
        group_output_dir
        / f"{stem}_gradcam_overlay.png"
    )

    save_panel(
        original=original_array,
        heatmap=colour_heatmap,
        overlay=overlay,
        output_path=panel_path,
        group=group,
        true_name=true_name,
        recorded_prediction_name=(
            recorded_prediction_name
        ),
        actual_prediction_name=(
            actual_prediction_name
        ),
        confidence=confidence,
    )

    Image.fromarray(
        overlay
    ).save(overlay_path)

    print(
        f"[{group.upper()}] {filename}\n"
        f"  Ground truth: {true_name}\n"
        f"  Prediction: {actual_prediction_name}\n"
        f"  Confidence: {confidence * 100:.2f}%\n"
        f"  Saved: {panel_path}"
    )

    return {
        "group": group,
        "filename": filename,
        "true_label": true_label,
        "true_species": true_name,
        "recorded_predicted_label": (
            recorded_predicted_label
        ),
        "recorded_predicted_species": (
            recorded_prediction_name
        ),
        "actual_predicted_label": (
            actual_predicted_label
        ),
        "actual_predicted_species": (
            actual_prediction_name
        ),
        "confidence": confidence,
        "panel_path": str(
            panel_path.relative_to(PROJECT_ROOT)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate Grad-CAM visualisations "
            "from cam_examples_manifest.csv."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=DEFAULT_LABEL_MAPPING_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-per-group",
        type=int,
        default=5,
        help=(
            "Maximum examples generated from each "
            "group. Use 0 to process every image."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha must be between 0 and 1."
        )

    checkpoint_path = find_checkpoint(
        args.checkpoint
    )

    manifest = load_manifest(
        args.manifest
    )

    label_mapping = load_label_mapping(
        args.label_mapping
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest: {args.manifest}")

    model = build_model(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes=args.num_classes,
    )

    model_transform, display_transform = (
        create_transforms()
    )

    gradcam = GradCAM(
        model=model,
        target_layer=model.layer4[-1],
    )

    selected_rows = []

    for group_name in ["correct", "incorrect"]:
        group_rows = manifest.loc[
            manifest["group"] == group_name
        ]

        if args.max_per_group > 0:
            group_rows = group_rows.head(
                args.max_per_group
            )

        selected_rows.append(group_rows)

    selected_manifest = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    results = []

    try:
        for _, row in selected_manifest.iterrows():
            result = process_example(
                row=row,
                manifest_path=args.manifest,
                model=model,
                gradcam=gradcam,
                model_transform=model_transform,
                display_transform=display_transform,
                label_mapping=label_mapping,
                device=device,
                output_dir=args.output_dir,
                alpha=args.alpha,
            )

            results.append(result)

    finally:
        gradcam.close()

    summary = pd.DataFrame(results)

    summary_path = (
        args.output_dir
        / "gradcam_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("Grad-CAM generation completed.")
    print(f"Processed images: {len(results)}")
    print(f"Summary saved to: {summary_path}")
    print(f"Figures saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()