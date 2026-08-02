from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from src.data.dataset import INatDataset
from src.models.resnet18_scratch import build_resnet18_scratch


NUM_CLASSES = 500

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate test-set predictions using the best "
            "aligned Scratch ResNet-18 checkpoint."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing train_mini and val.",
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data_metadata/test.csv"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/"
            "resnet18_scratch_aligned_best.pth"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/predictions/"
            "resnet18_scratch_aligned.csv"
        ),
    )

    parser.add_argument(
        "--timing-output",
        type=Path,
        default=Path(
            "results/evaluation/"
            "resnet18_scratch_aligned_inference_time.json"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.test_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {args.test_csv}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("Aligned Scratch ResNet-18 Test Prediction")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Data root: {args.data_root}")
    print("Evaluation transform:")
    print("  Resize(256)")
    print("  CenterCrop(224)")
    print("  ToTensor()")
    print("  ImageNet normalization")
    print("=" * 70)

    evaluation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    test_dataset = INatDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        transform=evaluation_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    test_dataframe = pd.read_csv(
        args.test_csv
    )

    required_columns = {
        "relative_path",
        "label",
    }

    missing_columns = (
        required_columns
        - set(test_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Test CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(test_dataframe) != len(test_dataset):
        raise ValueError(
            "Test CSV length does not match "
            "the test Dataset length."
        )

    relative_paths = (
        test_dataframe["relative_path"]
        .astype(str)
        .tolist()
    )

    expected_labels = (
        test_dataframe["label"]
        .astype(int)
        .tolist()
    )

    model = build_resnet18_scratch(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint.get("aligned_protocol")
        is not True
    ):
        print(
            "Warning: checkpoint is not marked "
            "as an aligned-protocol checkpoint."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    checkpoint_macro_f1 = checkpoint.get(
        "validation_macro_f1",
        "unknown",
    )

    checkpoint_accuracy = checkpoint.get(
        "validation_accuracy",
        "unknown",
    )

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint_epoch}"
    )
    print(
        f"Checkpoint validation Macro-F1: "
        f"{checkpoint_macro_f1}"
    )
    print(
        f"Checkpoint validation accuracy: "
        f"{checkpoint_accuracy}"
    )
    print(f"Test images: {len(test_dataset)}")
    print("=" * 70)

    prediction_rows: list[
        dict[str, object]
    ] = []

    sample_offset = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    with torch.no_grad():
        progress_bar = tqdm(
            test_loader,
            desc="Aligned test inference",
            unit="batch",
        )

        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            top5_scores, top5_labels = torch.topk(
                probabilities,
                k=5,
                dim=1,
            )

            predicted_labels = (
                top5_labels[:, 0]
            )

            batch_size = labels.size(0)

            batch_paths = relative_paths[
                sample_offset:
                sample_offset + batch_size
            ]

            batch_expected_labels = expected_labels[
                sample_offset:
                sample_offset + batch_size
            ]

            labels_cpu = (
                labels.cpu().tolist()
            )

            predicted_cpu = (
                predicted_labels
                .cpu()
                .tolist()
            )

            top5_labels_cpu = (
                top5_labels
                .cpu()
                .tolist()
            )

            top5_scores_cpu = (
                top5_scores
                .cpu()
                .tolist()
            )

            if (
                labels_cpu
                != batch_expected_labels
            ):
                raise ValueError(
                    "Dataset labels do not match "
                    "test.csv order. Paths and labels "
                    "may be misaligned."
                )

            for index in range(batch_size):
                top5_label_list = [
                    int(value)
                    for value
                    in top5_labels_cpu[index]
                ]

                top5_score_list = [
                    round(float(value), 8)
                    for value
                    in top5_scores_cpu[index]
                ]

                predicted_label = int(
                    predicted_cpu[index]
                )

                if (
                    predicted_label
                    != top5_label_list[0]
                ):
                    raise ValueError(
                        "Top-1 prediction does not "
                        "match the first Top-5 label."
                    )

                prediction_rows.append({
                    "relative_path": (
                        batch_paths[index]
                    ),
                    "true_label": int(
                        labels_cpu[index]
                    ),
                    "predicted_label": (
                        predicted_label
                    ),
                    "top5_labels": json.dumps(
                        top5_label_list
                    ),
                    "top5_scores": json.dumps(
                        top5_score_list
                    ),
                })

            sample_offset += batch_size

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    if sample_offset != len(test_dataset):
        raise ValueError(
            "Not all test images were processed."
        )

    if (
        len(prediction_rows)
        != len(test_dataframe)
    ):
        raise ValueError(
            "Prediction row count does not match "
            "the test CSV row count."
        )

    prediction_dataframe = pd.DataFrame(
        prediction_rows,
        columns=[
            "relative_path",
            "true_label",
            "predicted_label",
            "top5_labels",
            "top5_scores",
        ],
    )

    if not prediction_dataframe[
        "relative_path"
    ].is_unique:
        raise ValueError(
            "Duplicate relative_path values detected."
        )

    if not prediction_dataframe[
        "true_label"
    ].between(
        0,
        NUM_CLASSES - 1,
    ).all():
        raise ValueError(
            "Invalid true labels detected."
        )

    if not prediction_dataframe[
        "predicted_label"
    ].between(
        0,
        NUM_CLASSES - 1,
    ).all():
        raise ValueError(
            "Invalid predicted labels detected."
        )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dataframe.to_csv(
        args.output_csv,
        index=False,
    )

    average_seconds_per_image = (
        inference_seconds
        / len(test_dataset)
    )

    images_per_second = (
        len(test_dataset)
        / inference_seconds
    )

    timing_result = {
        "experiment": (
            "resnet18_scratch_aligned"
        ),
        "device": str(device),
        "checkpoint_epoch": (
            checkpoint_epoch
        ),
        "checkpoint_validation_macro_f1": (
            checkpoint_macro_f1
        ),
        "checkpoint_validation_accuracy": (
            checkpoint_accuracy
        ),
        "selection_metric": (
            checkpoint.get(
                "selection_metric",
                "unknown",
            )
        ),
        "aligned_protocol": (
            checkpoint.get(
                "aligned_protocol",
                False,
            )
        ),
        "imagenet_normalization": True,
        "number_of_test_images": len(
            test_dataset
        ),
        "total_inference_seconds": (
            inference_seconds
        ),
        "average_seconds_per_image": (
            average_seconds_per_image
        ),
        "images_per_second": (
            images_per_second
        ),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    args.timing_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.timing_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            timing_result,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("Aligned prediction completed")
    print(
        f"Rows saved: "
        f"{len(prediction_dataframe)}"
    )
    print(
        f"Prediction CSV: "
        f"{args.output_csv}"
    )
    print(
        f"Total inference time: "
        f"{inference_seconds:.2f} seconds"
    )
    print(
        f"Average time per image: "
        f"{average_seconds_per_image:.6f} seconds"
    )
    print(
        f"Throughput: "
        f"{images_per_second:.2f} images/second"
    )
    print(
        f"Timing JSON: "
        f"{args.timing_output}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()