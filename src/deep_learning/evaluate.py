# Member 4: Transfer learning evaluation and prediction export
# This file evaluates the trained transfer-learning models on the final test set
# and saves predictions in the common project CSV format.

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from src.data.dataset import INatDataset
from src.deep_learning.model import create_resnet18


def create_evaluation_transform():
    weights = ResNet18_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@torch.no_grad()
def run_inference(model, loader, dataframe, device):
    model.eval()

    prediction_rows = []
    total_inference_time = 0.0
    current_index = 0

    for images, labels in tqdm(loader, desc="Testing"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_inference_time += time.perf_counter() - start_time

        top5_scores, top5_labels = torch.topk(
            probabilities,
            k=5,
            dim=1,
        )

        batch_size = labels.size(0)

        labels_cpu = labels.cpu().tolist()
        top5_labels_cpu = top5_labels.cpu().tolist()
        top5_scores_cpu = top5_scores.cpu().tolist()

        for batch_index in range(batch_size):
            row = dataframe.iloc[current_index]

            prediction_rows.append({
                "relative_path": row["relative_path"],
                "true_label": int(labels_cpu[batch_index]),
                "predicted_label": int(
                    top5_labels_cpu[batch_index][0]
                ),
                "top5_labels": json.dumps(
                    top5_labels_cpu[batch_index]
                ),
                "top5_scores": json.dumps(
                    top5_scores_cpu[batch_index]
                ),
            })

            current_index += 1

    return prediction_rows, total_inference_time


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["frozen", "finetune"],
        required=True,
    )

    parser.add_argument(
        "--data-root",
        default=".",
    )

    parser.add_argument(
        "--test-csv",
        default="data_metadata/test.csv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=500,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    experiment_name = f"resnet18_{args.mode}"

    checkpoint_path = (
        Path("models")
        / f"{experiment_name}_best.pth"
    )

    output_directory = Path("results/predictions")
    timing_directory = Path("results/evaluation")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timing_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{experiment_name}.csv"
    )

    timing_path = (
        timing_directory
        / f"{experiment_name}_timing.json"
    )

    test_dataframe = pd.read_csv(args.test_csv)

    test_dataset = INatDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        transform=create_evaluation_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = create_resnet18(
        num_classes=args.num_classes,
        mode=args.mode,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    print(f"Experiment: {experiment_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Test images: {len(test_dataset)}")
    print(
        "Best validation Macro-F1:",
        checkpoint["metrics"]["validation_macro_f1"],
    )

    prediction_rows, total_inference_time = run_inference(
        model=model,
        loader=test_loader,
        dataframe=test_dataframe,
        device=device,
    )

    pd.DataFrame(prediction_rows).to_csv(
        output_path,
        index=False,
    )

    number_of_images = len(test_dataset)

    timing_results = {
        "experiment": experiment_name,
        "device": str(device),
        "number_of_images": number_of_images,
        "total_inference_time_seconds": total_inference_time,
        "average_inference_time_seconds_per_image": (
            total_inference_time / number_of_images
        ),
        "images_per_second": (
            number_of_images / total_inference_time
        ),
        "batch_size": args.batch_size,
    }

    with open(
        timing_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            timing_results,
            file,
            indent=4,
        )

    print(f"Predictions saved to: {output_path}")
    print(
        f"Total inference time: "
        f"{total_inference_time:.4f} seconds"
    )
    print(
        f"Average inference time per image: "
        f"{total_inference_time / number_of_images:.6f} seconds"
    )
    print(
        f"Images per second: "
        f"{number_of_images / total_inference_time:.2f}"
    )
    print(f"Timing saved to: {timing_path}")


if __name__ == "__main__":
    main()