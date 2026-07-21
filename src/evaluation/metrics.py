from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_top5_accuracy(
    true_labels: Sequence[int],
    top5_predictions: Sequence[Sequence[int]],
) -> float:
    if len(true_labels) != len(top5_predictions):
        raise ValueError(
            "true_labels and top5_predictions must have the same length."
        )

    correct = sum(
        int(true_label in predicted_labels)
        for true_label, predicted_labels
        in zip(true_labels, top5_predictions)
    )

    return correct / len(true_labels)


def calculate_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    top5_predictions: Sequence[Sequence[int]],
    num_classes: int = 500,
) -> dict:
    true_labels_array = np.asarray(true_labels)
    predicted_labels_array = np.asarray(predicted_labels)

    if len(true_labels_array) == 0:
        raise ValueError("Prediction data is empty.")

    top1_accuracy = accuracy_score(
        true_labels_array,
        predicted_labels_array,
    )

    top5_accuracy = calculate_top5_accuracy(
        true_labels_array,
        top5_predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_array,
        predicted_labels_array,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels_array,
        predicted_labels_array,
        labels=list(range(num_classes)),
    )

    return {
        "top1_accuracy": float(top1_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": matrix,
    }