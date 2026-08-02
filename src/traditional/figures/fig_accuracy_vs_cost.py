"""Test top-1 accuracy against inference time per image, one point per method."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

FIGURES_DIR = Path("results/figures/hog_svm")
FIG_NAME = "fig_accuracy_vs_cost"

COLOR_CPU = "#4C72B0"
COLOR_GPU = "#E69F00"
COLOR_MUTED = "0.45"
FS_AXIS_LABEL = 9
FS_TICK = 8
FS_LEGEND = 8
FS_ANNOT = 7

DOUBLE_COLUMN_WIDTH_IN = 6.9

# Measurement scope of a timing file.
FULL_PIPELINE = "full"          # JPEG decode + resize + features/forward + classify
FORWARD_ONLY = "forward_only"   # GPU forward pass only


def read_local_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def read_branch_json(branch: str, path: str) -> dict:
    """Read a JSON file from another branch, read-only, without checking it out."""
    ref = f"origin/{branch}:{path}"
    result = subprocess.run(
        ["git", "show", ref],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"cannot read {ref} -- fetch the teammates' branches first "
            f"(git fetch origin). git said: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


# Where each method's accuracy and timing come from, and what the timing covers.
METHODS = [
    {
        "key": "hog_svm",
        "label": "HOG + HSV colour + linear SVM",
        "metrics": ("local", "results/evaluation/hog_svm_metrics.json"),
        "timing": ("local", "results/evaluation/hog_svm_timing.json"),
        "timing_total_field": ("inference", "total_inference_seconds"),
        "timing_count_field": ("inference", "number_of_test_images"),
        "device": "CPU",
        "device_detail": "Apple M4 Pro, 12 cores",
        "scope": FULL_PIPELINE,
    },
    {
        "key": "resnet18_scratch",
        "label": "ResNet-18 from scratch",
        "metrics": (
            "scratch-resnet18",
            "results/evaluation/resnet18_scratch_aligned_metrics.json",
        ),
        "timing": (
            "scratch-resnet18",
            "results/evaluation/resnet18_scratch_aligned_inference_time.json",
        ),
        "timing_total_field": (None, "total_inference_seconds"),
        "timing_count_field": (None, "number_of_test_images"),
        "device": "CPU",
        "device_detail": "model not reported",
        "scope": FULL_PIPELINE,
    },
    {
        "key": "resnet18_frozen",
        "label": "ResNet-18 frozen features",
        "metrics": (
            "member4-aligned-transfer",
            "results/evaluation/resnet18_frozen_aligned_metrics.json",
        ),
        "timing": (
            "member4-aligned-transfer",
            "results/evaluation/resnet18_frozen_aligned_timing.json",
        ),
        "timing_total_field": (None, "total_gpu_forward_time_seconds"),
        "timing_count_field": (None, "number_of_images"),
        "device": "GPU",
        "device_detail": None,  # read from the "gpu" field of the timing file
        "scope": FORWARD_ONLY,
    },
    {
        "key": "resnet18_finetune",
        "label": "ResNet-18 fine-tuned",
        "metrics": (
            "member4-aligned-transfer",
            "results/evaluation/resnet18_finetune_aligned_metrics.json",
        ),
        "timing": (
            "member4-aligned-transfer",
            "results/evaluation/resnet18_finetune_aligned_timing.json",
        ),
        "timing_total_field": (None, "total_gpu_forward_time_seconds"),
        "timing_count_field": (None, "number_of_images"),
        "device": "GPU",
        "device_detail": None,
        "scope": FORWARD_ONLY,
    },
]


def load_source(source: tuple[str, str]) -> dict:
    origin, path = source
    if origin == "local":
        return read_local_json(Path(path))
    return read_branch_json(origin, path)


def pick(record: dict, field: tuple[str | None, str]) -> float:
    section, name = field
    node = record if section is None else record[section]
    return float(node[name])


def load_methods() -> list[dict]:
    loaded = []
    for spec in METHODS:
        metrics = load_source(spec["metrics"])
        timing = load_source(spec["timing"])

        total_seconds = pick(timing, spec["timing_total_field"])
        n_images = pick(timing, spec["timing_count_field"])
        if n_images <= 0:
            raise ValueError(f"{spec['key']}: non-positive image count {n_images}")

        detail = spec["device_detail"]
        if detail is None:
            detail = timing.get("gpu") or timing.get("hardware")
            if not detail:
                raise ValueError(
                    f"{spec['key']}: timing file names no device, refusing to guess"
                )
            detail = detail.replace("NVIDIA GeForce ", "").removesuffix(" GPU")

        loaded.append(
            {
                **spec,
                "device_detail": detail,
                "ms_per_image": 1000.0 * total_seconds / n_images,
                "top1": 100.0 * float(metrics["top1_accuracy"]),
                "top5": 100.0 * float(metrics["top5_accuracy"]),
                "macro_f1": 100.0 * float(metrics["macro_f1"]),
                "n_images": int(n_images),
                "total_seconds": total_seconds,
            }
        )
    return loaded


# Offsets in points; the two GPU points share an x, so their labels fan apart in y.
PLACEMENT = {
    "hog_svm": {"dx": 10, "dy": 0, "ha": "left", "va": "center"},
    "resnet18_scratch": {"dx": -11, "dy": 0, "ha": "right", "va": "center"},
    "resnet18_frozen": {"dx": 4, "dy": -10, "ha": "left", "va": "top"},
    "resnet18_finetune": {"dx": 4, "dy": 10, "ha": "left", "va": "bottom"},
}


def main() -> None:
    methods = load_methods()

    for method in methods:
        scope = (
            "full pipeline"
            if method["scope"] == FULL_PIPELINE
            else "GPU FORWARD PASS ONLY"
        )
        print(
            f"{method['key']:18s} "
            f"{method['ms_per_image']:8.3f} ms/img  "
            f"top1={method['top1']:5.2f}%  top5={method['top5']:5.2f}%  "
            f"macroF1={method['macro_f1']:5.2f}%  "
            f"{method['device']} ({method['device_detail']})  [{scope}]"
        )

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.0))
    ax.set_xscale("log")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.18, linewidth=0.5)

    for method in methods:
        is_forward_only = method["scope"] == FORWARD_ONLY
        color = COLOR_CPU if method["device"] == "CPU" else COLOR_GPU
        ax.scatter(
            method["ms_per_image"],
            method["top1"],
            s=64,
            marker="^" if is_forward_only else "o",
            facecolors="none" if is_forward_only else color,
            edgecolors=color,
            linewidths=1.5 if is_forward_only else 0.0,
            zorder=4,
        )

        place = PLACEMENT[method["key"]]
        ax.annotate(
            method["label"],
            xy=(method["ms_per_image"], method["top1"]),
            xytext=(place["dx"], place["dy"]),
            textcoords="offset points",
            ha=place["ha"],
            va=place["va"],
            fontsize=FS_ANNOT,
            linespacing=1.35,
            zorder=5,
        )

        # Direction only: decode and resize can only move these points right.
        if is_forward_only:
            ax.annotate(
                "",
                xy=(method["ms_per_image"] * 1.85, method["top1"]),
                xytext=(method["ms_per_image"] * 1.13, method["top1"]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": COLOR_MUTED,
                    "linewidth": 0.7,
                    "linestyle": (0, (2.5, 1.5)),
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=3,
            )

    ax.set_xlabel(
        "Inference time per image (ms, log scale)", fontsize=FS_AXIS_LABEL
    )
    ax.set_ylabel("Test top-1 accuracy (%)", fontsize=FS_AXIS_LABEL)

    times = [method["ms_per_image"] for method in methods]
    accuracies = [method["top1"] for method in methods]
    ax.set_xlim(min(times) / 2.1, max(times) * 2.0)
    ax.set_ylim(0, max(accuracies) * 1.28)

    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ticks = [t for t in ticks if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markersize=6,
            color=COLOR_CPU, markeredgewidth=0,
        ),
        Line2D(
            [], [], linestyle="none", marker="^", markersize=7,
            markerfacecolor="none", markeredgecolor=COLOR_GPU,
            markeredgewidth=1.5,
        ),
    ]
    labels = [
        "CPU, full pipeline",
        "GPU, forward pass only",
    ]
    ax.legend(
        handles,
        labels,
        fontsize=FS_LEGEND,
        loc="upper right",
        frameon=False,
        handlelength=1.2,
        handletextpad=0.6,
        borderaxespad=0.3,
        labelspacing=0.4,
    )

    # Fill the canvas so the tight bounding box comes out at the nominal
    # double-column width, keeping point sizes consistent across the figure set.
    fig.tight_layout(pad=0.25)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{FIG_NAME}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
