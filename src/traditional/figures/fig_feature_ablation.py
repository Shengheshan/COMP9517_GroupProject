"""Best validation accuracy of each feature configuration as grouped bars."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
FIG_NAME = "fig_feature_ablation"

COLOR_TOP1 = "#4C72B0"
COLOR_TOP5 = "#E69F00"
COLOR_CHANCE = "#C44E52"
FS_AXIS_LABEL = 9
FS_TICK = 8
FS_LEGEND = 8
FS_ANNOT = 7

SINGLE_COLUMN_WIDTH_IN = 3.3

CONFIGS = [
    ("hog_svm_", "HOG\n16×16\ncells"),
    ("hogl2_", "HOG\n+ L2\nnorm"),
    ("highres_", "HOG\n8×8\ncells"),
    ("color_", "HOG\n+ HSV\ncolour"),
]


def load_sweep(prefix: str) -> list[dict]:
    """Load every validation JSON belonging to one feature configuration."""
    paths = sorted(TABLES_DIR.glob(f"{prefix}*_validation.json"))
    if not paths:
        raise FileNotFoundError(f"no validation tables found for prefix {prefix!r}")
    runs = []
    for path in paths:
        with path.open() as handle:
            record = json.load(handle)
        metrics = record["validation_metrics"]
        runs.append(
            {
                "path": path,
                "C": float(record["classifier"]["C"]),
                "dim": int(record["feature_dimension"]),
                "num_classes": int(record["num_classes"]),
                "top1": 100.0 * metrics["top1_accuracy"],
                "top5": 100.0 * metrics["top5_accuracy"],
                "macro_f1": 100.0 * metrics["macro_f1"],
            }
        )
    dims = {run["dim"] for run in runs}
    if len(dims) != 1:
        raise ValueError(f"prefix {prefix!r} mixes feature dimensions: {sorted(dims)}")
    return runs


def best_run(runs: list[dict]) -> dict:
    """Best run of a sweep: top-1 accuracy, ties broken by top-5 then macro-F1."""
    return max(runs, key=lambda run: (run["top1"], run["top5"], run["macro_f1"]))


def main() -> None:
    selected = []
    for prefix, label in CONFIGS:
        runs = load_sweep(prefix)
        best = best_run(runs)
        best["label"] = label
        best["prefix"] = prefix
        best["n_runs"] = len(runs)
        best["ties"] = sorted(
            run["C"]
            for run in runs
            if run["top1"] == best["top1"] and run["C"] != best["C"]
        )
        selected.append(best)

    num_classes = {run["num_classes"] for run in selected}
    if len(num_classes) != 1:
        raise ValueError(f"inconsistent num_classes across configs: {num_classes}")
    chance = 100.0 / num_classes.pop()

    for run in selected:
        print(
            f"{run['prefix']:9s} dim={run['dim']:5d} best C={run['C']:<7g} "
            f"top1={run['top1']:.2f}% top5={run['top5']:.2f}% "
            f"(macro-F1 {run['macro_f1']:.2f}%, {run['n_runs']} runs in sweep)"
        )
        if run["ties"]:
            tied = ", ".join(f"C={c:g}" for c in run["ties"])
            print(
                f"{'':9s} note: top-1 tied with {tied}; "
                "broke the tie on top-5 accuracy"
            )
    print(f"chance level = {chance:.2f}%")

    x = list(range(len(selected)))
    width = 0.38
    top1 = [run["top1"] for run in selected]
    top5 = [run["top5"] for run in selected]

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, 2.9))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(False)

    bars1 = ax.bar(
        [xi - width / 2 for xi in x], top1, width, color=COLOR_TOP1, linewidth=0
    )
    bars5 = ax.bar(
        [xi + width / 2 for xi in x], top5, width, color=COLOR_TOP5, linewidth=0
    )

    for bars in (bars1, bars5):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FS_ANNOT,
            )

    ax.axhline(chance, color=COLOR_CHANCE, linestyle="--", linewidth=0.8, zorder=2)

    ax.set_ylabel("Validation accuracy (%)", fontsize=FS_AXIS_LABEL)
    ax.set_ylim(0, max(top5) * 1.34)
    ax.set_xlim(-0.62, len(selected) - 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [run["label"] for run in selected], fontsize=FS_TICK, linespacing=1.25
    )
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)

    for xi, run in zip(x, selected):
        ax.annotate(
            f"{run['dim']}-d\n$C$ = {run['C']:g}",
            xy=(xi, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -36),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=FS_ANNOT,
            color="0.35",
            linespacing=1.35,
        )

    handles = [
        bars1,
        bars5,
        Line2D([], [], color=COLOR_CHANCE, linestyle="--", linewidth=0.8),
    ]
    labels = [
        "Top-1 accuracy",
        "Top-5 accuracy",
        f"Random guessing ({chance:g}%)",
    ]
    ax.legend(
        handles,
        labels,
        fontsize=FS_LEGEND,
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        handletextpad=0.6,
        borderaxespad=0.2,
        labelspacing=0.35,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{FIG_NAME}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
