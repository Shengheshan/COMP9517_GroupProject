"""Validation top-1 accuracy against SVM C, one line per feature configuration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "results" / "tables"
FIGURE_DIR = REPO_ROOT / "results" / "figures"
FIGURE_NAME = "fig_c_sweep"

PALETTE = ["#4C72B0", "#E69F00", "#55A868", "#C44E52", "#8172B3", "#937860"]

# filename prefix -> (legend label, palette index)
CONFIGS = [
    ("hog_svm_", "HOG only", 0),
    ("hogl2_", "HOG + L2", 1),
    ("highres_", "HOG 8x8 cells", 2),
    ("color_", "HOG + HSV colour", 3),
]

# offsets in points that keep the four best-point labels from colliding
LABEL_OFFSETS = {
    "HOG only": (6, 4),
    "HOG + L2": (6, 3),
    "HOG 8x8 cells": (2, -11),
    "HOG + HSV colour": (2, 6),
}


def load_runs() -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Collect (C, top-1 accuracy) points and the feature dimension per configuration."""
    files = sorted(TABLE_DIR.glob("*_validation.json"))
    if not files:
        raise SystemExit(f"no *_validation.json files found in {TABLE_DIR}")

    runs: dict[str, list[dict[str, float]]] = {label: [] for _, label, _ in CONFIGS}
    dims: dict[str, set[int]] = {label: set() for _, label, _ in CONFIGS}
    unassigned: list[str] = []

    for path in files:
        matches = [(p, label) for p, label, _ in CONFIGS if path.name.startswith(p)]
        if len(matches) != 1:
            unassigned.append(path.name)
            continue
        label = matches[0][1]
        payload = json.loads(path.read_text())
        runs[label].append(
            {
                "C": float(payload["classifier"]["C"]),
                "top1_percent": 100.0 * float(
                    payload["validation_metrics"]["top1_accuracy"]
                ),
                "source": path.name,
            }
        )
        dims[label].add(int(payload["feature_dimension"]))

    if unassigned:
        raise SystemExit(
            "these result files matched no (or several) configuration prefixes, "
            f"refusing to plot a partial sweep: {unassigned}"
        )

    for label, points in runs.items():
        if not points:
            raise SystemExit(f"configuration '{label}' has no runs on disk")
        points.sort(key=lambda row: row["C"])
        if len(dims[label]) != 1:
            raise SystemExit(
                f"configuration '{label}' mixes feature dimensions {dims[label]}"
            )

    return runs, {label: next(iter(d)) for label, d in dims.items()}


def draw_panel(ax, runs, dims, *, relative: bool) -> None:
    for _, label, colour_index in CONFIGS:
        points = runs[label]
        cs = [row["C"] for row in points]
        accuracies = [row["top1_percent"] for row in points]
        best = max(accuracies)
        ys = [100.0 * acc / best for acc in accuracies] if relative else accuracies
        colour = PALETTE[colour_index]

        legend_label = f"{label} ({dims[label]}-d)"
        ax.plot(
            cs,
            ys,
            color=colour,
            linewidth=1.2,
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=legend_label if not relative else None,
            zorder=3,
        )

        best_x = [c for c, acc in zip(cs, accuracies) if acc == best]
        best_y = [y for y, acc in zip(ys, accuracies) if acc == best]
        ax.plot(
            best_x,
            best_y,
            linestyle="none",
            marker="o",
            markersize=5.0,
            color=colour,
            markeredgecolor=colour,
            zorder=4,
        )

        if not relative:
            dx, dy = LABEL_OFFSETS[label]
            ax.annotate(
                f"{best:.2f}",
                xy=(best_x[0], best_y[0]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                color=colour,
                ha="left",
                va="center",
            )

    ax.set_xscale("log")
    ax.set_xlabel("SVM regularisation strength $C$ (log scale)")
    ax.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1e0, 1e1]))
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(2e-4, 2e1)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def main() -> None:
    runs, dims = load_runs()

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_abs, ax_rel) = plt.subplots(
        1, 2, figsize=(6.9, 2.75), constrained_layout=True
    )

    draw_panel(ax_abs, runs, dims, relative=False)
    ax_abs.set_title("(a) Absolute accuracy")
    ax_abs.set_ylabel("Validation top-1 accuracy (%)")
    top = max(row["top1_percent"] for points in runs.values() for row in points)
    ax_abs.set_ylim(0, top * 1.25)

    draw_panel(ax_rel, runs, dims, relative=True)
    ax_rel.set_title("(b) Relative to each configuration's best")
    ax_rel.set_ylabel("Top-1 accuracy, % of\nconfiguration's own best")
    worst_relative = min(
        100.0
        * row["top1_percent"]
        / max(other["top1_percent"] for other in points)
        for points in runs.values()
        for row in points
    )
    ax_rel.set_ylim(worst_relative - 8, 106)

    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"{FIGURE_NAME}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    plt.close(fig)

    for _, label, _ in CONFIGS:
        points = runs[label]
        best = max(points, key=lambda row: row["top1_percent"])
        print(
            f"{label:20s} dim={dims[label]:5d} "
            f"C in [{points[0]['C']:g}, {points[-1]['C']:g}] "
            f"n={len(points)} best={best['top1_percent']:.2f}% at C={best['C']:g} "
            f"({best['source']})"
        )
        for row in points:
            print(f"    C={row['C']:<8g} top1={row['top1_percent']:6.2f}%")


if __name__ == "__main__":
    main()
