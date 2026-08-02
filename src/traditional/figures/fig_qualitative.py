"""Correct and same-genus misclassified example panels for the delivered model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
PREDICTIONS = REPO / "results/predictions/hog_svm.csv"
CONFUSION = REPO / "results/evaluation/hog_svm_confusion_matrix.npy"
TIMING = REPO / "results/evaluation/hog_svm_timing.json"  # model name + configuration
LABEL_MAP = REPO / "data_metadata/label_to_category.json"
OUT_DIR = REPO / "results/figures/hog_svm"
NAME = "fig_qualitative"

# Images are read from outside the repository, never copied.
DATASET_ROOT = Path(os.environ.get("COMP9517_DATA_ROOT", "data/raw"))

GREEN = "#55A868"
RED = "#C44E52"
GREY = "#555555"
FIG_WIDTH_IN = 6.9  # double column
PAD_IN = 0.01  # kept tiny so the saved width is exact
THUMB_PX = 512

N_COL = 4
NOTE_PT = 7
LINE_IN = 0.135  # baseline pitch for 7 pt text
CAP_LINES = 3  # every panel caption is exactly three lines
WS_IN = 0.05  # horizontal gap between panels
GAP_IN = 0.10  # gap between the top row's caption block and the bottom row


def load_label_map() -> dict[int, dict]:
    """Return {label_index: taxonomy dict}, parsing defensively."""
    raw = json.loads(LABEL_MAP.read_text())
    out: dict[int, dict] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                import ast

                value = ast.literal_eval(value)
        out[int(key)] = value
    return out


def model_description() -> str:
    """Read the delivered model's name and configuration from the timing record."""
    meta = json.loads(TIMING.read_text())
    method = meta["method"]
    config = meta.get("configuration", "")
    setting = config.split(" with ", 1)[1] if " with " in config else config
    return f"{method}, {setting}" if setting else method


def square_thumbnail(path: Path, side: int = THUMB_PX) -> np.ndarray:
    """Centre-crop to a square and resample to a fixed side length."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    edge = min(w, h)
    left = (w - edge) // 2
    top = (h - edge) // 2
    img = img.crop((left, top, left + edge, top + edge))
    img = img.resize((side, side), Image.LANCZOS)
    return np.asarray(img)


def first_image(preds: pd.DataFrame, true_label: int, pred_label: int) -> str:
    """Lexicographically first prediction row with this (true, predicted) pair."""
    rows = preds[
        (preds.true_label == true_label) & (preds.predicted_label == pred_label)
    ]
    if rows.empty:
        sys.exit(f"no prediction row for true={true_label} predicted={pred_label}")
    return str(rows.relative_path.sort_values().iloc[0])


def success_labels(cm: np.ndarray, k: int) -> list[int]:
    """Classes with the highest recall; ties broken by ascending label index."""
    diag = np.diag(cm)
    order = sorted(range(cm.shape[0]), key=lambda i: (-int(diag[i]), i))
    return order[:k]


def congeneric_cells(cm: np.ndarray, genus: list[str], k: int) -> list[tuple[int, int, int]]:
    """Largest same-genus off-diagonal cells as (count, true, predicted).

    Ranked by count, then true label, then predicted label, so the order is total.
    """
    cells = [
        (int(cm[t, p]), t, p)
        for t in range(cm.shape[0])
        for p in range(cm.shape[1])
        if t != p and cm[t, p] > 0 and genus[t] == genus[p]
    ]
    cells.sort(key=lambda c: (-c[0], c[1], c[2]))
    if len(cells) < k:
        sys.exit(f"only {len(cells)} congeneric confusion cells, need {k}")
    return cells[:k]


def taxonomic_error_stats(cm: np.ndarray, labels: dict[int, dict]) -> dict[str, float]:
    """Same-genus / same-family error shares plus a uniform-chance expectation."""
    n = cm.shape[0]
    genus = [labels[i]["genus"] for i in range(n)]
    family = [labels[i]["family"] for i in range(n)]
    genus_size: dict[str, int] = {}
    for g in genus:
        genus_size[g] = genus_size.get(g, 0) + 1

    errors = int(cm.sum() - np.trace(cm))
    same_genus = 0
    same_family = 0
    for t in range(n):
        for p in range(n):
            if t == p or cm[t, p] == 0:
                continue
            count = int(cm[t, p])
            if genus[t] == genus[p]:
                same_genus += count
            if family[t] == family[p]:
                same_family += count

    # Null model: every error spread uniformly over the other n-1 classes.
    expected_genus = sum(
        int(cm[t].sum() - cm[t, t]) * (genus_size[genus[t]] - 1) / (n - 1)
        for t in range(n)
    )
    return dict(
        errors=errors,
        same_genus=same_genus,
        same_family=same_family,
        cross_family=errors - same_family,
        expected_genus=expected_genus,
        genus_ratio=same_genus / expected_genus if expected_genus else float("nan"),
    )


def largest_off_diagonal(cm: np.ndarray, genus: list[str]) -> tuple[int, int, bool]:
    """(count, how many cells share it, whether any of them is congeneric)."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = int(off.max())
    ties = [(t, p) for t, p in zip(*np.where(off == top))]
    return top, len(ties), any(genus[t] == genus[p] for t, p in ties)


def main() -> int:
    for required in (PREDICTIONS, CONFUSION, TIMING, LABEL_MAP):
        if not required.exists():
            sys.exit(f"missing source file: {required}")
    if not (DATASET_ROOT / "val").is_dir():
        sys.exit(
            f"dataset root does not contain val/: {DATASET_ROOT}\n"
            "Point COMP9517_DATA_ROOT at the directory holding train_mini/ "
            "and val/, then run this script again."
        )

    labels = load_label_map()
    preds = pd.read_csv(PREDICTIONS)
    cm = np.load(CONFUSION)
    n_class = cm.shape[0]
    genus = [labels[i]["genus"] for i in range(n_class)]

    row_total = cm.sum(axis=1)  # test images per species
    col_total = cm.sum(axis=0)  # times the species was predicted
    correct = np.diag(cm)
    n_test = int(cm.sum())
    n_correct = int(np.trace(cm))
    top1 = n_correct / n_test
    per_class = int(np.unique(row_total)[0]) if len(np.unique(row_total)) == 1 else None
    if per_class is None:
        sys.exit("test set is not balanced; the '10 images per species' note would lie")

    stats = taxonomic_error_stats(cm, labels)
    big_count, big_ties, big_congeneric = largest_off_diagonal(cm, genus)

    panels: list[dict] = []

    for lab in success_labels(cm, N_COL):
        species = labels[lab]["species_name"]
        rel = first_image(preds, lab, lab)
        rec = int(correct[lab]) / int(row_total[lab])
        prec = int(correct[lab]) / int(col_total[lab])
        panels.append(
            dict(
                kind="ok",
                path=DATASET_ROOT / rel,
                lines=[
                    (species, GREEN, "italic"),
                    (f"recall {correct[lab]}/{row_total[lab]} = {rec:.0%}",
                     GREY, "normal"),
                    (f"precision {correct[lab]}/{col_total[lab]} = {prec:.0%}",
                     RED, "normal"),
                ],
                report=(
                    f"{species}: recall {int(correct[lab])}/{int(row_total[lab])} "
                    f"= {rec:.1%}, precision {int(correct[lab])}/"
                    f"{int(col_total[lab])} = {prec:.1%}"
                ),
            )
        )

    for count, t_lab, p_lab in congeneric_cells(cm, genus, N_COL):
        true_species = labels[t_lab]["species_name"]
        pred_species = labels[p_lab]["species_name"]
        rel = first_image(preds, t_lab, p_lab)
        panels.append(
            dict(
                kind="bad",
                path=DATASET_ROOT / rel,
                lines=[
                    (true_species, RED, "italic"),
                    (f"→ {pred_species}", RED, "italic"),
                    (f"{count} of {row_total[t_lab]} images → this class",
                     GREY, "normal"),
                ],
                report=(
                    f"{true_species} -> {pred_species}: {count}/"
                    f"{int(row_total[t_lab])} images (one confusion cell)"
                ),
            )
        )

    for panel in panels:
        if len(panel["lines"]) != CAP_LINES:
            sys.exit("caption line budget changed; update CAP_LINES")

    cap_in = CAP_LINES * LINE_IN + 0.04
    side_max = (FIG_WIDTH_IN - (N_COL - 1) * WS_IN) / N_COL
    fig_height = 2 * side_max + 2 * cap_in + GAP_IN + 2 * PAD_IN

    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
    axes: list[plt.Axes] = []
    for panel in panels:
        ax = fig.add_axes((0.0, 0.0, 0.1, 0.1))
        ax.imshow(square_thumbnail(panel["path"]), interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        edge = GREEN if panel["kind"] == "ok" else RED
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(edge)
            spine.set_linewidth(0.8)
        axes.append(ax)

    for idx, (panel, ax) in enumerate(zip(panels, axes)):
        if idx % N_COL == 0:
            ax.set_ylabel(
                "Correct" if panel["kind"] == "ok" else "Misclassified",
                fontsize=9,
                color=GREEN if panel["kind"] == "ok" else RED,
                labelpad=3,
            )

    caption_texts: list[list[plt.Text]] = []
    for panel, ax in zip(panels, axes):
        handles = []
        for li, (text, colour, style) in enumerate(panel["lines"]):
            handles.append(
                ax.text(
                    0.0,
                    0.0,
                    text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=NOTE_PT,
                    color=colour,
                    fontstyle=style,
                )
            )
        caption_texts.append(handles)

    def place(band_left: float, band_w: float) -> None:
        """Position every artist for a panel band of `band_w` inches at `band_left`."""
        side = (band_w - (N_COL - 1) * WS_IN) / N_COL
        row_top = [
            fig_height - PAD_IN,
            fig_height - PAD_IN - side - cap_in - GAP_IN,
        ]
        for idx, ax in enumerate(axes):
            r, c = divmod(idx, N_COL)
            x = band_left + c * (side + WS_IN)
            ax.set_position(
                (
                    x / FIG_WIDTH_IN,
                    (row_top[r] - side) / fig_height,
                    side / FIG_WIDTH_IN,
                    side / fig_height,
                )
            )
        step = LINE_IN / side  # axes-height units
        for handles in caption_texts:
            for li, handle in enumerate(handles):
                handle.set_position((0.0, -0.035 - li * step))

    # bbox_inches='tight' crops to the artists, so shrink the panel band until the
    # cropped output is exactly FIG_WIDTH_IN wide.
    band_left, band_w = PAD_IN, FIG_WIDTH_IN - 2 * PAD_IN
    for _ in range(4):
        place(band_left, band_w)
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        axes_x0 = min(a.get_position().x0 for a in axes) * FIG_WIDTH_IN
        axes_x1 = max(a.get_position().x1 for a in axes) * FIG_WIDTH_IN
        over_left = max(0.0, axes_x0 - tight.x0)
        over_right = max(0.0, tight.x1 - axes_x1)
        band_left = PAD_IN + over_left
        band_w = FIG_WIDTH_IN - 2 * PAD_IN - over_left - over_right
    place(band_left, band_w)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    final = fig.get_tightbbox(renderer)
    print(
        f"saved size: {final.width + 2 * PAD_IN:.3f} x "
        f"{final.height + 2 * PAD_IN:.3f} in (target width {FIG_WIDTH_IN})"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{NAME}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=PAD_IN)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    plt.close(fig)

    print(f"\nmodel: {model_description()}")
    print(
        f"top-1 {top1:.2%} ({n_correct}/{n_test}); "
        f"{n_class} species x {per_class} test images"
    )
    print("values plotted:")
    for panel in panels:
        print("  " + panel["report"])
    print(
        f"  errors {stats['errors']}, same-genus {stats['same_genus']} "
        f"({stats['same_genus'] / stats['errors']:.2%}), same-family "
        f"{stats['same_family']} ({stats['same_family'] / stats['errors']:.2%}), "
        f"cross-family {stats['cross_family']} "
        f"({stats['cross_family'] / stats['errors']:.2%})"
    )
    print(
        f"  expected same-genus under uniform chance {stats['expected_genus']:.3f} "
        f"-> ratio {stats['genus_ratio']:.2f}x"
    )
    print(
        f"  largest off-diagonal cell {big_count} ({big_ties} cells tied, "
        f"congeneric: {big_congeneric})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
