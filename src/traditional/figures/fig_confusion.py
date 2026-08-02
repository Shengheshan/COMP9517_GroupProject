"""Species-block and kingdom-level confusion matrices for the delivered model."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[3]
CM_PATH = REPO / "results/evaluation/hog_svm_confusion_matrix.npy"
METRICS_PATH = REPO / "results/evaluation/hog_svm_metrics.json"
META_PATH = REPO / "data_metadata/label_to_category.json"
OUT_DIR = REPO / "results/figures/hog_svm"
NAME = "fig_confusion"

# Largest block panel (a) can render legibly at 7 pt species labels.
MAX_BLOCK = 30
IMAGES_PER_CLASS = 10  # test images per species

# Space reserved for the in-matrix key, in cells.
KEY_ROWS, KEY_COLS = 3, 11

FS_AXIS, FS_TICK, FS_ANNOT = 9, 8, 7
FS_SPECIES = 7  # the block's binomials do not fit at 8 pt
CMAP = "Blues"
GREY = "#D9D9D9"  # diagonal cells in panel (a)


def load_metadata(path: Path) -> dict[int, dict[str, str]]:
    """Load label -> taxonomy. Parses defensively: values may be dicts or dict-like strings."""
    with path.open() as fh:
        raw = json.load(fh)

    out: dict[int, dict[str, str]] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                value = ast.literal_eval(value)
        if not isinstance(value, dict):
            raise TypeError(f"label {key}: unsupported metadata value type {type(value)}")
        out[int(key)] = {str(k): str(v) for k, v in value.items()}
    return out


def taxon(meta: dict[int, dict[str, str]], label: int, rank: str) -> str:
    """Taxon of `label` at `rank`, with a per-label sentinel when the field is absent.

    The sentinel keeps two labels with missing taxonomy from counting as a match.
    """
    value = meta[label].get(rank)
    return value if value else f"<unknown {rank} for label {label}>"


def abbreviate(binomial: str) -> str:
    """'Melanitta deglandi' -> 'M. deglandi'."""
    parts = binomial.split()
    if len(parts) < 2:
        return binomial
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def unique_labels(names: list[str]) -> list[str]:
    """Abbreviate, falling back to the full binomial where abbreviations collide."""
    short = [abbreviate(n) for n in names]
    seen: dict[str, int] = {}
    for s in short:
        seen[s] = seen.get(s, 0) + 1
    return [full if seen[s] > 1 else s for s, full in zip(short, names)]


def sanity_check(cm: np.ndarray, metrics: dict) -> list[str]:
    """Verify the matrix against the reported metrics. Returns a list of warnings."""
    warnings: list[str] = []
    row_sums = cm.sum(axis=1)
    if not np.all(row_sums == IMAGES_PER_CLASS):
        bad = np.flatnonzero(row_sums != IMAGES_PER_CLASS)
        warnings.append(
            f"row sums are not all {IMAGES_PER_CLASS}: "
            f"{len(bad)} offending rows, values {sorted(set(row_sums.tolist()))}"
        )

    total = int(cm.sum())
    correct = int(np.trace(cm))
    expected = metrics["top1_accuracy"] * total
    if abs(correct - expected) > 0.5:
        warnings.append(
            f"diagonal sum {correct} disagrees with top1_accuracy "
            f"{metrics['top1_accuracy']} x {total} = {expected:.1f}"
        )

    print(f"[check] matrix {cm.shape}, total {total}, diagonal {correct} "
          f"(top-1 {100.0 * correct / total:.2f}%)")
    print(f"[check] row sums all equal {IMAGES_PER_CLASS}: "
          f"{bool(np.all(row_sums == IMAGES_PER_CLASS))}")
    for w in warnings:
        print(f"[WARNING] {w}")
    return warnings


def select_confused_species(off: np.ndarray, max_block: int) -> tuple[list[int], int, int]:
    """Every species involved in the top tier of off-diagonal cells.

    Let t be the smallest cell count whose cells' true and predicted species still fit
    in `max_block` rows; the block is exactly that union, so no cell of count >= t is
    dropped. Species are ordered by the rank of the cell that introduced them, which
    puts the strongest confusions towards the upper left.

    Returns (selected labels, t, number of off-diagonal cells with count >= t).
    """
    if off.max() < 1:
        raise ValueError("no off-diagonal errors to show")

    def union(min_count: int) -> tuple[list[int], int]:
        i_idx, j_idx = np.nonzero(off >= min_count)
        counts = off[i_idx, j_idx]
        order = np.lexsort((j_idx, i_idx, -counts))
        selected: list[int] = []
        for pos in order:
            for label in (int(i_idx[pos]), int(j_idx[pos])):
                if label not in selected:
                    selected.append(label)
        return selected, len(order)

    best: tuple[list[int], int, int] | None = None
    for min_count in range(int(off.max()), 0, -1):
        selected, n_cells = union(min_count)
        if len(selected) > max_block:
            break
        best = (selected, min_count, n_cells)
    if best is None:
        raise ValueError(f"even the largest cells need more than {max_block} species")
    return best


def error_taxonomy(off: np.ndarray, meta: dict[int, dict[str, str]]) -> dict:
    """How much of the error mass stays inside the true genus / family, and its enrichment.

    The reference rate is a predictor that, whenever it is wrong, picks uniformly among the
    other C - 1 classes: the expected same-rank share is then the mean over true classes of
    (species sharing that taxon - 1) / (C - 1).
    """
    n = off.shape[0]
    total = int(off.sum())
    ranks = ("genus", "family")
    names = {r: [taxon(meta, i, r) for i in range(n)] for r in ranks}
    sizes = {r: {t: names[r].count(t) for t in set(names[r])} for r in ranks}

    out: dict = {}
    i_idx, j_idx = np.nonzero(off)
    for r in ranks:
        same = sum(int(off[i, j]) for i, j in zip(i_idx, j_idx) if names[r][i] == names[r][j])
        expected_frac = float(np.mean([(sizes[r][names[r][i]] - 1) / (n - 1) for i in range(n)]))
        expected = expected_frac * total
        out[r] = {
            "same": same,
            "pct": 100.0 * same / total,
            "expected": expected,
            "enrichment": same / expected if expected > 0 else float("nan"),
            "multi_taxa": sum(1 for c in sizes[r].values() if c > 1),
            "n_taxa": len(sizes[r]),
        }
    out["total"] = total
    out["cross_family"] = total - out["family"]["same"]
    out["cross_family_pct"] = 100.0 * out["cross_family"] / total
    return out


def directionality(off: np.ndarray, min_count: int) -> dict:
    """Split the cells of count >= min_count into one-directional and reciprocated ones."""
    i_idx, j_idx = np.nonzero(off >= min_count)
    one_way, reciprocated, symmetric = [], [], []
    for i, j in zip(i_idx, j_idx):
        forward, reverse = int(off[i, j]), int(off[j, i])
        entry = (int(i), int(j), forward, reverse)
        if reverse == 0:
            one_way.append(entry)
        else:
            reciprocated.append(entry)
            if forward == reverse:
                symmetric.append(entry)
    return {
        "n_cells": len(i_idx),
        "one_way": one_way,
        "reciprocated": reciprocated,
        "symmetric": symmetric,
    }


def aggregate_by_kingdom(
    cm: np.ndarray, meta: dict[int, dict[str, str]]
) -> tuple[list[str], np.ndarray]:
    """Aggregate the full matrix by kingdom, ordered by number of species (descending)."""
    kingdoms = [taxon(meta, i, "kingdom") for i in range(cm.shape[0])]
    species_per_kingdom: dict[str, int] = {}
    for k in kingdoms:
        species_per_kingdom[k] = species_per_kingdom.get(k, 0) + 1
    names = sorted(species_per_kingdom, key=lambda k: (-species_per_kingdom[k], k))

    index = {k: i for i, k in enumerate(names)}
    group = np.array([index[k] for k in kingdoms])

    agg = np.zeros((len(names), len(names)), dtype=np.int64)
    for a in range(len(names)):
        rows = cm[group == a]
        for b in range(len(names)):
            agg[a, b] = rows[:, group == b].sum()
    return names, agg


def chance_within_kingdom(meta: dict[int, dict[str, str]], n_classes: int) -> float:
    """Within-kingdom hit rate of a uniform-random predictor over all classes, as a percentage.

    Every species contributes the same number of test images, so the support of each
    kingdom is proportional to its species count.
    """
    kingdoms = [taxon(meta, i, "kingdom") for i in range(n_classes)]
    counts: dict[str, int] = {}
    for k in kingdoms:
        counts[k] = counts.get(k, 0) + 1
    return 100.0 * sum((c / n_classes) ** 2 for c in counts.values())


def draw_panel_a(
    ax, cax, cm: np.ndarray, selected: list[int], labels: list[str],
    min_count: int, n_cells: int,
) -> dict:
    """Sub-matrix heat map: off-diagonal cells coloured, diagonal cells greyed."""
    block = cm[np.ix_(selected, selected)].astype(float)
    n = len(selected)
    diag = np.diag(block).astype(int)

    errors = block.copy()
    np.fill_diagonal(errors, np.nan)
    vmax = np.nanmax(errors)

    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(GREY)  # diagonal
    norm = Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(errors, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, str(diag[i]), ha="center", va="center",
                        fontsize=FS_ANNOT - 1, color="#555555")
            elif block[i, j] > 0:
                shade = cmap(norm(block[i, j]))
                luma = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
                ax.text(j, i, str(int(block[i, j])), ha="center", va="center",
                        fontsize=FS_ANNOT - 1, color="white" if luma < 0.55 else "#1A1A1A")

    # The key sits in the upper-right corner, so check that corner really is free.
    corner = errors[:min(KEY_ROWS, n), max(0, n - KEY_COLS):]
    warnings: list[str] = []
    if np.nansum(corner) > 0:
        warnings.append(
            f"panel (a) key overlaps {int(np.count_nonzero(np.nan_to_num(corner)))} "
            f"non-empty cell(s) in the top-right {KEY_ROWS}x{KEY_COLS} corner"
        )
    ax.text(n - 0.85, 0.35, f"Grey: correct of {IMAGES_PER_CLASS}",
            ha="right", va="center", fontsize=FS_ANNOT, color="#555555")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=FS_SPECIES, rotation=90)
    ax.set_yticklabels(labels, fontsize=FS_SPECIES, style="italic")
    for lbl in ax.get_xticklabels():
        lbl.set_style("italic")
    ax.set_xlabel("Predicted species", fontsize=FS_AXIS)
    ax.set_ylabel("True species", fontsize=FS_AXIS)
    ax.set_title(
        f"(a) All {n_cells} confusions of {min_count}+ images ({n} species)",
        fontsize=FS_AXIS, pad=6,
    )
    ax.tick_params(length=2, width=0.5)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_linewidth(0.5)
        side.set_color("#AAAAAA")

    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Misclassified images (count)", fontsize=FS_AXIS, labelpad=2)
    cbar.set_ticks(np.arange(0, vmax + 1, 1))
    cbar.ax.tick_params(labelsize=FS_TICK, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_color("#AAAAAA")

    for w in warnings:
        print(f"[WARNING] {w}")
    return {
        "block": block,
        "in_block_errors": float(np.nansum(errors)),
        "vmax": float(vmax),
        "diag": diag,
        "warnings": warnings,
    }


def draw_panel_b(ax, cax, names: list[str], agg: np.ndarray) -> dict:
    """Kingdom-level row-normalised confusion matrix with percentages in every cell."""
    support = agg.sum(axis=1, keepdims=True)
    pct = 100.0 * agg / support
    n = len(names)

    cmap = plt.get_cmap(CMAP)
    norm = Normalize(vmin=0, vmax=100)
    im = ax.imshow(pct, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for i in range(n):
        for j in range(n):
            shade = cmap(norm(pct[i, j]))
            luma = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
            colour = "white" if luma < 0.55 else "#1A1A1A"
            ax.text(j, i - 0.10, f"{pct[i, j]:.1f}%", ha="center", va="center",
                    fontsize=FS_ANNOT, color=colour)
            ax.text(j, i + 0.17, f"({agg[i, j]:,})", ha="center", va="center",
                    fontsize=FS_ANNOT - 1.5, color=colour)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=FS_TICK, rotation=90)
    ax.set_yticklabels(names, fontsize=FS_TICK)
    ax.set_xlabel("Predicted kingdom", fontsize=FS_AXIS, labelpad=3)
    ax.set_ylabel("True kingdom", fontsize=FS_AXIS, labelpad=3)
    ax.set_title("(b) Errors by kingdom", fontsize=FS_AXIS, pad=6)
    ax.tick_params(length=2, width=0.5, pad=2)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_linewidth(0.5)
        side.set_color("#AAAAAA")

    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Share of true-kingdom images (%)", fontsize=FS_ANNOT, labelpad=2)
    cbar.set_ticks([0, 50, 100])
    cbar.ax.tick_params(labelsize=FS_TICK, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_color("#AAAAAA")

    return {
        "pct": pct,
        "agg": agg,
        "within_n": int(np.trace(agg)),
        "total": int(agg.sum()),
        "within": 100.0 * np.trace(agg) / agg.sum(),
        "support": support.ravel(),
    }


def main() -> None:
    cm = np.load(CM_PATH)
    with METRICS_PATH.open() as fh:
        metrics = json.load(fh)
    meta = load_metadata(META_PATH)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"expected a square confusion matrix, got {cm.shape}")
    missing = set(range(cm.shape[0])) - set(meta)
    if missing:
        raise KeyError(f"metadata missing {len(missing)} labels, e.g. {sorted(missing)[:5]}")

    sanity_check(cm, metrics)

    off = cm.copy()
    np.fill_diagonal(off, 0)

    selected, min_count, n_cells = select_confused_species(off, MAX_BLOCK)
    species = [meta[i]["species_name"] for i in selected]
    labels = unique_labels(species)

    kingdom_names, agg = aggregate_by_kingdom(cm, meta)
    chance = chance_within_kingdom(meta, cm.shape[0])
    tax = error_taxonomy(off, meta)
    direction = directionality(off, min_count)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Explicit inch-based layout; both heat maps get square axes rectangles.
    # Panel (a)'s colourbar is horizontal and sits below its x tick labels.
    FIG_W = 6.9                          # double-column width
    A_X, A_Y, A_SIDE = 1.22, 1.78, 3.50  # panel (a) matrix
    CA_X, CA_W, CA_Y, CA_H = 1.92, 2.10, 0.42, 0.11  # panel (a) colourbar
    B_X, B_SIDE = 5.44, 1.42             # panel (b) matrix
    B_Y = A_Y + A_SIDE - B_SIDE          # top-align the two matrices
    CB_Y, CB_H = 2.86, 0.11              # panel (b) colourbar
    FIG_H = A_Y + A_SIDE + 0.28          # room for the panel titles

    def rect(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = fig.add_axes(rect(A_X, A_Y, A_SIDE, A_SIDE))
    cax_a = fig.add_axes(rect(CA_X, CA_Y, CA_W, CA_H))
    ax_b = fig.add_axes(rect(B_X, B_Y, B_SIDE, B_SIDE))
    cax_b = fig.add_axes(rect(B_X, CB_Y, B_SIDE, CB_H))

    info_a = draw_panel_a(ax_a, cax_a, cm, selected, labels, min_count, n_cells)
    info_b = draw_panel_b(ax_b, cax_b, kingdom_names, agg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{NAME}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[write] {path} ({path.stat().st_size:,} bytes)")
    plt.close(fig)

    print(f"\n[panel a] selection rule: every off-diagonal cell with count >= {min_count}; "
          f"{n_cells} such cells need {len(selected)} species (cap {MAX_BLOCK})")
    for t in range(int(off.max()), 0, -1):
        n_sp = len({int(v) for pair in zip(*np.nonzero(off >= t)) for v in pair})
        print(f"    threshold >= {t}: {int((off >= t).sum()):4d} cells, {n_sp:3d} species"
              + ("   <-- chosen" if t == min_count else
                 "   (over cap)" if n_sp > MAX_BLOCK else ""))
    inside = sum(1 for i, j in zip(*np.nonzero(off >= min_count))
                 if i in set(selected) and j in set(selected))
    print(f"[panel a] cells with count >= {min_count} inside the block: {inside}/{n_cells}")
    tier_mass = int(off[off >= min_count].sum())
    print(f"[panel a] error mass carried by those {n_cells} cells: {tier_mass} of "
          f"{int(off.sum())} ({100.0 * tier_mass / off.sum():.1f}%)")
    print(f"[panel a] colour scale spans 0..{int(info_a['vmax'])} misclassified images")
    print(f"[panel a] misclassifications inside the block: {info_a['in_block_errors']:.0f} "
          f"of {int(off.sum())} total ({100.0 * info_a['in_block_errors'] / off.sum():.1f}%)")
    for lbl, label_id, name in zip(labels, selected, species):
        print(f"    {lbl:<18s} {name:<28s} correct {cm[label_id, label_id]}/{IMAGES_PER_CLASS}"
              f"   {taxon(meta, label_id, 'kingdom')} / {taxon(meta, label_id, 'order')}"
              f" / {taxon(meta, label_id, 'family')}")

    print(f"\n[direction] cells with count >= {min_count}: {direction['n_cells']}; "
          f"{len(direction['one_way'])} one-directional (reverse 0), "
          f"{len(direction['reciprocated'])} reciprocated, "
          f"{len(direction['symmetric'])} exactly symmetric")
    for i, j, forward, reverse in direction["one_way"]:
        print(f"    one-way    {meta[i]['species_name']:<28s} -> "
              f"{meta[j]['species_name']:<28s} {forward} (reverse {reverse})")
    for i, j, forward, reverse in direction["reciprocated"]:
        print(f"    both ways  {meta[i]['species_name']:<28s} -> "
              f"{meta[j]['species_name']:<28s} {forward} (reverse {reverse})")

    print("\n[taxonomy of errors]")
    for rank in ("genus", "family"):
        r = tax[rank]
        print(f"    same {rank:<7s} {r['same']:5d} of {tax['total']} errors "
              f"({r['pct']:.2f}%); expected {r['expected']:.2f} if wrong labels were drawn "
              f"uniformly -> {r['enrichment']:.1f}x enrichment "
              f"({r['multi_taxa']} of {r['n_taxa']} {rank}-level taxa hold >1 species)")
    print(f"    cross-family {tax['cross_family']} of {tax['total']} errors "
          f"({tax['cross_family_pct']:.2f}%)")

    print("\n[panel b] row-normalised kingdom matrix (%)")
    for k, row, sup in zip(kingdom_names, info_b["pct"], info_b["support"]):
        cells = "  ".join(f"{v:5.1f}" for v in row)
        print(f"    {k:<10s} {cells}   (support {int(sup)} images)")
    print(f"[panel b] within-kingdom {info_b['within_n']}/{info_b['total']} "
          f"({info_b['within']:.1f}%), uniform chance {chance:.1f}%")


if __name__ == "__main__":
    main()
