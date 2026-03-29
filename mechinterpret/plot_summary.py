import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# Fixed model order for the grid figure
GRID_MODEL_KEYS = [
    "llama-8b",
    "qwen2.5-7b",
]
GRID_MODEL_LABELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
]


def draw_cosine_ax(
    ax,
    sims: np.ndarray,
    layer_idxs: np.ndarray,
    null_p05: Optional[np.ndarray] = None,
    null_p95: Optional[np.ndarray] = None,
    null_p50: Optional[np.ndarray] = None,
    title: str = "",
    show_legend: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """Draw a cosine similarity curve + optional null bands into a single axes.

    null_p05/null_p50/null_p95 are all optional:
      - all three present → shaded band + p95 dashed + p50 dotted lines
      - only null_p95     → p95 dashed reference line (no fill)
    """
    ax.plot(layer_idxs, sims, marker="o", markersize=3, color="steelblue",
            linewidth=1.5, label="Cosine similarity", zorder=3)
    if null_p95 is not None:
        if null_p05 is not None:
            ax.fill_between(layer_idxs, null_p05, null_p95,
                            color="gray", alpha=0.25, label="Null 5–95 pct")
        ax.plot(layer_idxs, null_p95, color="gray", linestyle="--",
                linewidth=0.8, alpha=0.8, label="Null 95th pct")
        if null_p50 is not None:
            ax.plot(layer_idxs, null_p50, color="gray", linestyle=":",
                    linewidth=0.7, alpha=0.6)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=12)
    if show_xlabel:
        ax.set_xlabel("Layer", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("Cosine similarity", fontsize=8)
    if show_legend:
        ax.legend(fontsize=6)


def plot_cross_dataset_cosine_grid(output_dir: str) -> None:
    """3×N_pairs grid: rows = models (fixed order), cols = dataset pairs.

    Reads pre-computed data from experiment1_summary.json — no recomputation.
    Row order: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Llama-8B.
    """
    plt = _import_plt()

    summary_path = os.path.join(output_dir, "experiment1_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [C-grid] {summary_path} not found — run full analysis first.")
        return

    with open(summary_path) as f:
        summary: dict = json.load(f)

    all_data: List[Optional[Dict[str, dict]]] = [
        summary.get(f"cross_dataset_{mk}") or None
        for mk in GRID_MODEL_KEYS
    ]

    pair_keys: List[str] = next(
        (list(d.keys()) for d in all_data if d), []
    )
    if not pair_keys:
        print("  [C-grid] No cross_dataset_* entries in summary. Skipping.")
        return

    n_rows = len(GRID_MODEL_KEYS)
    n_cols = len(pair_keys)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for row, (label, row_data) in enumerate(zip(GRID_MODEL_LABELS, all_data)):
        for col, pair_key in enumerate(pair_keys):
            ax = axes[row, col]
            if row_data and pair_key in row_data:
                d = row_data[pair_key]
                sims = np.array(d["per_layer"])
                layer_idxs = np.arange(len(sims))
                null_p05 = np.array(d["null_05th_per_layer"]) if "null_05th_per_layer" in d else None
                null_p50 = np.array(d["null_50th_per_layer"]) if "null_50th_per_layer" in d else None
                null_p95 = np.array(d["null_95th_per_layer"]) if "null_95th_per_layer" in d else None
                draw_cosine_ax(
                    ax, sims, layer_idxs,
                    null_p05=null_p05, null_p50=null_p50, null_p95=null_p95,
                    title=pair_key.replace("_vs_", " vs ").upper() if row == 0 else "",
                    show_legend=(row == 0 and col == n_cols - 1),
                    show_xlabel=(row == n_rows - 1),
                    show_ylabel=(col == 0),
                )
            else:
                ax.set_visible(False)

            if col == 0:
                ax.set_ylabel(f"{label}\n\nCosine similarity", fontsize=12)

    fig.tight_layout()
    fname = os.path.join(output_dir, "cross_dataset_cosine_grid.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[C-grid] Grid plot saved: {fname}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot experiment1_summary.json figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="mechinterpret/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_cross_dataset_cosine_grid(args.output_dir)


if __name__ == "__main__":
    main()
