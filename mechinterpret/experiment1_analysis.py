import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from mechinterpret.plot_summary import draw_cosine_ax


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "differences": data["differences"],    # (N, n_layers+1, H)
        "ids": data["ids"].tolist(),
        "model_id": str(data["model_id"]),
        "dataset": str(data["dataset"]),
    }


def layer_wise_cosine(
    dir_a: np.ndarray,
    dir_b: np.ndarray,
) -> np.ndarray:
    """Cosine similarity at each layer between two mean directions.

    Args:
        dir_a, dir_b: (n_layers+1, H) arrays.
    Returns:
        (n_layers+1,) array of cosine similarities.
    """
    eps = 1e-8
    dot = (dir_a * dir_b).sum(axis=1)
    norm_a = np.linalg.norm(dir_a, axis=1)
    norm_b = np.linalg.norm(dir_b, axis=1)
    return dot / (norm_a * norm_b + eps)


def permutation_null_distribution(
    diffs_a: np.ndarray,
    dir_b: np.ndarray,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build null distribution of layer-wise cosine similarities by label shuffling.

    For each permutation the internal/external labels are shuffled by randomly
    flipping the sign of each example's difference vector in ``diffs_a``.  This
    preserves the marginal distribution of activation magnitudes while destroying
    the directional signal.  The mean direction is then recomputed and its cosine
    similarity with ``dir_b`` recorded.

    Args:
        diffs_a:        (N, n_layers, H) per-example difference arrays.
        dir_b:          (n_layers, H) real mean direction for the other dataset.
        n_permutations: Number of random permutations.
        rng:            Optional numpy Generator for reproducibility.

    Returns:
        null_sims: (n_permutations, n_layers) array of null cosine similarities.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, n_layers, H = diffs_a.shape
    null_sims = np.empty((n_permutations, n_layers), dtype=np.float32)

    norm_b = np.linalg.norm(dir_b, axis=1)  # (n_layers,)

    for p in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=diffs_a.dtype), size=(N, 1, 1))
        shuffled_mean = (diffs_a * signs).mean(axis=0)  # (n_layers, H)
        dot = (shuffled_mean * dir_b).sum(axis=1)
        norm_a = np.linalg.norm(shuffled_mean, axis=1)
        null_sims[p] = dot / (norm_a * norm_b + 1e-8)

    return null_sims


def _load_diffs_for_mode(
    model_key: str,
    dataset: str,
    results_dir: str,
    mode: str,
) -> Optional[np.ndarray]:
    """Load the per-example differences array for the given mode.

    mode='authorship': h_internal − h_external  from {model_key}_{dataset}.npz
    mode='wait':       h_wait − h_internal       from wait_{model_key}_{dataset}.npz
    Returns (N, n_layers+1, H) or None if the file is missing.
    """
    if mode == "authorship":
        path = os.path.join(results_dir, f"{model_key}_{dataset}.npz")
        if not os.path.exists(path):
            return None
        return load_npz(path)["differences"]
    elif mode == "wait":
        path = os.path.join(results_dir, f"wait_{model_key}_{dataset}.npz")
        if not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=True)
        h_wait     = data["wait_activations"].astype(np.float32)      # (N, L, H)
        h_internal = data["internal_activations"].astype(np.float32)  # (N, L, H)
        return h_wait - h_internal                                     # (N, L, H)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'authorship' or 'wait'.")


def _compute_cross_cosine_data(
    model_key: str,
    results_dir: str,
    datasets: List[str],
    mode: str,
    direction_method: str,
    top_k: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> Dict[str, dict]:
    """Compute per-pair cosine similarity data (no plotting).

    Returns a dict keyed by 'ds_a_vs_ds_b' with fields:
        sims, layer_idxs, null_p05, null_p95, null_p50, pvals  (last four may be None)
    and summary scalars: mean_cosine, max_cosine, layer_of_max, ...
    """
    from mechinterpret.model_utils import compute_direction as _compute_dir

    dirs: Dict[str, np.ndarray] = {}
    raw_diffs: Dict[str, np.ndarray] = {}
    for ds in datasets:
        diffs = _load_diffs_for_mode(model_key, ds, results_dir, mode)
        if diffs is None:
            print(f"Missing file for {ds} (mode={mode}) — skipping.")
            continue
        raw_diffs[ds] = diffs
        dirs[ds] = _compute_dir(diffs, method=direction_method, top_k=top_k)

    if len(dirs) < 2:
        print(f"Need at least 2 datasets. Skipping.")
        return {}

    results: Dict[str, dict] = {}
    pairs = [(a, b) for i, a in enumerate(dirs) for b in list(dirs.keys())[i + 1:]]

    for ds_a, ds_b in pairs:
        sims = layer_wise_cosine(dirs[ds_a], dirs[ds_b])
        layer_idxs = np.arange(len(sims))

        if n_permutations > 0:
            print(f"{ds_a} vs {ds_b}: running {n_permutations} permutations …")
            null = permutation_null_distribution(
                raw_diffs[ds_a], dirs[ds_b],
                n_permutations=n_permutations, rng=rng,
            )
            null_p05 = np.percentile(null, 5, axis=0)
            null_p95 = np.percentile(null, 95, axis=0)
            null_p50 = np.median(null, axis=0)
            pvals = (null >= sims[np.newaxis, :]).mean(axis=0)
        else:
            null_p05 = null_p95 = null_p50 = pvals = None

        peak_layer = int(np.argmax(sims))
        key = f"{ds_a}_vs_{ds_b}"
        entry: dict = {
            "sims": sims,
            "layer_idxs": layer_idxs,
            "null_p05": null_p05,
            "null_p95": null_p95,
            "null_p50": null_p50,
            "pvals": pvals,
            "mean_cosine": float(sims.mean()),
            "max_cosine": float(sims.max()),
            "layer_of_max": peak_layer,
            "per_layer": sims.tolist(),
        }
        if pvals is not None:
            p_at_peak = float(pvals[peak_layer])
            sig_layers = [int(l) for l in np.where(pvals < 0.05)[0]]
            entry["permutation_n"] = n_permutations
            entry["null_05th_per_layer"] = null_p05.tolist()
            entry["null_50th_per_layer"] = null_p50.tolist()
            entry["null_95th_per_layer"] = null_p95.tolist()
            entry["pvalue_per_layer"] = pvals.tolist()
            entry["pvalue_at_peak_layer"] = p_at_peak
            entry["significant_layers_p05"] = sig_layers
            sig_str = f"  p={p_at_peak:.4f} at peak  sig_layers={sig_layers}"
        else:
            sig_str = ""

        results[key] = entry
        print(
            f"{key}: mean={sims.mean():.3f}  max={sims.max():.3f}"
            f"  (layer {peak_layer}){sig_str}"
        )

    return results


def analysis_cross_dataset_cosine(
    model_key: str,
    results_dir: str,
    output_dir: str,
    datasets: List[str] = ("scli5", "gsm8k_sc", "prm800k_sc"),
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
    mode: str = "authorship",
    direction_method: str = "mean",
    top_k: int = 1,
) -> dict:
    """Compute and plot cosine similarity of the direction across datasets (single model)."""
    plt = _import_plt()
    if rng is None:
        rng = np.random.default_rng()

    data = _compute_cross_cosine_data(
        model_key, results_dir, datasets, mode,
        direction_method, top_k, n_permutations, rng,
    )
    if not data:
        return {}

    # Strip numpy arrays before returning summary
    return {k: {sk: sv for sk, sv in v.items()
                if sk not in ("sims", "layer_idxs", "null_p05", "null_p95", "null_p50", "pvals")}
            for k, v in data.items()}


def analysis_effective_alpha(
    model_key: str,
    results_dir: str,
    datasets: List[str] = ("scli5", "gsm8k_sc", "prm800k_sc"),
    direction_method: str = "mean",
    top_k: int = 1,
    target_layer: Optional[int] = None,
) -> dict:
    """Project the authorship direction onto the wait direction at target_layer.

        effective_alpha = dot(auth_dir[l], wait_dir[l]) / ||wait_dir[l]||²

    Reports per-dataset values and the macro average across datasets.
    No plots are produced.

    Files required:
        {results_dir}/{model_key}_{dataset}.npz        — authorship diffs
        {results_dir}/wait_{model_key}_{dataset}.npz   — wait activations
    """
    from mechinterpret.model_utils import compute_direction as _compute_dir

    results = {}

    for ds in datasets:
        auth_path = os.path.join(results_dir, f"{model_key}_{ds}.npz")
        wait_path = os.path.join(results_dir, f"wait_{model_key}_{ds}.npz")

        if not os.path.exists(auth_path):
            print(f"Missing authorship file for {ds} — skipping.")
            continue
        if not os.path.exists(wait_path):
            print(f"Missing wait file for {ds} — skipping.")
            continue

        auth_diffs = load_npz(auth_path)["differences"]                  # (N, L, H)
        wait_data  = np.load(wait_path, allow_pickle=True)
        h_wait     = wait_data["wait_activations"].astype(np.float32)    # (N, L, H)
        h_internal = wait_data["internal_activations"].astype(np.float32)
        wait_diffs = h_wait - h_internal                                  # (N, L, H)

        auth_dir = _compute_dir(auth_diffs, method=direction_method, top_k=top_k)  # (L, H)
        wait_dir = _compute_dir(wait_diffs, method=direction_method, top_k=top_k)  # (L, H)

        n_layers = auth_dir.shape[0]
        if target_layer is None or not (0 <= target_layer < n_layers):
            print(f"{ds}: target_layer {target_layer} out of range (n_layers={n_layers}) — skipping.")
            continue

        dot   = float(np.dot(auth_dir[target_layer], wait_dir[target_layer]))
        w_sq  = float(np.dot(wait_dir[target_layer], wait_dir[target_layer]))
        eff_alpha = dot / (w_sq + 1e-8)

        print(f"{model_key}  {ds}  layer {target_layer}: eff_alpha = {eff_alpha:.4f}")
        results[ds] = {"effective_alpha_at_target": eff_alpha, "target_layer": target_layer}

    if results:
        macro_avg = float(np.mean([v["effective_alpha_at_target"] for v in results.values()]))
        results["macro_avg"] = macro_avg
        print(f"{model_key}  macro avg across {list(results.keys())[:-1]}: eff_alpha = {macro_avg:.4f}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 analysis")
    parser.add_argument("--results_dir", default="mechinterpret/results")
    parser.add_argument("--output_dir", default="mechinterpret/plots")
    parser.add_argument(
        "--model_key", default=None,
        help="Restrict analysis to this model key (e.g. llama-8b). "
             "If omitted, all found .npz files are analysed.",
    )
    parser.add_argument(
        "--n_permutations", type=int, default=1000,
        help="Number of label-shuffle permutations for the cross-dataset null "
             "distribution (analysis C). Set to 0 to skip.",
    )
    parser.add_argument(
        "--permutation_seed", type=int, default=42,
        help="Random seed for permutation test reproducibility.",
    )
    parser.add_argument(
        "--direction_method", default="mean", choices=["mean", "pca"],
    )
    parser.add_argument(
        "--top_k", type=int, default=1,
        help="Number of PCA components. top_k>1 forces PCA.",
    )
    parser.add_argument(
        "--target_layer", type=int, default=None,
        help="Layer index to highlight in analysis F (effective alpha). E.g. --target_layer 13.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    npz_files = [
        f for f in os.listdir(args.results_dir) if f.endswith(".npz")
        if not os.path.basename(f).startswith("wait_")
    ]
    if args.model_key:
        npz_files = [f for f in npz_files if f.startswith(args.model_key + "_")]

    summary_path = os.path.join(args.output_dir, "experiment1_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary: dict = json.load(f)
    else:
        summary: dict = {}

    from mechinterpret.config import ALL_MODELS
    model_keys_found = set()
    for fname in npz_files:
        for mk in ALL_MODELS:
            if fname.startswith(mk + "_"):
                model_keys_found.add(mk)
    if args.model_key:
        model_keys_found = {args.model_key} & model_keys_found

    rng = np.random.default_rng(args.permutation_seed)

    for mk in sorted(model_keys_found):
        result = analysis_cross_dataset_cosine(
            mk, args.results_dir, args.output_dir,
            n_permutations=args.n_permutations,
            rng=rng,
            mode="authorship",
            direction_method=args.direction_method,
            top_k=args.top_k,
        )
        summary[f"cross_dataset_{mk}"] = result

        datasets_for_mk = [
            fname.replace(f"{mk}_", "").replace(".npz", "")
            for fname in npz_files
            if fname.startswith(mk + "_")
        ]
        eff_result = analysis_effective_alpha(
            mk, args.results_dir,
            datasets=datasets_for_mk,
            direction_method=args.direction_method,
            top_k=args.top_k,
            target_layer=args.target_layer,
        )
        summary[f"effective_alpha_{mk}"] = eff_result

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
