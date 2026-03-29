import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data, PROBLEM_ID_WITH_ERROR 
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from evaluation.evaluate_tool import get_is_correct_answer
from mechinterpret.config import ALL_MODELS


dataset_labels = {
    "gsm8k_sc": "gsm8k_sc (In distribution)",
    "scli5": "scli5 (Cross dataset transfer)",
    "prm800k_sc": "prm800k_sc (Cross dataset transfer)",
}

DATASET_CONFIG: Dict[str, Tuple] = {
    "scli5":      (load_scli5_eval_data,     "llm_evaluation",     "llm_evaluation_error_in_user"),
    "gsm8k_sc":   (load_gsm8k_sc_eval_data,  "llm_evaluation_bca", "llm_evaluation_error_in_user_bca"),
    "prm800k_sc": (load_prm800k_sc_eval_data, "llm_evaluation_bca", "llm_evaluation_error_in_user_bca"),
}

MODEL_ID_TO_KEY = {v: k for k, v in ALL_MODELS.items()}
# Some eval files use "Meta-Llama-..." while ALL_MODELS uses "Llama-..."; add aliases
for _v, _k in list(ALL_MODELS.items()):
    _alt = _k.replace("/Llama-", "/Meta-Llama-")
    if _alt not in MODEL_ID_TO_KEY:
        MODEL_ID_TO_KEY[_alt] = _v


def load_baseline_rates() -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load α=0 internal and external rates from the main eval loaders.

    Returns:
        {(model_short_key, dataset): {"internal": float, "external": float}}
    """
    rates: Dict[Tuple[str, str], Dict[str, float]] = {}

    for dataset, (loader, internal_key, external_key) in DATASET_CONFIG.items():
        try:
            data = loader()
        except FileNotFoundError as e:
            print(f"[WARN] Could not load {dataset} eval data: {e}")
            continue

        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"int_ok": 0, "ext_ok": 0, "total": 0})
        for d in data:
            model_id = d.get("model", "")
            counts[model_id]["total"] += 1
            counts[model_id]["int_ok"] += get_is_correct_answer(d, internal_key)
            counts[model_id]["ext_ok"] += get_is_correct_answer(d, external_key)

        for model_id, c in counts.items():
            # Strip _thinking suffix added by the loaders before lookup
            base_model_id = model_id.replace("_thinking", "")
            model_key = MODEL_ID_TO_KEY.get(base_model_id)
            if model_key is None:
                continue
            n = c["total"]
            rates[(model_key, dataset)] = {
                "internal": c["int_ok"] / n if n else 0.0,
                "external": c["ext_ok"] / n if n else 0.0,
            }

    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="mechinterpret/results")
    parser.add_argument(
        "--alphas", default="-1,-5",
        help="Comma-separated non-zero alpha values to show as columns (α=0 always included).",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Only include steering files whose name contains this suffix (e.g. 'wait').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_alphas = [float(a) for a in args.alphas.split(",")]

    print("[INFO] Loading baseline rates from main eval data...")
    baseline_rates = load_baseline_rates()

    pattern = os.path.join(args.results_dir, "steering_*_internal_*_eval.jsonl")
    files = sorted(glob.glob(pattern))
    if args.suffix:
        files = [f for f in files if args.suffix in os.path.basename(f)]

    rows = []
    for path in files:
        counts = {}

        json_path = path.replace("_eval.jsonl", ".json")
        if not os.path.exists(json_path):
            print(f"[WARN] No corresponding JSON for {path}, skipping.")
            continue
        with open(json_path) as f:
            json_data = json.load(f)
        model_key = json_data.get("model", "?")
        dataset   = json_data.get("dataset", "?")
        if dataset not in dataset_labels:
            continue

        # Initialise totals from the steering JSON (n_total already known per alpha)
        sweep_json = json_data.get("internal_sweep", {})
        n_total = sweep_json[list(sweep_json.keys())[0]]["n_total"]
        print(dataset, n_total)
        if dataset == "gsm8k_sc":
            n_total -= len(PROBLEM_ID_WITH_ERROR)
        
        # Accumulate corrected count from the eval JSONL
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if dataset == "gsm8k_sc" and d.get("id") in  PROBLEM_ID_WITH_ERROR:
                    continue
                alpha = float(d.get("alpha", 0))
                if alpha not in counts:
                    counts[alpha] = {"corrected": 0}
                counts[alpha]["corrected"] += get_is_correct_answer(d, "llm_evaluation")

        bl = baseline_rates.get((model_key, dataset), {})
        row = {
            "model":    model_key,
            "dataset":  dataset_labels[dataset],
            "external": bl.get("external"),
            0.0:        bl.get("internal"),
        }
        for alpha in target_alphas:
            if alpha in counts:
                row[alpha] = counts[alpha]["corrected"] / n_total
            else:
                row[alpha] = None
        rows.append(row)

    if not rows:
        print("No matching files found.")
        return

    col_order = ["external", 0.0] + target_alphas
    df = pd.DataFrame(rows).set_index(["model", "dataset"])
    df = df[col_order]
    df.columns = ["external", "internal"] + [f"α={a}" for a in target_alphas]

    dataset_order = [dataset_labels[d] for d in ["gsm8k_sc", "scli5", "prm800k_sc"]]
    df = df.reset_index()
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    df = df.sort_values(["model", "dataset"]).set_index(["model", "dataset"])

    print(df.to_string(float_format="{:.3f}".format))
    print()
    print(df.to_latex(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
