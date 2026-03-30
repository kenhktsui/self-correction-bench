import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate_tool import get_is_correct_answer
from evaluation.evaluate_gsm8k_sc import PROBLEM_ID_WITH_ERROR


DATASET_KEYS = {
    "scli5":      ("llm_evaluation",     "llm_evaluation_error_in_user"),
    "gsm8k_sc":   ("llm_evaluation_bca", "llm_evaluation_error_in_user_bca"),
    "prm800k_sc": ("llm_evaluation_bca", "llm_evaluation_error_in_user_bca"),
}

DEFAULT_INPUTS = [
    "scli5_completion_results_ft_llm_eval_gemini2_5_flash_0_0.jsonl",
    "gsm8k_sc_completion_results_ft_llm_eval_gemini2_5_flash_0_0.jsonl",
    "prm800k_sc_completion_results_ft_llm_eval_gemini2_5_flash_0_0.jsonl",
]


def detect_dataset(path: str) -> str:
    basename = os.path.basename(path)
    for ds in DATASET_KEYS:
        if basename.startswith(ds):
            return ds
    raise ValueError(f"Cannot detect dataset from filename: {basename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise fine-tuned model LLM-eval results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs", nargs="+", default=DEFAULT_INPUTS,
        help="Paths to LLM-evaluated JSONL files (one or more datasets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_rows = []

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[WARN] File not found, skipping: {path}")
            continue

        try:
            dataset = detect_dataset(path)
        except ValueError as e:
            print(f"[WARN] {e} — skipping.")
            continue

        internal_key, external_key = DATASET_KEYS[dataset]
        counts = defaultdict(lambda: {"internal": 0, "external": 0, "total": 0})

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if dataset == "gsm8k" and d['id'] in PROBLEM_ID_WITH_ERROR:
                    continue

                model = d.get("model", "unknown")
                counts[model]["total"] += 1
                counts[model]["external"] += get_is_correct_answer(d, external_key)
                counts[model]["internal"] += get_is_correct_answer(d, internal_key)

        for model, c in sorted(counts.items()):
            n = c["total"]
            all_rows.append({
                "dataset": dataset,
                "model": model,
                "external": c["external"] / n if n else 0.0,
                "internal": c["internal"] / n if n else 0.0,
            })

    if not all_rows:
        print("No data found.")
        return

    df = pd.DataFrame(all_rows).set_index(["dataset", "model"])

    df['Blind Spot'] = (df['external'] - df['internal'])/df['external']
    df['Blind Spot (Base Model)'] = [0.828947,  0.970794,  0.888889]
    print(df['Blind Spot'].mean()/df['Blind Spot (Base Model)'].mean()-1)

    print()
    print(df.to_string(float_format="{:.3f}".format))
    print()
    print(df.to_latex(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
