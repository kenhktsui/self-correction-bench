import argparse
import os

import numpy as np
import torch

from mechinterpret.config import ALL_MODELS
from mechinterpret.dataset_utils import load_dataset_pairs
from mechinterpret.model_utils import extract_dataset_differences, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract activation differences (Experiment 1)")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(ALL_MODELS.keys()),
        help="Model key (e.g. llama-8b, deepseek-r1-distill-8b)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["scli5", "gsm8k_sc", "prm800k_sc"],
        help="Dataset name",
    )
    parser.add_argument(
        "--output_dir",
        default="mechinterpret/results",
        help="Directory to save .npz files",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of examples (default: use all)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_id = ALL_MODELS[args.model]
    output_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}.npz")

    if os.path.exists(output_path):
        print(f"[INFO] Output already exists: {output_path}. Delete to re-run.")
        return

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, dtype=dtype)

    print(f"[INFO] Loading dataset: {args.dataset}")
    examples = load_dataset_pairs(args.dataset, max_examples=args.max_examples)
    print(f"[INFO] Number of examples: {len(examples)}")

    print("[INFO] Extracting activation differences...")
    results = extract_dataset_differences(model, tokenizer, examples)

    np.savez(
        output_path,
        differences=results["differences"].astype(np.float32),
        ids=np.array(results["ids"]),
        model_id=np.array(model_id),
        dataset=np.array(args.dataset),
    )
    print(f"[INFO] Saved to {output_path}")
    print(f"[INFO] Shape: {results['differences'].shape}")


if __name__ == "__main__":
    main()
