import argparse
import json
import os
import random
from typing import Optional

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

SUB_CONFIGS = ["math", "code", "science"]


def format_example(example: dict, tokenizer) -> Optional[str]:
    """Apply the Llama 3.1 chat template to a messages-format example."""
    messages = example.get("messages")
    if not messages:
        return None
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return None


def _load_and_sample(
    dataset_name: str,
    config: str,
    sample_fraction: float,
    seed: int,
) -> "datasets.Dataset":
    """Load a single sub-config and take sample_fraction of it."""
    ds = load_dataset(dataset_name, config, split="train")
    if sample_fraction < 1.0:
        n = max(1, int(len(ds) * sample_fraction))
        ds = ds.shuffle(seed=seed).select(range(n))
        print(f"[INFO]   {config:>8}: {len(ds):,} examples ({sample_fraction:.0%})")
    else:
        print(f"[INFO]   {config:>8}: {len(ds):,} examples")
    return ds


def prepare_dataset(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    dataset_name: str = "open-r1/Mixture-of-Thoughts",
    dataset_config: str = "all",
    output_dir: str = "finetune/data",
    max_seq_length: int = 4096,
    sample_fraction: float = 1.0,
    max_examples: Optional[int] = None,
    val_size: float = 0.02,
    seed: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if dataset_config == "all":
        print(f"[INFO] Loading {dataset_name} — all sub-configs"
              + (f" at {sample_fraction:.0%} each" if sample_fraction < 1.0 else ""))
        parts = [
            _load_and_sample(dataset_name, sub, sample_fraction, seed)
            for sub in SUB_CONFIGS
        ]
        ds = concatenate_datasets(parts).shuffle(seed=seed)
    else:
        print(f"[INFO] Loading {dataset_name}/{dataset_config}")
        ds = _load_and_sample(dataset_name, dataset_config, sample_fraction, seed)
    print(f"[INFO] {len(ds):,} total examples after sampling")

    token_budget = int(max_seq_length * 1.15)
    ds = ds.filter(lambda x: x["num_tokens"] <= token_budget, num_proc=4)
    print(f"[INFO] {len(ds):,} examples after num_tokens <= {token_budget} pre-filter")

    if max_examples and max_examples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_examples))
        print(f"[INFO] Capped to {len(ds):,} examples")

    texts: list[str] = []
    skipped = 0
    for ex in tqdm(ds, desc="Formatting", unit="ex"):
        text = format_example(ex, tokenizer)
        if text is None:
            skipped += 1
            continue
        if len(tokenizer.encode(text)) > max_seq_length:
            skipped += 1
            continue
        texts.append(text)

    print(f"[INFO] {len(texts):,} usable examples ({skipped} skipped: bad format or too long)")

    if not texts:
        raise RuntimeError(
            "No usable examples found. Try increasing --max-seq-length or "
            "choosing a different --dataset-config."
        )

    random.seed(seed)
    random.shuffle(texts)
    n_val = max(1, int(len(texts) * val_size))
    splits = {"train": texts[n_val:], "valid": texts[:n_val]}

    for name, split_texts in splits.items():
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for text in split_texts:
                f.write(json.dumps({"text": text}) + "\n")
        print(f"[INFO] {len(split_texts):,} {name} examples → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare open-r1/Mixture-of-Thoughts for LoRA fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",           default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset",         default="open-r1/Mixture-of-Thoughts")
    parser.add_argument("--dataset-config",  default="all",
                        choices=["all", "math", "code", "science"])
    parser.add_argument("--output-dir",      default="finetune/data")
    parser.add_argument("--max-seq-length",   type=int,   default=4096,
                        help="Discard examples longer than this many tokens.")
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="Fraction of each sub-config to use (0.1 = 10%%). "
                             "Applied proportionally so math/code/science ratios are preserved.")
    parser.add_argument("--max-examples",    type=int,   default=None,
                        help="Hard cap after sampling (useful for quick experiments).")
    parser.add_argument("--val-size",        type=float, default=0.02)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    prepare_dataset(
        model_id=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        sample_fraction=args.sample_fraction,
        max_examples=args.max_examples,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
