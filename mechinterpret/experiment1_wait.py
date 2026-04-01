import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from mechinterpret.config import ALL_MODELS
from mechinterpret.dataset_utils import load_dataset_pairs
from mechinterpret.model_utils import (
    _build_prompt,
    _extract_at_pos,
    _last_error_token_pos,
    load_model_and_tokenizer,
)


@torch.no_grad()
def extract_wait_and_internal(
    model,
    tokenizer,
    internal_messages,
    wait_string: str = " Wait",
):
    """Return (h_wait, h_internal, wait_token_id) for one example.

    h_wait:     (n_layers+1, H)  hidden states at last token of wait_string
    h_internal: (n_layers+1, H)  hidden states at last error token of internal prompt
    wait_token_id: int           token id at the Wait extraction position
    """
    # Internal-error prompt (same as experiment1_run.py)
    internal_prompt = _build_prompt(
        tokenizer, internal_messages,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    pos_internal = _last_error_token_pos(tokenizer, internal_prompt)

    # Append wait_string as a plain continuation
    wait_prompt = internal_prompt + wait_string
    wait_suffix_ids = tokenizer.encode(wait_string, add_special_tokens=False)
    if not wait_suffix_ids:
        raise ValueError(f"wait_string {repr(wait_string)} produced no tokens.")

    # Position of the last token of the wait_string in the full prompt
    base_len = len(tokenizer.encode(internal_prompt))
    pos_wait = base_len + len(wait_suffix_ids) - 1
    wait_token_id = tokenizer.encode(wait_prompt)[pos_wait]

    h_internal = _extract_at_pos(model, tokenizer, internal_prompt, pos=pos_internal)
    h_wait = _extract_at_pos(model, tokenizer, wait_prompt, pos=pos_wait)

    return h_wait, h_internal, wait_token_id


def extract_dataset_wait_activations(
    model,
    tokenizer,
    examples,
    wait_string: str = " Wait",
):
    wait_acts = []
    internal_acts = []
    ids = []
    wait_token_ids = []

    for ex in tqdm(examples):
        try:
            h_wait, h_internal, wait_tok_id = extract_wait_and_internal(
                model, tokenizer,
                ex["internal_messages"],
                wait_string=wait_string,
            )
            wait_acts.append(h_wait)
            internal_acts.append(h_internal)
            ids.append(ex["id"])
            wait_token_ids.append(wait_tok_id)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] Skipped example {ex.get('id')}: {e}")

    return {
        "wait_activations": np.stack(wait_acts, axis=0).astype(np.float32),
        "internal_activations": np.stack(internal_acts, axis=0).astype(np.float32),
        "ids": np.array(ids, dtype=object),
        "wait_token_ids": np.array(wait_token_ids, dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hidden states at the Wait token (Experiment 1 supplement)"
    )
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--dataset", required=True, choices=["scli5", "gsm8k_sc", "prm800k_sc"])
    parser.add_argument("--output_dir", default="mechinterpret/results")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--wait_string", default=" Wait",
        help="String appended to the internal-error prompt to simulate self-correction onset. "
             "Default: ' Wait' (leading space gives the token used mid-sentence).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_id = ALL_MODELS[args.model]
    output_path = os.path.join(
        args.output_dir, f"wait_{args.model}_{args.dataset}.npz"
    )

    if os.path.exists(output_path):
        print(f"[INFO] Output already exists: {output_path}. Delete to re-run.")
        return

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    print(f"[INFO] Loading model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, dtype=dtype_map[args.dtype])

    print(f"[INFO] Loading dataset: {args.dataset}")
    examples = load_dataset_pairs(args.dataset, max_examples=args.max_examples)
    print(f"[INFO] {len(examples)} examples  wait_string={repr(args.wait_string)}")

    print("[INFO] Extracting Wait activations …")
    results = extract_dataset_wait_activations(
        model, tokenizer, examples, wait_string=args.wait_string
    )

    np.savez(
        output_path,
        wait_activations=results["wait_activations"],
        internal_activations=results["internal_activations"],
        ids=results["ids"],
        wait_token_ids=results["wait_token_ids"],
        model_id=np.array(model_id),
        dataset=np.array(args.dataset),
    )
    print(f"[INFO] Saved to {output_path}")
    print(f"[INFO] Shape: {results['wait_activations'].shape}  (N, n_layers+1, H)")

    differences = results["wait_activations"] - results["internal_activations"]
    direction_path = os.path.join(
        args.output_dir, f"wait_direction_{args.model}_{args.dataset}.npz"
    )
    np.savez(
        direction_path,
        differences=differences,
        ids=results["ids"],
        model_id=np.array(model_id),
        dataset=np.array(args.dataset),
    )
    print(f"[INFO] Wait direction file saved to {direction_path}")


if __name__ == "__main__":
    main()
