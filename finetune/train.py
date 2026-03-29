import argparse
import os
import re
import subprocess
import sys
from typing import Any, Optional

from finetune.data import prepare_dataset

_TRAIN_RE = re.compile(
    r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+\-]+), "
    r"It/sec ([\d.]+), Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB"
)
_VAL_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+), Val took")


def _write_yaml(path: str, cfg: dict[str, Any]) -> None:
    """Write a simple YAML config without requiring pyyaml."""
    lines = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                if isinstance(vv, str):
                    lines.append(f"  {kk}: \"{vv}\"")
                elif isinstance(vv, bool):
                    lines.append(f"  {kk}: {str(vv).lower()}")
                else:
                    lines.append(f"  {kk}: {vv}")
        elif isinstance(v, str):
            lines.append(f"{k}: \"{v}\"")
        elif isinstance(v, bool):
            lines.append(f"{k}: {str(v).lower()}")
        else:
            lines.append(f"{k}: {v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_with_logging(
    cmd: list[str],
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    config: dict,
) -> None:
    """Run mlx-lm as a subprocess, parse its output, and log metrics to wandb."""
    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr so we see all output
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)  # pass through to terminal

        if wandb_project:
            m = _TRAIN_RE.match(line.strip())
            if m:
                import wandb
                wandb.log({
                    "train/loss":        float(m.group(2)),
                    "train/lr":          float(m.group(3)),
                    "train/it_per_sec":  float(m.group(4)),
                    "train/tok_per_sec": float(m.group(5)),
                    "train/tokens":      int(m.group(6)),
                    "train/peak_mem_gb": float(m.group(7)),
                }, step=int(m.group(1)))
                continue

            m = _VAL_RE.match(line.strip())
            if m:
                import wandb
                wandb.log({"val/loss": float(m.group(2))}, step=int(m.group(1)))

    proc.wait()
    if wandb_project:
        import wandb
        wandb.finish()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def build_mlx_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the mlx-lm YAML config dict from parsed args."""
    decay_steps = max(1, args.iters - args.warmup)
    min_lr = args.learning_rate * args.min_lr_ratio

    return {
        "model":             args.model,
        "train":             True,
        "data":              args.data_dir,
        "adapter_path":      args.adapter_path,
        # LoRA
        "lora_layers":       args.lora_layers,
        "lora_parameters": {
            "rank":    args.lora_rank,
            "alpha":   args.lora_alpha,
            "scale":   10.0,
            "dropout": 0.0,
        },
        # Training
        "batch_size":        args.batch_size,
        "iters":             args.iters,
        "learning_rate":     args.learning_rate,
        "lr_schedule": {
            "name":       "cosine_decay",
            # positional args for mlx.optimizers.cosine_decay(init, decay_steps, end)
            "arguments":  [args.learning_rate, decay_steps, min_lr],
            "warmup":     args.warmup,       # linear 0 → peak
            "warmup_init": 0.0,
        },
        "max_seq_length":    args.max_seq_length,
        "grad_checkpoint":   args.grad_checkpoint,
        # Eval / logging
        "steps_per_eval":    args.steps_per_eval,
        "steps_per_report":  args.steps_per_report,
        "save_every":        args.save_every,
        "val_batches":       args.val_batches,
        "seed":              args.seed,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Llama 3.1 8B on M3 Max via mlx-lm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--prepare-data", action="store_true",
        help="Download and preprocess the dataset before training.",
    )
    data.add_argument("--dataset",        default="open-r1/Mixture-of-Thoughts")
    data.add_argument("--dataset-config", default="all",
                      choices=["all", "math", "code", "science"])
    data.add_argument("--data-dir",       default="finetune/data")
    data.add_argument(
        "--sample-fraction", type=float, default=0.1,
        help="Fraction of each sub-config to use (0.1 = 10%%). "
             "Preserves math/code/science proportions when dataset-config=all.",
    )
    data.add_argument(
        "--max-examples", type=int, default=None,
        help="Hard cap after sampling (useful for quick experiments).",
    )

    mdl = parser.add_argument_group("Model")
    mdl.add_argument("--model",        default="meta-llama/Llama-3.1-8B-Instruct")
    mdl.add_argument("--adapter-path", default="finetune/adapters")

    lora = parser.add_argument_group("LoRA")
    lora.add_argument(
        "--lora-layers", type=int, default=16,
        help="Layers receiving LoRA adapters (16/32 for 36/48+ GB unified memory).",
    )
    lora.add_argument("--lora-rank",  type=int,   default=16)
    lora.add_argument("--lora-alpha", type=float, default=32.0)

    trn = parser.add_argument_group("Training")
    trn.add_argument(
        "--epochs", type=int, default=1,
        help="Number of full passes over the training set. "
             "Overrides --iters when set. Recommended: 2–3.",
    )
    trn.add_argument("--iters",          type=int,   default=None,
                     help="Total training iterations. Computed from --epochs if not set.")
    trn.add_argument(
        "--batch-size", type=int, default=2,
        help="2 at 4K ctx for 36 GB; 4 for 48 GB+.",
    )
    trn.add_argument("--learning-rate",  type=float, default=5e-5,
                     help="Peak learning rate (after warmup).")
    trn.add_argument("--min-lr-ratio",  type=float, default=0.1,
                     help="LR at end of cosine decay as a fraction of --learning-rate "
                          "(0.1 → decays to 5e-6 when init=5e-5).")
    trn.add_argument("--warmup",         type=int,   default=100)
    trn.add_argument(
        "--max-seq-length", type=int, default=1024,
        help="4096 for 36 GB; 8192 if you have 48 GB+.",
    )
    trn.add_argument(
        "--grad-checkpoint", action="store_true", default=True,
        help="Gradient checkpointing — saves ~30%% memory at the cost of ~20%% speed.",
    )
    trn.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")

    ckpt = parser.add_argument_group("Logging & checkpointing")
    ckpt.add_argument("--steps-per-eval",   type=int, default=100)
    ckpt.add_argument("--steps-per-report", type=int, default=10)
    ckpt.add_argument("--save-every",       type=int, default=1000)
    ckpt.add_argument("--val-batches",      type=int, default=25)
    ckpt.add_argument("--seed",             type=int, default=42)

    wb = parser.add_argument_group("Weights & Biases  (optional)")
    wb.add_argument("--wandb-project", default="self-correction",
                    help="W&B project name. Omit to disable wandb logging.")
    wb.add_argument("--wandb-run-name", default=None,
                    help="W&B run name. Defaults to wandb auto-generated name.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prepare_data:
        print("[INFO] Preparing dataset...")
        prepare_dataset(
            model_id=args.model,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            output_dir=args.data_dir,
            max_seq_length=args.max_seq_length,
            sample_fraction=args.sample_fraction,
            max_examples=args.max_examples,
        )

    train_file = os.path.join(args.data_dir, "train.jsonl")
    if not os.path.exists(train_file):
        print(
            f"[ERROR] {train_file} not found.\n"
            f"        Run with --prepare-data to download and preprocess the dataset."
        )
        sys.exit(1)

    if args.epochs is not None:
        with open(train_file) as f:
            n_train = sum(1 for line in f if line.strip())
        args.iters = max(1, round(args.epochs * n_train / args.batch_size))
        print(f"[INFO] {n_train} train examples  ×  {args.epochs} epochs  "
              f"÷  batch {args.batch_size}  =  {args.iters} iters")
    elif args.iters is None:
        args.iters = 1000  # default fallback

    os.makedirs(args.adapter_path, exist_ok=True)

    config_path = os.path.join(args.adapter_path, "lora_config.yaml")
    mlx_config = build_mlx_config(args)
    _write_yaml(config_path, mlx_config)
    print(f"[INFO] mlx-lm config → {config_path}")

    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", config_path]
    print(f"[INFO] Launching: {' '.join(cmd)}\n")
    run_with_logging(cmd, args.wandb_project, args.wandb_run_name, mlx_config)


if __name__ == "__main__":
    main()
