import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from mechinterpret.config import ALL_MODELS, CORRECTION_MARKERS
from mechinterpret.dataset_utils import load_dataset_pairs
from mechinterpret.model_utils import compute_direction, generate_batch_with_steering, generate_with_steering, load_model_and_tokenizer


def _dir_tag(direction_method: str, top_k: int) -> str:
    """Return a filename tag reflecting the direction method, e.g. '_pca_k3'."""
    if direction_method == "pca" or top_k > 1:
        return f"_pca_k{top_k}"
    return ""


# --------------------------------------------------------------------------------
# evaluate_correction is heuristic only, the answer is then evaluated using Gemini
# --------------------------------------------------------------------------------
def contains_correction(text: str) -> bool:
    """Heuristic: does the generated text contain a self-correction signal?"""
    lower = text.lower()
    return any(marker in lower for marker in CORRECTION_MARKERS)


def _strip_latex(s: str) -> str:
    """Convert LaTeX math notation to a sympy-parseable string.

    Handles common PRM800K patterns: \\frac, \\sqrt, \\pi, \\text, matrices.
    Returns the input unchanged if no LaTeX markup is detected.
    """
    s = str(s).strip()
    # Process inner patterns FIRST so they don't block outer frac matching.
    # \\sqrt{x} and \\sqrt x
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    # \\text{...}
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    # Shorthand \\frac XY before standard form: \\frac59 -> (5)/(9)
    s = re.sub(r"\\frac\s*(\d)\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"((\1)/(\2))", s)
    # Standard \\frac{a}{b} and \\dfrac{a}{b} — repeat for nesting
    for _ in range(4):
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    # Matrix environments — remove entirely (can't compare numerically)
    s = re.sub(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", " ", s, flags=re.DOTALL)
    # \\left, \\right — remove, keep the bracket character
    s = re.sub(r"\\(?:left|right)\s*", "", s)
    # Constants and operators
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # Remove remaining \\commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Braces to parentheses
    s = s.replace("{", "(").replace("}", ")")
    # LaTeX number spacing: 10,\\!080 -> 10080
    s = re.sub(r",\s*\\?\s*!?\s*(?=\d)", "", s)
    # Powers
    s = s.replace("^", "**")
    # Insert explicit * for juxtaposition (with or without space):
    # '8 pi' -> '8*pi', '12pi' -> '12*pi', '3sqrt' -> '3*sqrt'
    s = re.sub(r"(\d)\s*(pi|oo|sqrt)", r"\1*\2", s)
    s = re.sub(r"(pi|oo)\s*(\d)", r"\1*\2", s)
    return s.strip()


def _try_sympy_eval(expr: str) -> Optional[float]:
    """Try to evaluate a math expression to a float using sympy."""
    # Skip pure letter expressions — they could be sympy built-ins (E=Euler)
    # misidentified from \text{(E)} or similar text labels.
    clean = re.sub(r"[\s()]", "", expr)
    if re.match(r"^[A-Za-z]{1,2}$", clean):
        return None
    try:
        import sympy  # type: ignore
        result = sympy.sympify(expr)
        if result.free_symbols:
            return None  # symbolic variables present — can't evaluate
        return float(result.evalf())
    except Exception:
        return None


def _latex_to_plain(s: str) -> str:
    """Convert LaTeX to a human-readable form for substring matching.

    E.g. \\frac{1}{2} -> 1/2,  \\sqrt{5} -> sqrt(5).
    """
    s = str(s).strip()
    for _ in range(4):
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\frac\s*(\d)\s*\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\1/\2", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\\(?:left|right)\s*", "", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", " ").replace("\\times", "x")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}\\]", "", s)
    s = re.sub(r",\s*\\?\s*!?\s*(?=\d)", "", s)  # comma-spacing in numbers
    return re.sub(r"\s+", " ", s).strip()


def extract_final_number(text: str) -> Optional[str]:
    """Extract the concluding numeric answer from generated text.

    Tries (in order):
      1. GSM8K-style '#### N'
      2. 'the answer is N' / 'answer: N' patterns
      3. Last number in the text
    """
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(
        r"(?:the answer is|answer is|answer:|=)\s*(-?[\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _extract_all_numeric_values(text: str) -> list:
    """Extract all numeric values from text, including fraction patterns (a/b).

    Returns a list of floats.
    """
    values = []
    # Plain numbers (with optional thousands commas)
    for m in re.finditer(r"-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?", text):
        try:
            values.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    # Fraction patterns: a/b
    for m in re.finditer(r"(-?\d+)\s*/\s*(-?\d+)", text):
        try:
            num, den = int(m.group(1)), int(m.group(2))
            if den != 0:
                values.append(num / den)
        except ValueError:
            pass
    return values


def answer_is_correct(generated: str, ground_truth: Optional[str]) -> bool:
    """Return True if the generated text reaches the correct answer.

    For numeric ground truths: strip commas and compare as floats.
    For LaTeX ground truths (PRM800K): convert via sympy and compare
      against all numeric values and fraction patterns in generated text.
    Fallback: plain-text substring match after stripping LaTeX markup.
    """
    if not ground_truth:
        return False
    gt = str(ground_truth).strip()

    # ── 1. Numeric comparison (comma-stripped for GSM8K thousands separators) ──
    gt_clean = gt.replace(",", "")
    try:
        gt_float = float(gt_clean)
        extracted = extract_final_number(generated)
        if extracted is not None:
            try:
                if abs(float(extracted) - gt_float) < 1e-6:
                    return True
            except ValueError:
                pass
        # Also try all numbers and fractions in generated text
        for val in _extract_all_numeric_values(generated):
            if abs(val - gt_float) < 1e-4:
                return True
        # Numeric gt matched against numerics; skip LaTeX path
        gt_float_from_latex = None
    except ValueError:
        gt_float = None

        # ── 2. LaTeX → numeric comparison (PRM800K) ──────────────────────────────
        gt_sympy_str = _strip_latex(gt)
        gt_float_from_latex = _try_sympy_eval(gt_sympy_str)
        if gt_float_from_latex is not None:
            for val in _extract_all_numeric_values(generated):
                # Use relative tolerance: models may round irrational answers
                denom = max(abs(gt_float_from_latex), abs(val), 1.0)
                if abs(val - gt_float_from_latex) / denom < 1e-3:
                    return True

    # ── 3. Plain-text substring match after stripping LaTeX markup ───────────
    gt_plain = _latex_to_plain(gt)
    gt_plain_norm = re.sub(r"\s+", "", gt_plain).lower()
    gen_norm = re.sub(r"\s+", "", generated).lower()
    if gt_plain_norm and gt_plain_norm in gen_norm:
        return True

    # ── 4. Verbatim substring fallback ───────────────────────────────────────
    return gt.lower() in generated.lower()


def evaluate_correction(generated: str, ground_truth: Optional[str] = None) -> dict:
    """Evaluate whether the generated text constitutes a self-correction.

    Two complementary signals:
      marker_based   — correction-language heuristic (CORRECTION_MARKERS)
      answer_correct — model reached the ground-truth answer

    self_corrected is True when either signal fires (or only marker_based when
    no ground truth is available).
    """
    marker_based = contains_correction(generated)
    correct_answer = answer_is_correct(generated, ground_truth)
    return {
        "marker_based": marker_based,
        "answer_correct": correct_answer,
        "self_corrected": marker_based or correct_answer,
        "generated": generated,
    }


def run_steering_sweep(
    model,
    tokenizer,
    examples: List[dict],
    direction: np.ndarray,
    alphas: List[float],
    target_layers: Optional[List[int]],
    max_new_tokens: int,
    batch_size: int = 1,
    msg_key: str = "internal_messages",
    add_generation_prompt: bool = False,
    continue_final_message: bool = True,
) -> Dict[float, dict]:
    """For each alpha, generate from all examples and measure correction rate.

    Args:
        msg_key: which message list to use from each example dict.
        add_generation_prompt / continue_final_message: prompt format.
            Internal (default): add_generation_prompt=False, continue_final_message=True
            External:           add_generation_prompt=True,  continue_final_message=False

    Returns:
        {alpha: {"correction_rate": float, "n_corrected": int, "n_total": int}}
    """
    results_by_alpha: Dict[float, dict] = {}

    for alpha in alphas:
        print(f"\n  alpha={alpha:+.2f}")
        n_self_corrected = 0
        n_marker = 0
        n_answer_correct = 0
        per_example = []

        batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
        for batch in tqdm(batches, desc=f"  alpha={alpha:+.2f}"):
            generated_texts = generate_batch_with_steering(
                model=model,
                tokenizer=tokenizer,
                messages_batch=[ex[msg_key] for ex in batch],
                direction=direction,
                alpha=alpha,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                target_layers=target_layers,
                max_new_tokens=max_new_tokens,
            )
            for ex, generated in zip(batch, generated_texts):
                ev = evaluate_correction(generated, ex.get("ground_truth"))
                if ev["self_corrected"]:
                    n_self_corrected += 1
                if ev["marker_based"]:
                    n_marker += 1
                if ev["answer_correct"]:
                    n_answer_correct += 1
                per_example.append({"id": ex["id"], **ev})

        n = len(examples)
        rate = n_self_corrected / n if n else 0.0
        results_by_alpha[alpha] = {
            "correction_rate": rate,
            "n_corrected": n_self_corrected,
            "n_marker_based": n_marker,
            "n_answer_correct": n_answer_correct,
            "n_total": n,
            "per_example": per_example,
        }
        print(
            f"  self_corrected: {rate:.3f} ({n_self_corrected}/{n})"
            f"  [marker={n_marker}  answer={n_answer_correct}]"
        )

    return results_by_alpha


def run_layer_sweep_steering(
    model,
    tokenizer,
    examples: List[dict],
    direction: np.ndarray,
    alpha: float,
    max_new_tokens: int,
    batch_size: int,
) -> dict:
    """Steer one layer at a time with a fixed alpha and record correction rate per layer.

    Runs a true baseline (alpha=0) first, then steers each transformer layer
    individually with the given alpha to identify the most causally effective layer.

    Returns:
        {
            "alpha": float,
            "baseline_rate": float,
            "sweep": {str(layer_idx): {"correction_rate": float, ...}},
            "best_layer": int,
            "best_rate": float,
        }
    """
    n_layers = len(model.model.layers)

    def _generate_and_eval(steer_alpha: float, target_layers) -> dict:
        n_corrected = n_marker = n_answer = 0
        batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
        for batch in batches:
            texts = generate_batch_with_steering(
                model=model,
                tokenizer=tokenizer,
                messages_batch=[ex["internal_messages"] for ex in batch],
                direction=direction,
                alpha=steer_alpha,
                add_generation_prompt=False,
                continue_final_message=True,
                target_layers=target_layers,
                max_new_tokens=max_new_tokens,
            )
            for ex, text in zip(batch, texts):
                ev = evaluate_correction(text, ex.get("ground_truth"))
                if ev["self_corrected"]: n_corrected += 1
                if ev["marker_based"]:   n_marker += 1
                if ev["answer_correct"]: n_answer += 1
        n = len(examples)
        return {
            "correction_rate": n_corrected / n if n else 0.0,
            "n_corrected": n_corrected,
            "n_marker_based": n_marker,
            "n_answer_correct": n_answer,
            "n_total": n,
        }

    # Baseline: alpha=0, no steering
    print(f"\n[INFO] Layer sweep (alpha={alpha:+.2f}) — running baseline (alpha=0)")
    baseline = _generate_and_eval(0.0, None)
    baseline_rate = baseline["correction_rate"]
    print(f"  baseline correction_rate={baseline_rate:.3f}  "
          f"({baseline['n_corrected']}/{baseline['n_total']})")

    # Per-layer sweep with the requested alpha
    sweep: Dict[int, dict] = {}
    print(f"\n[INFO] Layer sweep — steering each of {n_layers} layers  (alpha={alpha:+.2f})")
    print(f"  {'layer':>6}  {'rate':>6}  {'Δ vs baseline':>14}  {'corrected':>10}")
    print(f"  {'-' * 44}")

    for layer_idx in range(n_layers):
        r = _generate_and_eval(alpha, [layer_idx])
        delta = r["correction_rate"] - baseline_rate
        sweep[layer_idx] = r
        print(f"  {layer_idx:>6}  {r['correction_rate']:>6.3f}  {delta:>+14.3f}  "
              f"{r['n_corrected']:>10}/{r['n_total']}")

    best_layer = max(sweep, key=lambda l: sweep[l]["correction_rate"])
    print(f"\n[INFO] Best layer: {best_layer}  "
          f"rate={sweep[best_layer]['correction_rate']:.3f}  "
          f"(baseline={baseline_rate:.3f}  "
          f"Δ={sweep[best_layer]['correction_rate'] - baseline_rate:+.3f})")

    return {
        "alpha": alpha,
        "baseline_rate": baseline_rate,
        "sweep": {str(l): r for l, r in sweep.items()},
        "best_layer": best_layer,
        "best_rate": sweep[best_layer]["correction_rate"],
    }


def parse_args() -> argparse.Namespace:
    # argparse cannot handle '--alphas -3,-2,...' because the value starts with '-'
    # (it tries to parse it as a flag).  Rewrite as '--alphas=-3,-2,...' transparently.
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--alphas":
            val = sys.argv[i + 1]
            if re.fullmatch(r"-?[\d.]+(?:,-?[\d.]+)*", val):
                sys.argv[i : i + 2] = [f"--alphas={val}"]
            break

    parser = argparse.ArgumentParser(description="Experiment 2: Activation steering")
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument(
        "--dataset", required=True, choices=["scli5", "gsm8k_sc", "prm800k_sc"]
    )
    parser.add_argument(
        "--direction_file",
        required=True,
        help="Path to .npz file produced by experiment1_run.py (source of direction)",
    )
    parser.add_argument("--output_dir", default="mechinterpret/results")
    parser.add_argument(
        "--alphas",
        default="-3,-2,-1,0,1,2,3",
        help="Comma-separated scaling coefficients to sweep. "
             "Negative = toward external attribution (expected to boost correction). "
             "When the list starts with a negative value, use --alphas=VALUE syntax "
             "(with =) to avoid argparse treating '-N' as a flag.",
    )
    parser.add_argument(
        "--target_layers",
        default=None,
        help="Comma-separated transformer layer indices to steer. "
             "Default: all layers.",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of examples to process in parallel.")
    parser.add_argument(
        "--sweep_layers", action="store_true",
        help="Sweep each layer individually with --sweep_alpha and report correction "
             "rate per layer. Identifies which layer is most causally effective for steering.",
    )
    parser.add_argument(
        "--sweep_alpha", type=float, default=-1.0,
        help="Alpha to use during --sweep_layers. Default: -1.0.",
    )
    parser.add_argument(
        "--direction_method", default="mean", choices=["mean", "pca"],
        help="How to compute the authorship direction from per-example differences. "
             "'mean' (default): mean difference; 'pca': PCA projection.",
    )
    parser.add_argument(
        "--top_k", type=int, default=1,
        help="Number of PCA components to use (only with --direction_method pca). "
             "The direction is the projection of the mean difference onto the top-k "
             "PCA subspace. Default: 1 (single PC, original behaviour).",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Suffix to add to the output file name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the pre-computed direction
    data = np.load(args.direction_file, allow_pickle=True)
    differences = data["differences"]        # (N, n_layers+1, H)
    direction = compute_direction(differences, method=args.direction_method, top_k=args.top_k)

    dir_label = f"PCA (top_k={args.top_k})" if (args.direction_method == "pca" or args.top_k > 1) else "mean diff"
    print(f"[INFO] Loaded direction from {args.direction_file}  [{dir_label}]")
    print(f"       Shape: {direction.shape}")

    # Load model
    model_id = ALL_MODELS[args.model]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, dtype=dtype)

    # Load target dataset
    print(f"[INFO] Loading dataset: {args.dataset} ({args.max_examples} examples)")
    examples = load_dataset_pairs(args.dataset, max_examples=args.max_examples)

    # ── Layer sweep mode ──────────────────────────────────────────────────────
    if args.sweep_layers:
        sweep_results = run_layer_sweep_steering(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            direction=direction,
            alpha=args.sweep_alpha,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        output = {
            "model": args.model,
            "model_id": model_id,
            "dataset": args.dataset,
            "direction_file": args.direction_file,
            "mode": "sweep_layers",
            **sweep_results,
        }
        out_path = os.path.join(
            args.output_dir,
            f"steering_sweep_{args.model}_{args.dataset}{_dir_tag(args.direction_method, args.top_k)}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[INFO] Sweep results saved to {out_path}")
        return

    # ── Standard alpha sweep mode ─────────────────────────────────────────────
    alphas = [float(a) for a in args.alphas.split(",")]
    target_layers = (
        [int(l) for l in args.target_layers.split(",")]
        if args.target_layers
        else None
    )

    output = {
        "model": args.model,
        "model_id": model_id,
        "dataset": args.dataset,
        "direction_file": args.direction_file,
        "alphas": alphas,
        "target_layers": target_layers,
    }

    print(f"\n[INFO] Internal steering sweep  alphas: {alphas}")
    sweep_results = run_steering_sweep(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        direction=direction,
        alphas=alphas,
        target_layers=target_layers,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        msg_key="internal_messages",
        add_generation_prompt=False,
        continue_final_message=True,
    )
    output["internal_sweep"] = {str(k): v for k, v in sweep_results.items()}
    output["sweep"] = output["internal_sweep"]

    # Save results
    out_path = os.path.join(
        args.output_dir,
        f"steering_{args.model}_{args.dataset}"
        "_internal"
        f"_{args.target_layers}"
        f"{_dir_tag(args.direction_method, args.top_k)}"
        f"{'_' + args.suffix if args.suffix else ''}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Results saved to {out_path}")

    # Print summary table
    print(f"\n[INFO] steering sweep summary:")
    print(f"  {'alpha':>8}  {'self_corrected':>14}  {'marker':>8}  {'answer':>8}  {'n_total':>8}")
    for alpha in alphas:
        r = sweep_results[alpha]
        print(
            f"  {alpha:>8.2f}  {r['correction_rate']:>14.3f}"
            f"  {r['n_marker_based']:>8}  {r['n_answer_correct']:>8}  {r['n_total']:>8}"
        )


if __name__ == "__main__":
    main()
