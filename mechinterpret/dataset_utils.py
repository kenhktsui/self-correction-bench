import json
import os
from typing import Dict, List, Optional, Tuple

from mechinterpret.config import HF_DATASETS, COMPLETION_RESULT_FILES


def _load_hf_dataset(hf_id: str, split: str = "test"):
    """Lazy-import datasets and load a HuggingFace dataset."""
    from datasets import load_dataset
    return load_dataset(hf_id, split=split)


def _parse_gsm8k_answer(raw: Optional[str]) -> Optional[str]:
    """Extract the numeric answer after '#### ' in a GSM8K answer string."""
    if raw is None:
        return None
    import re
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", raw)
    if m:
        return m.group(1).replace(",", "")
    return None


def load_scli5_pairs(max_examples: Optional[int] = None) -> List[dict]:
    """Load SCLI5 examples as internal/external message pairs."""
    ds = _load_hf_dataset(HF_DATASETS["scli5"])
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    pairs = []
    for d in ds:
        pairs.append({
            "id": d["id"],
            "dataset": "scli5",
            "internal_messages": d["messages_error_injection_in_model"],
            "external_messages": d["messages_error_in_user"],
            "ground_truth": d.get("correct_answer"),
            "meta": {"type": d.get("type", "")},
        })
    return pairs


def load_gsm8k_sc_pairs(
    variant: str = "bca",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load GSM8K-SC examples.

    Args:
        variant: 'bca' (before concluding answer) or 'aca' (after concluding answer).
    """
    ds = _load_hf_dataset(HF_DATASETS["gsm8k_sc"])
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    pairs = []
    for i, d in enumerate(ds):
        internal_key = f"messages_error_injection_in_model_{variant}"
        external_key = f"messages_error_in_user_{variant}"
        pairs.append({
            "id": d.get("id", i),
            "dataset": "gsm8k_sc",
            "internal_messages": d[internal_key],
            "external_messages": d[external_key],
            "ground_truth": _parse_gsm8k_answer(d.get("answer")),
            "meta": {"variant": variant},
        })
    return pairs


def load_prm800k_sc_pairs(
    variant: str = "bca",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load PRM800K-SC examples.

    Args:
        variant: 'bca' or 'aca'.
    """
    ds = _load_hf_dataset(HF_DATASETS["prm800k_sc"])
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    pairs = []
    for i, d in enumerate(ds):
        internal_key = f"messages_error_injection_in_model_{variant}"
        external_key = f"messages_error_in_user_{variant}"
        pairs.append({
            "id": d.get("id", i),
            "dataset": "prm800k_sc",
            "internal_messages": d[internal_key],
            "external_messages": d[external_key],
            "ground_truth": d.get("ground_truth_answer"),
            "meta": {"variant": variant},
        })
    return pairs


def load_dataset_pairs(
    dataset_name: str,
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Unified loader for all three datasets."""
    loaders = {
        "scli5": load_scli5_pairs,
        "gsm8k_sc": load_gsm8k_sc_pairs,
        "prm800k_sc": load_prm800k_sc_pairs,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(loaders)}")
    return loaders[dataset_name](max_examples=max_examples)


def _build_question_to_id(dataset_name: str) -> Optional[Dict[str, int]]:
    """Build a question-text → HF dataset index map for datasets that lack an id field.

    Used to correctly match JSONL records (which have no id) back to the sequential
    indices stored in NPZ files.  Returns None for datasets that have explicit ids.
    """
    if dataset_name not in ("prm800k_sc",):
        return None
    ds = _load_hf_dataset(HF_DATASETS[dataset_name])
    return {
        (example.get("question") or example.get("problem", "")): i
        for i, example in enumerate(ds)
    }


def load_correction_labels(
    dataset_name: str,
    model_name: str,
    project_root: str = ".",
) -> Dict[int, bool]:
    """Load correction-success labels from existing LLM evaluation files.

    Returns a dict mapping example id → True if the model corrected the error.
    Correction is defined as the LLM evaluator marking is_correct_answer=True
    for the internal-error (model-authored) condition.
    """
    result_file = COMPLETION_RESULT_FILES.get(dataset_name)
    if result_file is None:
        return {}

    path = os.path.join(project_root, result_file)
    if not os.path.exists(path):
        return {}

    # For datasets without explicit ids, map question text → HF dataset index
    # so that JSONL records (which may be in arbitrary order) get the correct id.
    question_to_id = _build_question_to_id(dataset_name)

    labels: Dict[int, bool] = {}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("model") != model_name:
                continue
            # Resolve example id
            example_id = d.get("id")
            if example_id is None:
                if question_to_id is not None:
                    example_id = question_to_id.get(d.get("question", ""))
                    if example_id is None:
                        continue  # question not found in HF dataset — skip
                else:
                    continue  # no id and no question map — skip
            # Eval result: scli5 uses "llm_evaluation"; gsm8k_sc / prm800k_sc
            # use "llm_evaluation_bca" (internal-error, BCA variant).
            eval_result = d.get("llm_evaluation") or d.get("llm_evaluation_bca")
            if eval_result is None:
                continue
            parsed = eval_result.get("parsed") if isinstance(eval_result, dict) else None
            if parsed is None:
                continue
            labels[example_id] = bool(parsed.get("is_correct_answer", False))
    return labels
