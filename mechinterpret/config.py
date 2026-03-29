# Device to use for model inference
DEVICE = "mps"

BASE_MODELS = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

ALL_MODELS = BASE_MODELS

HF_DATASETS = {
    "scli5": "kenhktsui/scli5",
    "gsm8k_sc": "kenhktsui/gsm8k_sc",
    "prm800k_sc": "kenhktsui/prm800k_sc",
}

# Existing completion result files (relative to project root), keyed by (model_name, dataset)
# "model_name" here is the string used in the original jsonl files
COMPLETION_RESULT_FILES = {
    "scli5": "scli5_completion_results_llm_eval_gemini2_5_flash_0_0.jsonl",
    "gsm8k_sc": "gsm8k_sc_completion_results_llm_eval_gemini2_5_flash_0_0.jsonl",
    "prm800k_sc": "prm800k_sc_completion_results_llm_eval_gemini2_5_flash_0_0.jsonl",
}

# Correction marker strings to detect in generated text (for experiment 2), heuristic only
CORRECTION_MARKERS = [
    "wait",
    "actually",
    "that's incorrect",
    "that is incorrect",
    "that's wrong",
    "that is wrong",
    "i made a mistake",
    "let me correct",
    "let me reconsider",
    "i was wrong",
    "hmm",
    "hold on",
    "let me re-examine",
]
