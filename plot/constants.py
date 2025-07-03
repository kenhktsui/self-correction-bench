NON_REASONING_MODELS = [
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
    'deepseek-ai/DeepSeek-V3-0324',
    'Qwen/Qwen2.5-72B-Instruct',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'meta-llama/Llama-3.3-70B-Instruct',
    'Qwen/Qwen3-235B-A22B',
    'microsoft/phi-4',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2-7B-Instruct',
    'Qwen/Qwen3-14B',
    'Qwen/Qwen3-30B-A3B',
    'Qwen/Qwen3-32B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'mistralai/Mistral-Small-24B-Instruct-2501',
]



REASONING_MODELS = [
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-14B_thinking",
    "Qwen/Qwen3-32B_thinking",
    "Qwen/Qwen3-30B-A3B_thinking",
    "Qwen/Qwen3-235B-A22B_thinking",
    "deepseek-ai/DeepSeek-R1-0528",
    # "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "microsoft/phi-4-reasoning-plus",
]

# Model name mapping for cleaner display
MODEL_LIST = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Qwen/Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen/QwQ-32B": "QwQ-32B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "Qwen/Qwen3-14B_thinking": "Qwen3-14B (thinking)",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "Qwen/Qwen3-32B_thinking": "Qwen3-32B (thinking)",
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B_thinking": "Qwen3-30B-A3B (thinking)",
    "Qwen/Qwen3-235B-A22B": "Qwen3-235B-A22B",
    "Qwen/Qwen3-235B-A22B_thinking": "Qwen3-235B-A22B (thinking)",
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528": "DeepSeek-R1-0528",
    "mistralai/Mistral-Small-24B-Instruct-2501": "Mistral-Small-24B-Instruct-2501",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "google/gemma-3-27b-it": "gemma-3-27b-it",
    "microsoft/phi-4": "Phi-4",
    "microsoft/phi-4-reasoning-plus": "Phi-4-reasoning-plus",
}