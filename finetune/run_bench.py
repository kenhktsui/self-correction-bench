import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


MODEL_NAME_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct-ft"
HF_BASE_MODEL      = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH       = "finetune/adapters"
MAX_TOKENS         = 1024
TEMPERATURE        = 0.0


def _build_prompt(tokenizer, messages, add_generation_prompt, continue_final_message):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )


def _generate(model, tokenizer, prompt, max_tokens, temp):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temp), verbose=False)


def process_scli5(d, model, tokenizer, model_name, max_tokens, temp):
    def run(messages, add_gen, cont_final):
        prompt = _build_prompt(tokenizer, messages, add_gen, cont_final)
        return prompt, _generate(model, tokenizer, prompt, max_tokens, temp)

    prompt_m, resp_m = run(d['messages_error_injection_in_model'], False, True)
    prompt_u, resp_u = run(d['messages_error_in_user'], True, False)

    return {
        "model": model_name,
        "id": d['id'],
        "question_type": d['type'],
        "messages_error_injection_in_model": d['messages_error_injection_in_model'],
        "messages_error_in_user": d['messages_error_in_user'],
        "prompt_error_injection_in_model": prompt_m,
        "prompt_error_in_user": prompt_u,
        "response_error_injection_in_model": resp_m,
        "response_error_in_user": resp_u,
        "enable_thinking": False,
        "temperature": temp,
    }


def process_gsm8k_sc(d, model, tokenizer, model_name, max_tokens, temp):
    def run(messages, add_gen, cont_final):
        prompt = _build_prompt(tokenizer, messages, add_gen, cont_final)
        return prompt, _generate(model, tokenizer, prompt, max_tokens, temp)

    prompt_bca, resp_bca = run(d['messages_error_injection_in_model_bca'], False, True)
    prompt_u_bca, resp_u_bca = run(d['messages_error_in_user_bca'], True, False)

    return {
        "model": model_name,
        "id": d['id'],
        "mistake_type": d['type_of_mistake'],
        "messages_error_injection_in_model_bca": d['messages_error_injection_in_model_bca'],
        "messages_error_in_user_bca": d['messages_error_in_user_bca'],
        "prompt_error_injection_in_model_bca": prompt_bca,
        "prompt_error_in_user_bca": prompt_u_bca,
        "response_error_injection_in_model_bca": resp_bca,
        "response_error_in_user_bca": resp_u_bca,
        "enable_thinking": False,
        "temperature": temp,
    }


def process_prm800k_sc(d, model, tokenizer, model_name, max_tokens, temp):
    def run(messages, add_gen, cont_final):
        prompt = _build_prompt(tokenizer, messages, add_gen, cont_final)
        return prompt, _generate(model, tokenizer, prompt, max_tokens, temp)

    prompt_bca, resp_bca = run(d['messages_error_injection_in_model_bca'], False, True)
    prompt_u_bca, resp_u_bca = run(d['messages_error_in_user_bca'], True, False)

    return {
        "model": model_name,
        "question": d['question'],
        "ground_truth_answer": d['ground_truth_answer'],
        "pre_generated_answer": d['pre_generated_answer'],
        "n_reasoning_step": d['n_reasoning_step'],
        "messages_error_injection_in_model_bca": d['messages_error_injection_in_model_bca'],
        "messages_error_in_user_bca": d['messages_error_in_user_bca'],
        "prompt_error_injection_in_model_bca": prompt_bca,
        "prompt_error_in_user_bca": prompt_u_bca,
        "response_error_injection_in_model_bca": resp_bca,
        "response_error_in_user_bca": resp_u_bca,
        "enable_thinking": False,
        "temperature": temp,
    }


DATASET_CONFIG = {
    "scli5": {
        "hf_name": "kenhktsui/scli5",
        "id_key": "id",
        "processor": process_scli5,
    },
    "gsm8k_sc": {
        "hf_name": "kenhktsui/gsm8k_sc",
        "id_key": "id",
        "processor": process_gsm8k_sc,
    },
    "prm800k_sc": {
        "hf_name": "kenhktsui/prm800k_sc",
        "id_key": "question",
        "processor": process_prm800k_sc,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fine-tuned Llama 3.1 8B on self-correction benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))
    parser.add_argument("--model", default=HF_BASE_MODEL,
                        help="HuggingFace base model ID (used to load weights + tokenizer)")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH,
                        help="Path to LoRA adapters from finetune/train.py")
    parser.add_argument("--model-name", default=MODEL_NAME_DEFAULT,
                        help="Model label written to the output JSONL (used by eval scripts)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--output-file", default=None,
                        help="Output JSONL path. Defaults to {dataset}_completion_results_ft.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]

    output_file = args.output_file or f"{args.dataset}_completion_results_ft.jsonl"

    # Resume: collect already-processed IDs
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_ids.add(str(d[cfg["id_key"]]))
                except json.JSONDecodeError:
                    pass
    print(f"[INFO] {len(done_ids)} examples already processed — will skip them.")

    print(f"[INFO] Loading {args.model} with adapters from {args.adapter_path} …")
    from mlx_lm import load
    model, tokenizer_mlx = load(args.model, adapter_path=args.adapter_path)

    tokenizer_hf = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset(cfg["hf_name"], split="test")
    print(f"[INFO] Dataset: {cfg['hf_name']}  ({len(dataset)} examples)")

    temp = 0.0 if args.temperature == 0.0 else args.temperature

    with open(output_file, "a") as f:
        for d in tqdm(dataset, desc=args.dataset):
            ex_id = str(d[cfg["id_key"]])
            if ex_id in done_ids:
                continue
            result = cfg["processor"](d, model, tokenizer_mlx, args.model_name,
                                      args.max_tokens, temp)
            f.write(json.dumps(result) + "\n")
            f.flush()
            done_ids.add(ex_id)

    print(f"[INFO] Done. Results written to {output_file}")


if __name__ == "__main__":
    main()
