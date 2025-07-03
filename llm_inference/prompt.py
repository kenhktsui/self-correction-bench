from transformers import AutoTokenizer

# Global tokenizer cache to avoid repeated loading
_tokenizer_cache = {}

def get_prompt_eos_token(
                         model,
                         hf_tokenizer_name,
                         messages,
                         add_generation_prompt,
                         continue_final_message,
                         enable_thinking
                         ):
    # Use cached tokenizer if available, otherwise load and cache it
    if hf_tokenizer_name not in _tokenizer_cache:
        _tokenizer_cache[hf_tokenizer_name] = AutoTokenizer.from_pretrained(hf_tokenizer_name)

    tokenizer = _tokenizer_cache[hf_tokenizer_name]
    
    # for Qwen/Qwen3, we need to handle the case of rasoning and non-reasoning models
    if model.startswith("Qwen/Qwen3") and enable_thinking and continue_final_message:
        assert messages[-1]["role"] == "assistant"
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=enable_thinking
            )
        prompt += "<think>\n" + messages[-1]['content']

    elif model.startswith("Qwen/QwQ") and continue_final_message:
        assert messages[-1]["role"] == "assistant"
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=enable_thinking
        )
        prompt += messages[-1]['content']
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=enable_thinking
        )
    return prompt, tokenizer.eos_token



if __name__ == "__main__":
    import json
    import re
    from tqdm import tqdm


    MODEL_LIST = {
        "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Qwen/Qwen3-235B-A22B": "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B": "Qwen/Qwen3-32B",
        "Qwen/Qwen3-14B": "Qwen/Qwen3-14B",
        "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
        "deepseek-ai/DeepSeek-V3-0324": "deepseek-ai/DeepSeek-V3-0324",
        "mistralai/Mistral-Small-24B-Instruct-2501": "mistralai/Mistral-Small-24B-Instruct-2501",
        "google/gemma-3-27b-it": "google/gemma-3-27b-it",
        "google/gemma-3-12b-it": "google/gemma-3-12b-it",
        "google/gemma-3-4b-it": "google/gemma-3-4b-it",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/phi-4": "microsoft/phi-4",
        "microsoft/phi-4-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
        "deepseek-ai/DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528",
    }


    def create_prompt_unit_test():
        prompt_dict = {}

        with open("scli5_completion_results_api.jsonl") as f:
            for line in f:
                d = json.loads(line)
                if d['model'] not in MODEL_LIST:
                    continue
                if (d["model"], "scli5") not in prompt_dict:
                    prompt_dict[d["model"], "scli5"] = {
                        "model": [
                        (d["messages_error_injection_in_model"], d["prompt_error_injection_in_model"]),
                        (d["messages_error_injection_in_model_wait"], d["prompt_error_injection_in_model_wait"]),
                        (d["messages_error_injection_in_model_cot"], d["prompt_error_injection_in_model_cot"])
                        ],
                        "user": [
                            (d["messages_error_in_user"], d["prompt_error_in_user"]),
                        ]
                    }

        with open("gsm8k_sc_completion_results_api.jsonl") as f:
            for line in f:
                d = json.loads(line)
                if d['model'] not in MODEL_LIST:
                    continue
                if (d["model"], "gsm8k_sc") not in prompt_dict:
                    prompt_dict[d["model"], "gsm8k_sc"] = {
                        "model": [
                            (d["messages_error_injection_in_model_bca"], d["prompt_error_injection_in_model_bca"]),
                            (d["messages_error_injection_in_model_aca"], d["prompt_error_injection_in_model_aca"]),
                            (d["messages_error_injection_in_model_bca_wait"], d["prompt_error_injection_in_model_bca_wait"]),
                            (d["messages_error_injection_in_model_aca_wait"], d["prompt_error_injection_in_model_aca_wait"]),
                        ],
                        "user": [
                            (d["messages_error_in_user_bca"], d["prompt_error_in_user_bca"]),
                            (d["messages_error_in_user_aca"], d["prompt_error_in_user_aca"]),
                        ]
                    }

        with open("prm800k_sc_completion_results_api.jsonl") as f:
            for line in f:
                d = json.loads(line)
                if d['model'] not in MODEL_LIST:
                    continue
                if (d["model"], "prm800k_sc") not in prompt_dict:
                    prompt_dict[d["model"], "prm800k_sc"] = {
                        "model": [
                            (d["messages_error_injection_in_model_bca"], d["prompt_error_injection_in_model_bca"]),
                            (d["messages_error_injection_in_model_aca"], d["prompt_error_injection_in_model_aca"]),
                            (d["messages_error_injection_in_model_bca_wait"], d["prompt_error_injection_in_model_bca_wait"]),
                            (d["messages_error_injection_in_model_aca_wait"], d["prompt_error_injection_in_model_aca_wait"]),
                            (d["messages_error_injection_in_model_bca_cot"], d["prompt_error_injection_in_model_bca_cot"]),
                            (d["messages_error_injection_in_model_aca_cot"], d["prompt_error_injection_in_model_aca_cot"]),
                        ],
                        "user": [
                            (d["messages_error_in_user_bca"], d["prompt_error_in_user_bca"]),
                            (d["messages_error_in_user_aca"], d["prompt_error_in_user_aca"]),
                        ]
                    }   


        test_data = [
            {
                "model": model,
                "dataset": dataset,
                "messages_prompt_model": v["model"],
                "messages_prompt_user": v["user"]
            } for (model, dataset), v in prompt_dict.items()
        ]
        sorted_test_data = sorted(test_data, key=lambda x: x["model"])
        return sorted_test_data


    def remove_yyyy_mm_dd_dates(text):
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        cleaned_text = re.sub(date_pattern, '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()    
        return cleaned_text


    test_list = create_prompt_unit_test()
    for d in tqdm(test_list):
        for message, prompt in d["messages_prompt_model"]:
            try:
                assert remove_yyyy_mm_dd_dates(get_prompt_eos_token(d["model"], MODEL_LIST[d["model"]], message, False, True, False)[0]) == remove_yyyy_mm_dd_dates(prompt)
            except Exception as e:
                print(f"Error in {d['model']} for user prompt")
                print('-' * 100)
                print(get_prompt_eos_token(d["model"], MODEL_LIST[d["model"]], message, False, True, False)[0])
                print('-' * 100)
                print(prompt)
                raise e
        for message, prompt in d["messages_prompt_user"]:
            try:
                assert remove_yyyy_mm_dd_dates(get_prompt_eos_token(d["model"], MODEL_LIST[d["model"]], message, True, False, False)[0]) == remove_yyyy_mm_dd_dates(prompt)
            except Exception as e:
                print(f"Error in {d['model']} for user prompt")
                print('-' * 100)
                print(get_prompt_eos_token(d["model"], MODEL_LIST[d["model"]], message, True, False, False)[0])
                print('-' * 100)
                print(prompt)
                raise e

    # non reasoning mode, user prompt
    assert get_prompt_eos_token("Qwen/Qwen3-32B", "Qwen/Qwen3-32B",
                                [{"role": "user", "content": "Hi"}], True, False, False)[0] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    # reasoning mode, user prompt
    assert get_prompt_eos_token("Qwen/Qwen3-32B", "Qwen/Qwen3-32B",
                                [{"role": "user", "content": "Hi"}], True, False, True)[0] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"

    # non reasoning mode, model completion
    assert get_prompt_eos_token("Qwen/Qwen3-32B", "Qwen/Qwen3-32B",
                                [{"role": "user", "content": "Hi"},
                                {"role": "assistant", "content": "Hey"},
                                ], False, True, False)[0] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHey"

    # reasoning mode, model completion, original chat template
    assert get_prompt_eos_token("Qwen/Qwen3-32B", "Qwen/Qwen3-32B",
                                [{"role": "user", "content": "Hi"},
                                {"role": "assistant", "content": "Hey"},
                                ], False, True, True)[0] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nHey"

    assert get_prompt_eos_token("Qwen/QwQ-32B", "Qwen/QwQ-32B",
                                [{"role": "user", "content": "Hi"}], True, False, False)[0] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    
    # enforce <think> at the beginning https://huggingface.co/Qwen/QwQ-32B
    assert get_prompt_eos_token("Qwen/QwQ-32B", "Qwen/QwQ-32B",
                                [{"role": "user", "content": "Hi"},
                                {"role": "assistant", "content": "Hey"},
                                ], False, True, False)[0] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nHey"
