import os
import json
import random
import concurrent.futures
import threading
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from llm_inference.prompt import get_prompt_eos_token


TEMPERATURE = 0.6
FILE_NAME = "gsm8k_sc_completion_results_api.jsonl" if TEMPERATURE == 0.0 else f"gsm8k_sc_completion_results_api_{str(TEMPERATURE).replace('.', '_')}.jsonl"

if os.path.exists(FILE_NAME):
    print(f"File {FILE_NAME} already exists.")
else:
    with open(FILE_NAME, "w") as f:
        pass


MODEL_LIST = {
    "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Qwen/Qwen3-235B-A22B": "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen/Qwen3-235B-A22B_thinking": "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B_thinking": "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B_thinking": "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B_thinking": "Qwen/Qwen3-14B",
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
    "Qwen/QwQ-32B": "Qwen/QwQ-32B",
}


dataset = load_dataset("kenhktsui/gsm8k_sc", split="test")


# Create an OpenAI client with your deepinfra token and endpoint
deepinfra_client = OpenAI(api_key=os.environ.get("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai")


id_set = set()
with open(FILE_NAME, "r") as f:
    for line in f:
        d = json.loads(line)
        id_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))

print(f"Number of results that have been processed: {len(id_set)}")

def openai_client_process_item(client, item_tuple):
    model, hf_tokenizer_name, d = item_tuple
    key = str(d['id']) + "_" + (model[:-9] if model.endswith("_thinking") else model) + "_" + str(True if model.endswith("_thinking") else False)
    if key in id_set:
        return None

    # handling case of hybrid model
    if model.endswith("_thinking"):
        model = model[:-9]
        enable_thinking = True
    else:
        enable_thinking = False

    def get_prompt_and_response(messages, add_generation_prompt, continue_final_message, enable_thinking):
        prompt, eos_token = get_prompt_eos_token(model,
                                                 hf_tokenizer_name,
                                                 messages,
                                                 add_generation_prompt,
                                                 continue_final_message,
                                                 enable_thinking
                                                 )
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1024,
            temperature=TEMPERATURE,
            stop=[eos_token]
        )
        return prompt, response


    prompt_error_injection_in_model_bca, response_error_injection_in_model_bca = get_prompt_and_response(d['messages_error_injection_in_model_bca'],
                                                                                                         False,
                                                                                                         True,
                                                                                                         enable_thinking)
    prompt_error_injection_in_model_aca, response_error_injection_in_model_aca = get_prompt_and_response(d['messages_error_injection_in_model_aca'],
                                                                                                         False,
                                                                                                         True,
                                                                                                         enable_thinking)

    prompt_error_in_user_bca, response_error_in_user_bca = get_prompt_and_response(d['messages_error_in_user_bca'],
                                                                             True,
                                                                             False,
                                                                             enable_thinking)
    prompt_error_in_user_aca, response_error_in_user_aca = get_prompt_and_response(d['messages_error_in_user_aca'],
                                                                             True,
                                                                             False,
                                                                             enable_thinking)
    
    messages_error_injection_in_model_bca_wait = deepcopy(d['messages_error_injection_in_model_bca'])
    messages_error_injection_in_model_aca_wait = deepcopy(d['messages_error_injection_in_model_aca'])
    assert messages_error_injection_in_model_bca_wait[-1]['role'] == "assistant"
    assert messages_error_injection_in_model_aca_wait[-1]['role'] == "assistant"
    messages_error_injection_in_model_bca_wait[-1]['content'] += " Wait"
    messages_error_injection_in_model_aca_wait[-1]['content'] += " Wait"
    prompt_error_injection_in_model_bca_wait, response_error_injection_in_model_bca_wait = get_prompt_and_response(messages_error_injection_in_model_bca_wait,
                                                                                                                   False,
                                                                                                                   True,
                                                                                                                   enable_thinking)
    prompt_error_injection_in_model_aca_wait, response_error_injection_in_model_aca_wait = get_prompt_and_response(messages_error_injection_in_model_aca_wait,
                                                                                                                   False,
                                                                                                                   True,
                                                                                                                   enable_thinking)
    return {
        "model": model,
        "id": d['id'],
        "mistake_type": d['type_of_mistake'],
        "messages_error_injection_in_model_bca": d['messages_error_injection_in_model_bca'],
        "messages_error_injection_in_model_aca": d['messages_error_injection_in_model_aca'],
        "messages_error_in_user_bca": d['messages_error_in_user_bca'],
        "messages_error_in_user_aca": d['messages_error_in_user_aca'],
        "messages_error_injection_in_model_bca_wait": messages_error_injection_in_model_bca_wait,
        "messages_error_injection_in_model_aca_wait": messages_error_injection_in_model_aca_wait,
        "prompt_error_injection_in_model_bca": prompt_error_injection_in_model_bca,
        "prompt_error_injection_in_model_aca": prompt_error_injection_in_model_aca,
        "prompt_error_in_user_bca": prompt_error_in_user_bca,
        "prompt_error_in_user_aca": prompt_error_in_user_aca,
        "prompt_error_injection_in_model_bca_wait": prompt_error_injection_in_model_bca_wait,
        "prompt_error_injection_in_model_aca_wait": prompt_error_injection_in_model_aca_wait,
        "response_error_injection_in_model_bca": response_error_injection_in_model_bca.choices[0].text,
        "response_error_injection_in_model_aca": response_error_injection_in_model_aca.choices[0].text,
        "response_error_in_user_bca": response_error_in_user_bca.choices[0].text,
        "response_error_in_user_aca": response_error_in_user_aca.choices[0].text,
        "response_error_injection_in_model_bca_wait": response_error_injection_in_model_bca_wait.choices[0].text,
        "response_error_injection_in_model_aca_wait": response_error_injection_in_model_aca_wait.choices[0].text,
        "enable_thinking": enable_thinking,
        "temperature": TEMPERATURE
    }


deepinfra_client_process_item = partial(openai_client_process_item, deepinfra_client)


file_lock = threading.Lock()

with open(FILE_NAME, "a") as f:
    items_to_process = []
    for model, hf_tokenizer_name in MODEL_LIST.items():
        for d in dataset:
            items_to_process.append((model, hf_tokenizer_name, d))

    # shuffle the model to reduce the chance of concurrent requests to the same model
    random.shuffle(items_to_process)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(deepinfra_client_process_item, item) for item in items_to_process]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(items_to_process)):
            result = future.result()
            if result:
                with file_lock:
                    f.write(json.dumps(result) + "\n")
                    id_set.add(str(result['id']) + "_" + result['model'] + "_" + str(d.get('enable_thinking', False)))
