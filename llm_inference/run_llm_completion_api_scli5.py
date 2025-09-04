import os
import json
import random
from copy import deepcopy
import concurrent.futures
import threading
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from llm_inference.prompt import get_prompt_eos_token
from llm_inference.constants import MODEL_LIST


TEMPERATURE = 0.6
FILE_NAME = "scli5_completion_results_api.jsonl" if TEMPERATURE == 0.0 else f"scli5_completion_results_api_{str(TEMPERATURE).replace('.', '_')}.jsonl"

if os.path.exists(FILE_NAME):
    print(f"File {FILE_NAME} already exists.")
else:
    with open(FILE_NAME, "w") as f:
        pass


dataset = load_dataset("super-brown/scli5", split="test")


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


    prompt_error_injection_in_model, response_error_injection_in_model = get_prompt_and_response(d['messages_error_injection_in_model'], False, True, enable_thinking)
    prompt_error_in_user, response_error_in_user = get_prompt_and_response(d['messages_error_in_user'], True, False, enable_thinking)

    messages_error_injection_in_model_wait = deepcopy(d['messages_error_injection_in_model'])
    assert messages_error_injection_in_model_wait[-1]['role'] == "assistant"
    messages_error_injection_in_model_wait[-1]['content'] += " Wait"
    
    prompt_error_injection_in_model_wait, response_error_injection_in_model_wait = get_prompt_and_response(messages_error_injection_in_model_wait, False, True, enable_thinking)


    messages_error_injection_in_model_cot = deepcopy(d['messages_error_injection_in_model'])
    assert messages_error_injection_in_model_cot[0]['role'] == "user"
    messages_error_injection_in_model_cot[0]['content'] += " Let's think step by step."
    
    prompt_error_injection_in_model_cot, response_error_injection_in_model_cot = get_prompt_and_response(messages_error_injection_in_model_cot, False, True, enable_thinking)
    return {
        "model": model,
        "id": d['id'],
        "question_type": d['type'],
        "messages_error_injection_in_model": d['messages_error_injection_in_model'],
        "messages_error_in_user": d['messages_error_in_user'],
        "messages_error_injection_in_model_wait": messages_error_injection_in_model_wait,
        "messages_error_injection_in_model_cot": messages_error_injection_in_model_cot,
        "prompt_error_injection_in_model": prompt_error_injection_in_model,
        "prompt_error_in_user": prompt_error_in_user,
        "prompt_error_injection_in_model_wait": prompt_error_injection_in_model_wait,
        "prompt_error_injection_in_model_cot": prompt_error_injection_in_model_cot,
        "response_error_injection_in_model": response_error_injection_in_model.choices[0].text,
        "response_error_in_user": response_error_in_user.choices[0].text,
        "response_error_injection_in_model_wait": response_error_injection_in_model_wait.choices[0].text,
        "response_error_injection_in_model_cot": response_error_injection_in_model_cot.choices[0].text,
        "enable_thinking": enable_thinking,
        "temperature": TEMPERATURE
    }


deepinfra_client_process_item = partial(openai_client_process_item, deepinfra_client)

# Test 
print(deepinfra_client_process_item(("meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct", dataset[0])))
d = {'id': 9999,
     'type': 'get_add_one',
     'messages_error_injection_in_model': [
         {'content': "What is the answer of 2 + 1? Let's think step by step.", 'role': 'user'},
         {'content': "The answer is 4.", 'role': 'assistant'}
         ], 
    'messages_error_in_user': [
        {'content': "What is the answer of 2 + 1? Let's think step by step. The answer is 4.", 'role': 'user'}
        ], 
    'correct_answer': '2'}
print(deepinfra_client_process_item(("mistralai/Mistral-Small-24B-Instruct-2501", "mistralai/Mistral-Small-24B-Instruct-2501", d)))


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
