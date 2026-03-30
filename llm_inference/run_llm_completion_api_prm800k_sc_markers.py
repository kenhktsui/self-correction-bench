import os
import json
import random
from copy import deepcopy
import concurrent.futures
import threading
import queue
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from llm_inference.prompt import get_prompt_eos_token
from llm_inference.constants import NON_RESAONING_MODEL_LIST as MODEL_LIST


TEMPERATURE = 0.0
FILE_NAME = "prm800k_sc_completion_results_api_markers.jsonl" if TEMPERATURE == 0.0 else f"prm800k_sc_completion_results_api_markers_{str(TEMPERATURE).replace('.', '_')}.jsonl"

if os.path.exists(FILE_NAME):
    print(f"File {FILE_NAME} already exists.")
else:
    with open(FILE_NAME, "w") as f:
        pass


dataset = load_dataset("super-brown/prm800k_sc", split="test")


# Create an OpenAI client with your deepinfra token and endpoint
deepinfra_client = OpenAI(api_key=os.environ.get("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai", timeout=180)


id_set = set()
with open(FILE_NAME, "r") as f:
    for line in f:
        d = json.loads(line)
        id_set.add(str(d['question']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))

print(f"Number of results that have been processed: {len(id_set)}")

def openai_client_process_item(client, item_tuple):
    model, hf_tokenizer_name, d = item_tuple
    key = str(d['question']) + "_" + (model[:-9] if model.endswith("_thinking") else model) + "_" + str(True if model.endswith("_thinking") else False)
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

    messages_error_injection_in_model_bca_but = deepcopy(d['messages_error_injection_in_model_bca'])
    assert messages_error_injection_in_model_bca_but[-1]['role'] == "assistant"
    messages_error_injection_in_model_bca_but[-1]['content'] += " But"
    prompt_error_injection_in_model_bca_but, response_error_injection_in_model_bca_but = get_prompt_and_response(messages_error_injection_in_model_bca_but,
                                                                                                                   False,
                                                                                                                   True,
                                                                                                                   enable_thinking)
    messages_error_injection_in_model_bca_however = deepcopy(d['messages_error_injection_in_model_bca'])
    assert messages_error_injection_in_model_bca_however[-1]['role'] == "assistant"
    messages_error_injection_in_model_bca_however[-1]['content'] += " However"
    prompt_error_injection_in_model_bca_however, response_error_injection_in_model_bca_however = get_prompt_and_response(messages_error_injection_in_model_bca_however,
                                                                                                                   False,
                                                                                                                   True,
                                                                                                                   enable_thinking)

    return {
        "model": model,
        "question": d['question'],
        "ground_truth_answer": d['ground_truth_answer'],
        "pre_generated_answer": d['pre_generated_answer'],
        "n_reasoning_step": d['n_reasoning_step'],
        "messages_error_injection_in_model_bca": d['messages_error_injection_in_model_bca'],
        "messages_error_injection_in_model_aca": d['messages_error_injection_in_model_aca'],
        "messages_error_in_user_bca": d['messages_error_in_user_bca'],
        "messages_error_in_user_aca": d['messages_error_in_user_aca'],
        "prompt_error_injection_in_model_bca_but": prompt_error_injection_in_model_bca_but,
        "prompt_error_injection_in_model_bca_however": prompt_error_injection_in_model_bca_however,
        "response_error_injection_in_model_bca_but": response_error_injection_in_model_bca_but.choices[0].text,
        "response_error_injection_in_model_bca_however": response_error_injection_in_model_bca_however.choices[0].text,
        "enable_thinking": enable_thinking,
        "temperature": TEMPERATURE
    }


deepinfra_client_process_item = partial(openai_client_process_item, deepinfra_client)

result_queue = queue.Queue()
writer_finished = threading.Event()

def writer_thread():
    with open(FILE_NAME, "a") as f:
        while not writer_finished.is_set() or not result_queue.empty():
            try:
                result = result_queue.get(timeout=1)
                f.write(json.dumps(result) + "\n")
                f.flush() # ensure immediate write
                id_set.add(str(result['question']) + "_" + result['model'] + "_" + str(result.get('enable_thinking', False)))
                result_queue.task_done()
            except queue.Empty:
                continue

# Start the writer thread
writer = threading.Thread(target=writer_thread, daemon=True)
writer.start()


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
            result_queue.put(result)

# Signal writer thread to finish and wait for it
writer_finished.set()
writer.join()
