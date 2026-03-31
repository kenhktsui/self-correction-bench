import os
import json
import random
import concurrent.futures
import threading
import queue
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from llm_inference.prompt import get_prompt_eos_token


MODEL_LIST = {
    "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Qwen/Qwen3-14B": "Qwen/Qwen3-14B",
    "mistralai/Mistral-Small-24B-Instruct-2501": "mistralai/Mistral-Small-24B-Instruct-2501"
}



TEMPERATURE = 0.0
FILE_NAME = "extended_validation/domain/tracking_shuffled_objects_completion_results_api.jsonl" if TEMPERATURE == 0.0 else f"extended_validation/domain/tracking_shuffled_objects_completion_results_api_{str(TEMPERATURE).replace('.', '_')}.jsonl"

if os.path.exists(FILE_NAME):
    print(f"File {FILE_NAME} already exists.")
else:
    with open(FILE_NAME, "w") as f:
        pass

raw_dataset = []
with open("/Users/superbrown/Downloads/tracking_shuffled_objects.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        raw_dataset.append(d)


dataset = []
for i, d in enumerate(raw_dataset):
    if d['mistake_index'] is None:
        continue

    d['id'] = i
    reasoning_step_without_answer = " ".join(d['steps'][:-1])
    d['messages_error_injection_in_model_bca'] = [
        {"role": "user", "content": d['input']},
        {"role": "assistant", "content": reasoning_step_without_answer},
    ]
    d['messages_error_in_user_bca'] = [
        {"role": "user", "content": d['input'] + "\n" + reasoning_step_without_answer},
    ]
    dataset.append(d)

print(len(dataset))


# Create an OpenAI client with your deepinfra token and endpoint
deepinfra_client = OpenAI(api_key=os.environ.get("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai", timeout=60, max_retries=3)

id_set = set()
with open(FILE_NAME, "r") as f:
    for line in f:
        d = json.loads(line)
        id_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))

print(f"Number of results that have been processed: {len(id_set)}")

def openai_client_process_item(client, item_tuple):
    model, hf_tokenizer_name, d = item_tuple
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
    prompt_error_in_user_bca, response_error_in_user_bca = get_prompt_and_response(d['messages_error_in_user_bca'],
                                                                             True,
                                                                             False,
                                                                             enable_thinking)
    
    messages_error_injection_in_model_bca_wait = deepcopy(d['messages_error_injection_in_model_bca'])
    assert messages_error_injection_in_model_bca_wait[-1]['role'] == "assistant"
    messages_error_injection_in_model_bca_wait[-1]['content'] += " Wait"
    prompt_error_injection_in_model_bca_wait, response_error_injection_in_model_bca_wait = get_prompt_and_response(messages_error_injection_in_model_bca_wait,
                                                                                                                   False,
                                                                                                                   True,
                                                                                                                   enable_thinking)
    return {
        "model": model,
        "id": d['id'],
        "messages_error_injection_in_model_bca": d['messages_error_injection_in_model_bca'],
        "messages_error_in_user_bca": d['messages_error_in_user_bca'],
        "messages_error_injection_in_model_bca_wait": messages_error_injection_in_model_bca_wait,
        "prompt_error_injection_in_model_bca": prompt_error_injection_in_model_bca,
        "prompt_error_in_user_bca": prompt_error_in_user_bca,
        "prompt_error_injection_in_model_bca_wait": prompt_error_injection_in_model_bca_wait,
        "response_error_injection_in_model_bca": response_error_injection_in_model_bca.choices[0].text,
        "response_error_in_user_bca": response_error_in_user_bca.choices[0].text,
        "response_error_injection_in_model_bca_wait": response_error_injection_in_model_bca_wait.choices[0].text,
        "enable_thinking": enable_thinking,
        "temperature": TEMPERATURE,
        "answer": d["target"],
    }


deepinfra_client_process_item = partial(openai_client_process_item, deepinfra_client)

# Queue for collecting results and dedicated writer thread
result_queue = queue.Queue()
writer_finished = threading.Event()
processed_count = 0
failed_count = 0

def writer_thread():
    global processed_count
    print("Writer thread started")
    try:
        with open(FILE_NAME, "a") as f:
            while not writer_finished.is_set() or not result_queue.empty():
                try:
                    result = result_queue.get(timeout=1)
                    f.write(json.dumps(result) + "\n")
                    f.flush()  # Ensure immediate write
                    id_set.add(str(result['id']) + "_" + result['model'] + "_" + str(result.get('enable_thinking', False)))
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} items successfully, {failed_count} failed")
                    result_queue.task_done()
                except queue.Empty:
                    continue
        print("Writer thread finished normally")
    except Exception as e:
        print(f"Writer thread crashed with error: {str(e)}")
        raise

# Start the writer thread
writer = threading.Thread(target=writer_thread, daemon=True)
writer.start()

items_to_process = []
for model, hf_tokenizer_name in MODEL_LIST.items():
    for d in dataset:
        if str(d['id']) + "_" + (model[:-9] if model.endswith("_thinking") else model) + "_" + str(True if model.endswith("_thinking") else False) not in id_set:
            items_to_process.append((model, hf_tokenizer_name, d))

# shuffle the model to reduce the chance of concurrent requests to the same model
random.shuffle(items_to_process)




with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(deepinfra_client_process_item, item) for item in items_to_process]
    for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(items_to_process))):
        try:
            result = future.result()
            if result:
                result_queue.put(result)
        except Exception as e:
            print(f"Thread failed with error: {str(e)}")
            failed_count += 1
            continue
        
        # Check writer thread health every 50 items
        if i % 50 == 0:
            if not writer.is_alive():
                print("WARNING: Writer thread has died! Restarting...")
                writer = threading.Thread(target=writer_thread, daemon=True)
                writer.start()
            else:
                print(f"Writer thread is alive. Queue size: {result_queue.qsize()}")

# Signal writer thread to finish and wait for it
writer_finished.set()
writer.join()
