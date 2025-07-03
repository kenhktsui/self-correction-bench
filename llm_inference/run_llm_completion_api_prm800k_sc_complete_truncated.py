import os
import json
import random
import pickle
from types import MappingProxyType
import concurrent.futures
import threading
import queue
from functools import partial
from tqdm import tqdm
from openai import OpenAI
from llm_inference.prompt import get_prompt_eos_token
from llm_inference.constants import NON_RESAONING_MODEL_LIST as MODEL_LIST


TEMPERATURE = 0.0
FILE_NAME_IN = "prm800k_sc_completion_results_api.jsonl" if TEMPERATURE == 0.0 else f"prm800k_sc_completion_results_api_{str(TEMPERATURE).replace('.', '_')}.jsonl"
FILE_NAME_OUT = "prm800k_sc_completion_results_api_supplement.jsonl" if TEMPERATURE == 0.0 else f"prm800k_sc_completion_results_api_supplement_{str(TEMPERATURE).replace('.', '_')}.jsonl"

if os.path.exists(FILE_NAME_OUT):
    print(f"File {FILE_NAME_OUT} already exists.")
else:
    with open(FILE_NAME_OUT, "w") as f:
        pass


RESPONSE_LIST = [
        "response_error_injection_in_model_bca",
        "response_error_injection_in_model_aca",
        "response_error_in_user_bca",
        "response_error_in_user_aca",
        "response_error_injection_in_model_bca_wait",
        "response_error_injection_in_model_aca_wait"
    ]

# Create an OpenAI client with your deepinfra token and endpoint
deepinfra_client = OpenAI(api_key=os.environ.get("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai", timeout=300)

FILE_NAME_LEN_PROFILE = FILE_NAME_IN[:-6] + "_len_profile.pkl"
if not os.path.exists(FILE_NAME_LEN_PROFILE):
    
    from transformers import AutoTokenizer
    TOKENIZER_MAP =  MappingProxyType({k: AutoTokenizer.from_pretrained(v) for k, v in MODEL_LIST.items()})


    def get_tokens_len_list(d):
        text_list = [d[key] for key in RESPONSE_LIST]

        tokenizer = TOKENIZER_MAP[d['model']]
        tokens_list = tokenizer(
            text_list,
            add_special_tokens=False,
        )['input_ids']
        tokens_list = [len(tokens) for tokens in tokens_list]
        return tokens_list


    length_dict = {}
    with open(FILE_NAME_IN, "r") as f:
        for line in tqdm(f):
            d = json.loads(line)
            if d['model'] not in MODEL_LIST or d.get('enable_thinking'):
                continue
            tokens_len_list = get_tokens_len_list(d)
            length_dict[d['model'], d['question'], d.get('enable_thinking', False)] = tokens_len_list

    with open(FILE_NAME_LEN_PROFILE, "wb") as f:
        pickle.dump(length_dict, f)
    print("Length profile file created. You have to run the script again to get the supplement results.")
else:
    id_set = set()
    with open(FILE_NAME_OUT, "r") as f:
        for line in f:
            d = json.loads(line)
            id_set.add(str(d['question']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))

    print(f"Number of results that have been processed: {len(id_set)}")

    items_to_process = []
    with open(FILE_NAME_LEN_PROFILE, "rb") as f:
        length_dict = pickle.load(f)

    with open(FILE_NAME_IN, "r") as f:
        for line in tqdm(f):
            d = json.loads(line)
            if d['model'] not in MODEL_LIST or d.get('enable_thinking'):
                continue
            tokens_len_list = length_dict[d['model'], d['question'], d.get('enable_thinking', False)]
            if max(tokens_len_list) <= 1022:
                continue
            items_to_process.append([d['model'], MODEL_LIST[d['model']], d, tokens_len_list])


    def openai_client_process_item(client, item_tuple):
        model, hf_tokenizer_name, d, tokens_len_list = item_tuple
        key = str(d['question']) + "_" + (model[:-9] if model.endswith("_thinking") else model) + "_" + str(True if model.endswith("_thinking") else False)
        if key in id_set:
            return None

        # handling case of hybrid model
        enable_thinking = d.get('enable_thinking', False)

        def get_prompt_and_response(messages, add_generation_prompt, continue_final_message, enable_thinking):
            prompt, eos_token = get_prompt_eos_token(model,
                                                    hf_tokenizer_name,
                                                    messages,
                                                    add_generation_prompt,
                                                    continue_final_message,
                                                    enable_thinking,
                                                    )
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=TEMPERATURE,
                stop=[eos_token],
                max_tokens=4096
            )
            return prompt, response

        prompt_error_injection_in_model_bca, response_error_injection_in_model_bca = None, None
        prompt_error_injection_in_model_aca, response_error_injection_in_model_aca = None, None
        prompt_error_in_user_bca, response_error_in_user_bca = None, None
        prompt_error_in_user_aca, response_error_in_user_aca = None, None
        prompt_error_injection_in_model_bca_wait, response_error_injection_in_model_bca_wait = None, None
        prompt_error_injection_in_model_aca_wait, response_error_injection_in_model_aca_wait = None, None

        for tok_length, response_type in zip(tokens_len_list, RESPONSE_LIST):
            if tok_length >= 1022:
                if response_type == "response_error_injection_in_model_bca":
                    prompt_error_injection_in_model_bca, response_error_injection_in_model_bca = get_prompt_and_response(d['messages_error_injection_in_model_bca'], False, True, enable_thinking)
                elif response_type == "response_error_injection_in_model_aca":
                    prompt_error_injection_in_model_aca, response_error_injection_in_model_aca = get_prompt_and_response(d['messages_error_injection_in_model_aca'], False, True, enable_thinking)
                elif response_type == "response_error_in_user_bca":
                    prompt_error_in_user_bca, response_error_in_user_bca = get_prompt_and_response(d['messages_error_in_user_bca'], True, False, enable_thinking)
                elif response_type == "response_error_in_user_aca":
                    prompt_error_in_user_aca, response_error_in_user_aca = get_prompt_and_response(d['messages_error_in_user_aca'], True, False, enable_thinking)
                elif response_type == "response_error_injection_in_model_bca_wait":
                    prompt_error_injection_in_model_bca_wait, response_error_injection_in_model_bca_wait = get_prompt_and_response(d["messages_error_injection_in_model_bca_wait"], False, True, enable_thinking)
                elif response_type == "response_error_injection_in_model_aca_wait":
                    prompt_error_injection_in_model_aca_wait, response_error_injection_in_model_aca_wait = get_prompt_and_response(d["messages_error_injection_in_model_aca_wait"], False, True, enable_thinking)

        if (
            response_error_injection_in_model_bca is None and
            response_error_injection_in_model_aca is None and
            response_error_in_user_bca is None and
            response_error_in_user_aca is None and
            response_error_injection_in_model_bca_wait is None and
            response_error_injection_in_model_aca_wait is None
            ):
            return None

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
            "messages_error_injection_in_model_bca_wait": d["messages_error_injection_in_model_bca_wait"],
            "messages_error_injection_in_model_aca_wait": d["messages_error_injection_in_model_aca_wait"],
            "prompt_error_injection_in_model_bca": prompt_error_injection_in_model_bca,
            "prompt_error_injection_in_model_aca": prompt_error_injection_in_model_aca,
            "prompt_error_in_user_bca": prompt_error_in_user_bca,
            "prompt_error_in_user_aca": prompt_error_in_user_aca,
            "prompt_error_injection_in_model_bca_wait": prompt_error_injection_in_model_bca_wait,
            "prompt_error_injection_in_model_aca_wait": prompt_error_injection_in_model_aca_wait,
            "response_error_injection_in_model_bca": response_error_injection_in_model_bca.choices[0].text if response_error_injection_in_model_bca else None,
            "response_error_injection_in_model_aca": response_error_injection_in_model_aca.choices[0].text if response_error_injection_in_model_aca else None,
            "response_error_in_user_bca": response_error_in_user_bca.choices[0].text if response_error_in_user_bca else None,
            "response_error_in_user_aca": response_error_in_user_aca.choices[0].text if response_error_in_user_aca else None,
            "response_error_injection_in_model_bca_wait": response_error_injection_in_model_bca_wait.choices[0].text if response_error_injection_in_model_bca_wait else None,
            "response_error_injection_in_model_aca_wait": response_error_injection_in_model_aca_wait.choices[0].text if response_error_injection_in_model_aca_wait else None,
            "enable_thinking": enable_thinking,
            "temperature": TEMPERATURE
        }


    deepinfra_client_process_item = partial(openai_client_process_item, deepinfra_client)

    result_queue = queue.Queue()
    writer_finished = threading.Event()

    def writer_thread():
        with open(FILE_NAME_OUT, "a") as f:
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

    # shuffle the model to reduce the chance of concurrent requests to the same model
    random.shuffle(items_to_process)

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(deepinfra_client_process_item, item) for item in items_to_process]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(items_to_process)):
            result = future.result()
            if result:
                result_queue.put(result)

    # Signal writer thread to finish and wait for it
    writer_finished.set()
    writer.join()
