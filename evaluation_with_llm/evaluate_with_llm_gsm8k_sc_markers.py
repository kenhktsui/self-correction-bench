import os
import json
import re
import concurrent
import threading
from copy import deepcopy
from google import genai
from google.genai import types
from datasets import load_dataset
from tqdm import tqdm
from evaluation_with_llm.eval_prompt import (
    Evaluation,
    eval_system_prompt,
    eval_prompt_template_bca,
)


FILE_NAME_IN = "gsm8k_sc_completion_results_api_markers.jsonl" 
FILE_NAME_OUT = "gsm8k_sc_completion_results_llm_eval_gemini2_5_flash_markers.jsonl"


if os.path.exists(FILE_NAME_OUT):
    print(f"File {FILE_NAME_OUT} already exists.")
else:
    with open(FILE_NAME_OUT, "w") as f:
        pass


dataset = load_dataset("kenhktsui/gsm8k_sc", split="test")
correct_answer = {}
for d in dataset:
    a = d["answer"].split("\n")
    answer = a[-1]
    answer = re.sub(re.escape("####"), "", answer).strip()
    correct_answer[d['id']] = answer


client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version='v1alpha', timeout=60 * 10 * 1000)
)


def evaluate_with_llm(eval_system_prompt, prompt, response_schema):
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=eval_system_prompt,
            temperature=0.0,
            response_mime_type='application/json',
            response_schema=response_schema,
            )
    )
    return response.model_dump()


data = []
result_id_set = set()
with open(FILE_NAME_IN, "r") as f:
    for line in f:
        d = json.loads(line)
        data.append(d)
        result_id_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))


result_id_llm_eval_set = set()
with open(FILE_NAME_OUT, "r") as f:
    for line in f:
        d = json.loads(line)
        result_id_llm_eval_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))


unprocessed_set = result_id_set - result_id_llm_eval_set
print(f"No of unprocessed: {len(unprocessed_set)}")


def process_data(d):
    if str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)) not in unprocessed_set:
        return None

    d['llm_evaluation_prompt_but'] = None
    d['llm_evaluation_but'] = None

    d['llm_evaluation_prompt_however'] = None
    d['llm_evaluation_however'] = None


    if not d['response_error_injection_in_model_bca_but'].strip() and not d['response_error_injection_in_model_bca_however'].strip():
        return d

    messages_error_injection_in_model_but = deepcopy(d['messages_error_injection_in_model_bca'])
    assert messages_error_injection_in_model_but[-1]['role'] == "assistant"
    messages_error_injection_in_model_but[-1]['content'] += " But"

    messages_error_injection_in_model_however = deepcopy(d['messages_error_injection_in_model_bca'])
    assert messages_error_injection_in_model_however[-1]['role'] == "assistant"
    messages_error_injection_in_model_however[-1]['content'] += " However"

    d['llm_evaluation_system_prompt'] = eval_system_prompt
    if d['response_error_injection_in_model_bca_but']:
        eval_prompt_but = eval_prompt_template_bca.format(
            question=messages_error_injection_in_model_but[0]['content'],
            golden_answer= correct_answer[d['id']],
            given_wrong_reasoning=messages_error_injection_in_model_but[1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca_but']
            )
        response_but = evaluate_with_llm(eval_system_prompt, eval_prompt_but, Evaluation)
        d['llm_evaluation_prompt_bca_but'] = eval_prompt_but
        d['llm_evaluation_bca_but'] = response_but

    if d['response_error_injection_in_model_bca_however']:
        eval_prompt_however = eval_prompt_template_bca.format(
            question=messages_error_injection_in_model_however[0]['content'],
            golden_answer= correct_answer[d['id']],
            given_wrong_reasoning=messages_error_injection_in_model_however[1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca_however']
            )
        response_however = evaluate_with_llm(eval_system_prompt, eval_prompt_however, Evaluation)
        d['llm_evaluation_prompt_bca_however'] = eval_prompt_however
        d['llm_evaluation_bca_however'] = response_however

    return d



file_lock = threading.Lock()


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_data, item) for item in data]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
        try:
            result = future.result()
            if result:
                with file_lock:
                    with open(FILE_NAME_OUT, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    result_id_set.add(str(result['id']) + "_" + result['model'] + "_" + str(result.get('enable_thinking', False)))
        except concurrent.futures.TimeoutError:
            print("Future timed out")
        except Exception as e:
            print(f"Error processing future: {e}")
