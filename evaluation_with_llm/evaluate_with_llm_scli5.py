import os
import json
import concurrent
import threading
from google import genai
from google.genai import types
from datasets import load_dataset
from tqdm import tqdm
from evaluation_with_llm.eval_prompt import (
    Evaluation,
    EvaluationErrorInUser,
    eval_system_prompt,
    eval_prompt_template,
    eval_prompt_template_error_in_user
)


TEMPERATURE = 0.6
FILE_NAME_IN =  "scli5_completion_results_api.jsonl" if TEMPERATURE == 0.0 else f"scli5_completion_results_api_{str(TEMPERATURE).replace('.', '_')}.jsonl"
FILE_NAME_OUT = "scli5_completion_results_llm_eval_gemini2_5_flash.jsonl" if TEMPERATURE == 0.0 else f"scli5_completion_results_llm_eval_gemini2_5_flash_{str(TEMPERATURE).replace('.', '_')}.jsonl"

if os.path.exists(FILE_NAME_OUT):
    print(f"File {FILE_NAME_OUT} already exists.")
else:
    with open(FILE_NAME_OUT, "w") as f:
        pass


dataset = load_dataset("kenhktsui/scli5", split="test")
correct_answer = {d['id']: d['correct_answer'] for d in dataset}


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                      http_options=types.HttpOptions(api_version='v1alpha')
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

    d['llm_evaluation_system_prompt'] = None
    d['llm_evaluation_prompt'] = None
    d['llm_evaluation'] = None

    d['llm_evaluation_prompt_error_in_user'] = None
    d['llm_evaluation_error_in_user'] = None

    d['llm_evaluation_prompt_wait'] = None
    d['llm_evaluation_wait'] = None

    d['llm_evaluation_prompt_cot'] = None
    d['llm_evaluation_cot'] = None

    if not d['response_error_injection_in_model'].strip() and not d['response_error_injection_in_model_wait'].strip() and not d['response_error_injection_in_model_cot'].strip() and not d['response_error_in_user'].strip():
        return d

    d['llm_evaluation_system_prompt'] = eval_system_prompt

    if d['response_error_injection_in_model']:
        eval_prompt = eval_prompt_template.format(
            question=d['messages_error_injection_in_model'][0]['content'],
            golden_answer=correct_answer[d['id']],
            given_wrong_answer=d['messages_error_injection_in_model'][1]['content'],
            completion_from_model=d['response_error_injection_in_model']
            )
        response = evaluate_with_llm(eval_system_prompt, eval_prompt, Evaluation)
        d['llm_evaluation_prompt'] = eval_prompt
        d['llm_evaluation'] = response

    if d['response_error_in_user']:
        eval_prompt_error_in_user = eval_prompt_template_error_in_user.format(
            question_and_user_answer=d['messages_error_in_user'][0]['content'],
            golden_answer=correct_answer[d['id']],
            response_from_model=d['response_error_in_user']
            )
        response_error_in_user = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user'] = eval_prompt_error_in_user
        d['llm_evaluation_error_in_user'] = response_error_in_user

    if d['response_error_injection_in_model_wait']:
        eval_prompt_wait = eval_prompt_template.format(
            question=d['messages_error_injection_in_model_wait'][0]['content'],
            golden_answer= correct_answer[d['id']],
            given_wrong_answer=d['messages_error_injection_in_model_wait'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_wait']
            )
        response_wait = evaluate_with_llm(eval_system_prompt, eval_prompt_wait, Evaluation)
        d['llm_evaluation_prompt_wait'] = eval_prompt_wait
        d['llm_evaluation_wait'] = response_wait

    if d['response_error_injection_in_model_cot']:
        eval_prompt_cot = eval_prompt_template.format(
            question=d['messages_error_injection_in_model_cot'][0]['content'],
            golden_answer= correct_answer[d['id']],
            given_wrong_answer=d['messages_error_injection_in_model_cot'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_cot']
            )
        response_cot = evaluate_with_llm(eval_system_prompt, eval_prompt_cot, Evaluation)
        d['llm_evaluation_prompt_cot'] = eval_prompt_cot
        d['llm_evaluation_cot'] = response_cot

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
