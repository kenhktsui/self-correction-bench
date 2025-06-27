import os
import json
import re
import concurrent
import threading
from google import genai
from google.genai import types
from datasets import load_dataset
from tqdm import tqdm
from eval_prompt import (
    Evaluation,
    EvaluationErrorInUser,
    eval_system_prompt,
    eval_prompt_template_bca,
    eval_prompt_template,
    eval_prompt_template_error_in_user_bca,
    eval_prompt_template_error_in_user
)


dataset = load_dataset("kenhktsui/gsm8k_sc", split="test")
correct_answer = {}
for d in dataset:
    a = d["answer"].split("\n")
    answer = a[-1]
    answer = re.sub(re.escape("####"), "", answer).strip()
    correct_answer[d['id']] = answer


client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version='v1alpha', timeout=60 * 3 * 1000)
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
with open("gsm8k_sc_completion_results_api.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        data.append(d)
        result_id_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))


result_id_llm_eval_set = set()
with open("gsm8k_sc_completion_results_llm_eval_gemini2_5_flash.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        result_id_llm_eval_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))


unprocessed_set = result_id_set - result_id_llm_eval_set
print(f"No of unprocessed: {len(unprocessed_set)}")


def process_data(d):
    if str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)) not in unprocessed_set:
        return None

    d['llm_evaluation_system_prompt'] = None

    d['llm_evaluation_prompt_bca'] = None
    d['llm_evaluation_bca'] = None

    d['llm_evaluation_prompt_aca'] = None
    d['llm_evaluation_aca'] = None

    d['llm_evaluation_prompt_error_in_user_bca'] = None
    d['llm_evaluation_error_in_user_bca'] = None

    d['llm_evaluation_prompt_error_in_user_aca'] = None
    d['llm_evaluation_error_in_user_aca'] = None

    d['llm_evaluation_prompt_bca_wait'] = None
    d['llm_evaluation_bca_wait'] = None

    d['llm_evaluation_prompt_aca_wait'] = None
    d['llm_evaluation_aca_wait'] = None

    if (
        not d['response_error_injection_in_model_bca'].strip() 
        and not d['response_error_injection_in_model_aca'].strip() 
        and not d['response_error_in_user_bca'].strip()
        and not d['response_error_in_user_aca'].strip()
        and not d['response_error_injection_in_model_bca_wait'].strip() 
        and not d['response_error_injection_in_model_aca_wait'].strip()
    ):
        return d

    d['llm_evaluation_system_prompt'] = eval_system_prompt

    if d['response_error_injection_in_model_bca']:
        eval_prompt_bca = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca'][0]['content'],
            golden_answer=correct_answer[d['id']],
            given_wrong_reasoning=d['messages_error_injection_in_model_bca'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca']
            )
        response_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_bca, Evaluation)
        d['llm_evaluation_prompt_bca'] = eval_prompt_bca
        d['llm_evaluation_bca'] = response_bca

    if d['response_error_injection_in_model_aca']:
        eval_prompt_aca = eval_prompt_template.format(
            question=d['messages_error_injection_in_model_aca'][0]['content'],
            golden_answer= correct_answer[d['id']],
            given_wrong_answer=d['messages_error_injection_in_model_aca'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_aca']
            )
        response_aca = evaluate_with_llm(eval_system_prompt, eval_prompt_aca, Evaluation)
        d['llm_evaluation_prompt_aca'] = eval_prompt_aca
        d['llm_evaluation_aca'] = response_aca

    if d['response_error_in_user_bca']:
        eval_prompt_error_in_user_bca = eval_prompt_template_error_in_user_bca.format(
            question_and_user_reasoning=d['messages_error_in_user_bca'][0]['content'],
            golden_answer=correct_answer[d['id']],
            response_from_model=d['response_error_in_user_bca']
            )
        response_error_in_user_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user_bca, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user_bca'] = eval_prompt_error_in_user_bca
        d['llm_evaluation_error_in_user_bca'] = response_error_in_user_bca

    if d['response_error_in_user_aca']:
        eval_prompt_error_in_user_aca = eval_prompt_template_error_in_user.format(
            question_and_user_answer=d['messages_error_in_user_aca'][0]['content'],
            golden_answer=correct_answer[d['id']],
            response_from_model=d['response_error_in_user_aca']
            )
        response_error_in_user_aca = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user_aca, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user_aca'] = eval_prompt_error_in_user_aca
        d['llm_evaluation_error_in_user_aca'] = response_error_in_user_aca

    if d['response_error_injection_in_model_bca_wait']:
        eval_prompt_bca_wait = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca_wait'][0]['content'],
            golden_answer=correct_answer[d['id']],
            given_wrong_reasoning=d['messages_error_injection_in_model_bca_wait'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca_wait']
            )
        response_bca_wait = evaluate_with_llm(eval_system_prompt, eval_prompt_bca_wait, Evaluation)
        d['llm_evaluation_prompt_bca_wait'] = eval_prompt_bca_wait
        d['llm_evaluation_bca_wait'] = response_bca_wait

    if d['response_error_injection_in_model_aca_wait']:
        eval_prompt_aca_wait = eval_prompt_template.format(
            question=d['messages_error_injection_in_model_aca_wait'][0]['content'],
            golden_answer= correct_answer[d['id']],
            given_wrong_answer=d['messages_error_injection_in_model_aca_wait'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_aca_wait']
            )
        response_aca_wait = evaluate_with_llm(eval_system_prompt, eval_prompt_aca_wait, Evaluation)
        d['llm_evaluation_prompt_aca_wait'] = eval_prompt_aca_wait
        d['llm_evaluation_aca_wait'] = response_aca_wait

    return d



file_lock = threading.Lock()


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_data, item) for item in data]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
        try:
            result = future.result()
            if result:
                with file_lock:
                    with open("gsm8k_sc_completion_results_llm_eval_gemini2_5_flash.jsonl", "a") as f:
                        f.write(json.dumps(result) + "\n")
                    result_id_set.add(str(result['id']) + "_" + result['model'] + "_" + str(result.get('enable_thinking', False)))
        except concurrent.futures.TimeoutError:
            print("Future timed out")
        except Exception as e:
            print(f"Error processing future: {e}")
