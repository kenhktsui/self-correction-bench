import os
import json
import concurrent
import threading
from google import genai
from google.genai import types
from datasets import load_dataset
from tqdm import tqdm
from evaluation_with_llm.eval_prompt import (
    eval_system_prompt,
    eval_prompt_template_bca,
    eval_prompt_template,
    eval_prompt_template_error_in_user, 
    eval_prompt_template_error_in_user_bca,
    Evaluation,
    EvaluationErrorInUser
)


TEMPERATURE = 0.6
FILE_NAME_IN = "prm800k_sc_completion_results_api.jsonl" if TEMPERATURE == 0.0 else f"prm800k_sc_completion_results_api_{str(TEMPERATURE).replace('.', '_')}.jsonl"
FILE_NAME_OUT = "prm800k_sc_completion_results_llm_eval_gemini2_5_flash.jsonl" if TEMPERATURE == 0.0 else f"prm800k_sc_completion_results_llm_eval_gemini2_5_flash_{str(TEMPERATURE).replace('.', '_')}.jsonl"

if os.path.exists(FILE_NAME_OUT):
    print(f"File {FILE_NAME_OUT} already exists.")
else:
    with open(FILE_NAME_OUT, "w") as f:
        pass


dataset = load_dataset("kenhktsui/prm800k_sc", split="test")
correct_answer = {d['question']: d['ground_truth_answer'] for d in dataset}


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
        result_id_set.add(str(d['question']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))


result_id_llm_eval_set = set()
with open(FILE_NAME_OUT, "r") as f:
    for line in f:
        d = json.loads(line)
        result_id_llm_eval_set.add(str(d['question']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))


unprocessed_set = result_id_set - result_id_llm_eval_set
print(f"No of unprocessed: {len(unprocessed_set)}")



def process_data(d):
    if str(d['question']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)) not in unprocessed_set:
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

    d['llm_evaluation_prompt_bca_cot'] = None
    d['llm_evaluation_bca_cot'] = None

    d['llm_evaluation_prompt_aca_cot'] = None
    d['llm_evaluation_aca_cot'] = None

    d['llm_evaluation_system_prompt'] = eval_system_prompt

    if d['response_error_injection_in_model_bca']:
        eval_prompt_bca = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca'][0]['content'],
            golden_answer=correct_answer[d['question']],
            given_wrong_reasoning=d['messages_error_injection_in_model_bca'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca']
            )
        response_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_bca, Evaluation)
        d['llm_evaluation_prompt_bca'] = eval_prompt_bca
        d['llm_evaluation_bca'] = response_bca

    if d['response_error_injection_in_model_aca']:
        eval_prompt_aca = eval_prompt_template.format(
            question=d['messages_error_injection_in_model_aca'][0]['content'],
            golden_answer=correct_answer[d['question']],
            given_wrong_answer=d['messages_error_injection_in_model_aca'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_aca']
            )
        response_aca = evaluate_with_llm(eval_system_prompt, eval_prompt_aca, Evaluation)
        d['llm_evaluation_prompt_aca'] = eval_prompt_aca
        d['llm_evaluation_aca'] = response_aca

    if d['response_error_in_user_bca']:
        eval_prompt_error_in_user_bca = eval_prompt_template_error_in_user_bca.format(
            question_and_user_reasoning=d['messages_error_in_user_bca'][0]['content'],
            golden_answer= correct_answer[d['question']],
            response_from_model=d['response_error_in_user_bca']
            )
        response_error_in_user_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user_bca, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user_bca'] = eval_prompt_error_in_user_bca
        d['llm_evaluation_error_in_user_bca'] = response_error_in_user_bca

    if d['response_error_in_user_aca']:
        eval_prompt_error_in_user_aca = eval_prompt_template_error_in_user.format(
            question_and_user_answer=d['messages_error_in_user_aca'][0]['content'],
            golden_answer= correct_answer[d['question']],
            response_from_model=d['response_error_in_user_aca']
            )
        response_error_in_user_aca = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user_aca, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user_aca'] = eval_prompt_error_in_user_aca
        d['llm_evaluation_error_in_user_aca'] = response_error_in_user_aca
            
    if d['response_error_injection_in_model_bca_wait']:
        eval_prompt_bca_wait = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca_wait'][0]['content'],
            golden_answer= correct_answer[d['question']],
            given_wrong_reasoning=d['messages_error_injection_in_model_bca_wait'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca_wait']
            )
        response_bca_wait = evaluate_with_llm(eval_system_prompt, eval_prompt_bca_wait, Evaluation)
        d['llm_evaluation_prompt_bca_wait'] = eval_prompt_bca_wait
        d['llm_evaluation_bca_wait'] = response_bca_wait

    if d['response_error_injection_in_model_aca_wait']:
        eval_prompt_aca_wait = eval_prompt_template.format(
            question=d['messages_error_injection_in_model_aca_wait'][0]['content'],
            golden_answer= correct_answer[d['question']],
            given_wrong_answer=d['messages_error_injection_in_model_aca_wait'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_aca_wait']
            )
        response_aca_wait = evaluate_with_llm(eval_system_prompt, eval_prompt_aca_wait, Evaluation)
        d['llm_evaluation_prompt_aca_wait'] = eval_prompt_aca_wait
        d['llm_evaluation_aca_wait'] = response_aca_wait

    if d['response_error_injection_in_model_bca_cot']:
        eval_prompt_cot_bca = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca_cot'][0]['content'],
            golden_answer= correct_answer[d['question']],
            given_wrong_reasoning=d['messages_error_injection_in_model_bca_cot'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca_cot']
            )
        response_cot_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_cot_bca, Evaluation)
        d['llm_evaluation_prompt_bca_cot'] = eval_prompt_cot_bca
        d['llm_evaluation_bca_cot'] = response_cot_bca

    if d['response_error_injection_in_model_aca_cot']:
        eval_prompt_cot_aca = eval_prompt_template.format(
            question=d['messages_error_injection_in_model_aca_cot'][0]['content'],
            golden_answer= correct_answer[d['question']],
            given_wrong_answer=d['messages_error_injection_in_model_aca_cot'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_aca_cot']
            )
        response_cot_aca = evaluate_with_llm(eval_system_prompt, eval_prompt_cot_aca, Evaluation)
        d['llm_evaluation_prompt_aca_cot'] = eval_prompt_cot_aca
        d['llm_evaluation_aca_cot'] = response_cot_aca

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
                    result_id_set.add(str(result['question']) + "_" + result['model'] + "_" + str(result.get('enable_thinking', False)))
        except concurrent.futures.TimeoutError:
            print("Future timed out")
        except Exception as e:
            print(f"Error processing future: {e}")
