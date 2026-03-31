import os
import json
import re
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


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                      http_options=types.HttpOptions(api_version='v1alpha', timeout=60 * 10 * 1000)
                    )


def evaluate_with_llm(eval_system_prompt, prompt, response_schema):
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=eval_system_prompt,
            temperature=0.0,
            response_mime_type='application/json',
            response_schema=response_schema,
            )
    )
    return response.model_dump()


def process_data(d):
    d['llm_evaluation_system_prompt'] = None

    d['llm_evaluation_prompt_bca'] = None
    d['llm_evaluation_bca'] = None

    d['llm_evaluation_prompt_error_in_user_bca'] = None
    d['llm_evaluation_error_in_user_bca'] = None

    d['llm_evaluation_prompt_bca_wait'] = None
    d['llm_evaluation_bca_wait'] = None

    if (
        not d['response_error_injection_in_model_bca'].strip() 
        and not d['response_error_in_user_bca'].strip()
        and not d['response_error_injection_in_model_bca_wait'].strip() 
    ):
        return d

    d['llm_evaluation_system_prompt'] = eval_system_prompt

    if d['response_error_injection_in_model_bca']:
        eval_prompt_bca = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca'][0]['content'],
            golden_answer=d['answer'],
            given_wrong_reasoning=d['messages_error_injection_in_model_bca'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca']
            )
        response_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_bca, Evaluation)
        d['llm_evaluation_prompt_bca'] = eval_prompt_bca
        d['llm_evaluation_bca'] = response_bca

    if d['response_error_in_user_bca']:
        eval_prompt_error_in_user_bca = eval_prompt_template_error_in_user_bca.format(
            question_and_user_reasoning=d['messages_error_in_user_bca'][0]['content'],
            golden_answer=d['answer'],
            response_from_model=d['response_error_in_user_bca']
            )
        response_error_in_user_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user_bca, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user_bca'] = eval_prompt_error_in_user_bca
        d['llm_evaluation_error_in_user_bca'] = response_error_in_user_bca

    if d['response_error_injection_in_model_bca_wait']:
        eval_prompt_bca_wait = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca_wait'][0]['content'],
            golden_answer=d['answer'],
            given_wrong_reasoning=d['messages_error_injection_in_model_bca_wait'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca_wait']
            )
        response_bca_wait = evaluate_with_llm(eval_system_prompt, eval_prompt_bca_wait, Evaluation)
        d['llm_evaluation_prompt_bca_wait'] = eval_prompt_bca_wait
        d['llm_evaluation_bca_wait'] = response_bca_wait

    return d



processed_id_set = set()
if os.path.exists("extended_validation/domain/tracking_shuffled_objects_completion_results_api_llm_eval.jsonl"):
    with open("extended_validation/domain/tracking_shuffled_objects_completion_results_api_llm_eval.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            processed_id_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))

print(f"Number of results that have been processed: {len(processed_id_set)}")

# domain
with open("extended_validation/domain/tracking_shuffled_objects_completion_results_api.jsonl") as f:
    data = [json.loads(line) for line in f]
    for d in tqdm(data, desc="Processing domain data"):
        if str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)) in processed_id_set:
            continue
        d = process_data(d)
        if d:
            with open("extended_validation/domain/tracking_shuffled_objects_completion_results_api_llm_eval.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")



processed_id_set = set()
if os.path.exists("extended_validation/domain/logic_deduction_completion_results_api_llm_eval.jsonl"):
    with open("extended_validation/domain/logic_deduction_completion_results_api_llm_eval.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            processed_id_set.add(str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)))

print(f"Number of results that have been processed: {len(processed_id_set)}")

# domain
with open("extended_validation/domain/logic_deduction_completion_results_api.jsonl") as f:
    data = [json.loads(line) for line in f]
    for d in tqdm(data, desc="Processing domain data"):
        if str(d['id']) + "_" + d['model'] + "_" + str(d.get('enable_thinking', False)) in processed_id_set:
            continue
        d = process_data(d)
        if d:
            with open("extended_validation/domain/logic_deduction_completion_results_api_llm_eval.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")
