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


scli5_dataset = load_dataset("super-brown/scli5", split="test")
scli5_correct_answer = {d['id']: d['correct_answer'] for d in scli5_dataset}
gsm8k_sc_dataset = load_dataset("super-brown/gsm8k_sc", split="test")
gsm8k_sc_correct_answer = {}
for d in gsm8k_sc_dataset:
    a = d["answer"].split("\n")
    answer = a[-1]
    answer = re.sub(re.escape("####"), "", answer).strip()
    gsm8k_sc_correct_answer[d['id']] = answer
prm800k_sc_dataset = load_dataset("super-brown/prm800k_sc", split="test")
prm800k_sc_correct_answer = {d['question']: d['ground_truth_answer'] for d in prm800k_sc_dataset}


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
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


def process_data(d, correct_answer=None, correct_answer_key=None):
    if 'llm_evaluation' in d and 'llm_evaluation_error_in_user' in d:
        return d

    d['llm_evaluation_system_prompt'] = None
    d['llm_evaluation_prompt'] = None
    d['llm_evaluation'] = None

    d['llm_evaluation_prompt_error_in_user'] = None
    d['llm_evaluation_error_in_user'] = None

    if not d['response_error_injection_in_model'].strip() and not d['response_error_in_user'].strip():
        return d

    d['llm_evaluation_system_prompt'] = eval_system_prompt

    golden_answer = correct_answer[d[correct_answer_key]] if correct_answer is not None and correct_answer_key is not None else d["answer"]

    if d['response_error_injection_in_model']:
        eval_prompt = eval_prompt_template.format(
            question=d['messages_error_injection_in_model'][0]['content'],
            golden_answer=golden_answer,
            given_wrong_answer=d['messages_error_injection_in_model'][1]['content'],
            completion_from_model=d['response_error_injection_in_model']
            )
        response = evaluate_with_llm(eval_system_prompt, eval_prompt, Evaluation)
        d['llm_evaluation_prompt'] = eval_prompt
        d['llm_evaluation'] = response

    if d['response_error_in_user']:
        eval_prompt_error_in_user = eval_prompt_template_error_in_user.format(
            question_and_user_answer=d['messages_error_in_user'][0]['content'],
            golden_answer=golden_answer,
            response_from_model=d['response_error_in_user']
            )
        response_error_in_user = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user'] = eval_prompt_error_in_user
        d['llm_evaluation_error_in_user'] = response_error_in_user

    return d



def process_data_bca(d, correct_answer=None, correct_answer_key=None):
    if 'llm_evaluation_bca' in d and 'llm_evaluation_error_in_user_bca' in d:
        return d

    d['llm_evaluation_system_prompt'] = None

    d['llm_evaluation_prompt_bca'] = None
    d['llm_evaluation_bca'] = None

    d['llm_evaluation_prompt_aca'] = None
    d['llm_evaluation_aca'] = None

    d['llm_evaluation_prompt_error_in_user_bca'] = None
    d['llm_evaluation_error_in_user_bca'] = None

    if not d['response_error_injection_in_model_bca'].strip() and not d['response_error_in_user_bca'].strip():
        return d

    d['llm_evaluation_system_prompt'] = eval_system_prompt

    golden_answer = correct_answer[d[correct_answer_key]] if correct_answer is not None and correct_answer_key is not None else d["answer"]

    if d['response_error_injection_in_model_bca']:
        eval_prompt_bca = eval_prompt_template_bca.format(
            question=d['messages_error_injection_in_model_bca'][0]['content'],
            golden_answer=golden_answer,
            given_wrong_reasoning=d['messages_error_injection_in_model_bca'][1]['content'],
            completion_from_model=d['response_error_injection_in_model_bca']
            )
        response_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_bca, Evaluation)
        d['llm_evaluation_prompt_bca'] = eval_prompt_bca
        d['llm_evaluation_bca'] = response_bca

    if d['response_error_in_user_bca']:
        eval_prompt_error_in_user_bca = eval_prompt_template_error_in_user_bca.format(
            question_and_user_reasoning=d['messages_error_in_user_bca'][0]['content'],
            golden_answer=golden_answer,
            response_from_model=d['response_error_in_user_bca']
            )
        response_error_in_user_bca = evaluate_with_llm(eval_system_prompt, eval_prompt_error_in_user_bca, EvaluationErrorInUser)
        d['llm_evaluation_prompt_error_in_user_bca'] = eval_prompt_error_in_user_bca
        d['llm_evaluation_error_in_user_bca'] = response_error_in_user_bca

    return d


# claude
with open("rebuttal/claude/scli5_completion_results_claude.jsonl") as f:
    data = [json.loads(line) for line in f]
    for d in tqdm(data, desc="Processing scli5 data"):
        d = process_data(d, scli5_correct_answer, 'id')
        if d:
            with open("rebuttal/claude/scli5_completion_results_claude_llm_eval.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")

with open("rebuttal/claude/gsm8k_sc_completion_results_claude.jsonl") as f:
    data = [json.loads(line) for line in f]
    for d in tqdm(data, desc="Processing gsm8k data"):
        d = process_data_bca(d, gsm8k_sc_correct_answer, 'id')
        if d:
            with open("rebuttal/claude/gsm8k_sc_completion_results_claude_llm_eval.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")

with open("rebuttal/claude/prm800k_sc_completion_results_claude.jsonl") as f:
    data = [json.loads(line) for line in f]
    for d in tqdm(data, desc="Processing prm800k data"):
        d = process_data_bca(d, prm800k_sc_correct_answer, 'question')
        if d:
            with open("rebuttal/claude/prm800k_sc_completion_results_claude_llm_eval.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")
