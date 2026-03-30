import os
import json
import random
from collections import Counter, defaultdict
from datasets import Dataset
import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm
from evaluation_with_llm.eval_prompt import eval_system_prompt


random.seed(42)


data = []
with open("/Users/superbrown/Downloads/phase2_test.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print('Total', len(data))
print('Excluding bad problem and give up', len([d for d in data if d['label']['finish_reason'] not in ["bad_problem", "give_up"]]))
print(Counter([d['generation'] for d in data]))


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                      http_options=types.HttpOptions(api_version='v1alpha')
                      )


class CheckAnswer(BaseModel):
    is_equal: bool


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


question_dict = defaultdict(lambda: [])
for i, d in tqdm(enumerate(data)):
    if not d['question']['pre_generated_answer'].strip():
        continue

    if d['is_quality_control_question'] or d['is_initial_screening_question']:
        continue

    if d['label']['finish_reason'] in ["bad_problem", "give_up"]:
        continue

    # it reflects higher complexity if ground truth answer is not the same as the pre-generated answer because the step might be to far away from the answer
    prompt = f"Is the following two answers equal?\n\nPre-generated answer: {d['question']['pre_generated_answer']}\nGround truth answer: {d['question']['ground_truth_answer']}"
    check_answer_response = evaluate_with_llm(eval_system_prompt, prompt, CheckAnswer)
    with open(f"prm800k_sc_check_answer_response.jsonl", "a") as f:
        f.write(json.dumps(
            {
                "question": d['question']['problem'],
                "ground_truth_answer": d['question']['ground_truth_answer'],
                "pre_generated_answer": d['question']['pre_generated_answer'],
                "eval_model": "gemini-2.5-flash-preview-05-20",
                "system_prompt": eval_system_prompt,
                "prompt": prompt,
                "check_answer_response": check_answer_response
            }
            ) + "\n")

    if check_answer_response['parsed']['is_equal']:
        continue
    # take the whole pregenerated steps, identify the first error step, then randomly pick the cumulative error step, create BCA; take the whole pregen steps as ACA
    question = d['question']['problem']
    wrong_reasoning_and_answer = d['question']['pre_generated_steps']

    n_reasoning_step = len(wrong_reasoning_and_answer) - 1
    bca = " ".join(wrong_reasoning_and_answer[:-1])
    aca = " ".join(wrong_reasoning_and_answer)

    question_dict[question].append({
        'question': question,
        'ground_truth_answer': d['question']['ground_truth_answer'],
        'pre_generated_answer': d['question']['pre_generated_answer'],
        'n_reasoning_step': n_reasoning_step,
        'messages_error_injection_in_model_bca': [{"role": "user", "content": question}, {"role": "assistant", "content": bca}],
        'messages_error_injection_in_model_aca': [{"role": "user", "content": question}, {"role": "assistant", "content": aca}],
        'messages_error_in_user_bca': [{"role": "user", "content": question + " " + bca}],
        'messages_error_in_user_aca': [{"role": "user", "content": question + " " + aca}],
        'check_answer_system_prompt': eval_system_prompt,
        'check_answer_prompt': prompt,
        'check_answer_response': check_answer_response,
        'eval_model': "gemini-2.5-flash-preview-05-20",
    })



prm800k_sc_data = []
for question, d in question_dict.items():
    prm800k_sc_data.append(random.choice(d))


ds_sc = Dataset.from_list(prm800k_sc_data)
assert len(set(ds_sc["question"])) == len(ds_sc)
print("n_reasoning_step distribution", np.bincount(ds_sc["n_reasoning_step"]))
print(ds_sc)
ds_sc.push_to_hub("super-brown/prm800k_sc", split="test")
