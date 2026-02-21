import os
import json
import re
from google import genai
from google.genai import types
from datasets import load_dataset
from tqdm import tqdm
from pydantic import BaseModel
from evaluation_with_llm.eval_prompt import (
    eval_system_prompt,
    Evaluation,
)


class Evaluation(BaseModel):
    is_correct_answer: bool



eval_prompt_template_on_policy_error = """<golden_answer>
{golden_answer}
</golden_answer>

<response_from_model>
{response_from_model}
</response_from_model>

The model was provided with the golden answer <golden_answer>.
You have to assess if <response_from_model> provides the correct answer that matches the <golden_answer>."""


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                      http_options=types.HttpOptions(api_version='v1alpha', timeout=60 * 10 * 1000)
                    )


def evaluate_with_llm(eval_system_prompt, prompt, response_schema):
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        # model="gemini-2.5-flash-preview-05-20",
        # model="gemini-2.5-flash-lite",
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
    if 'llm_evaluation_bca' in d and 'llm_evaluation_error_in_user_bca' in d:
        return d

    d['llm_evaluation_system_prompt'] = None

    d['llm_evaluation_prompt_on_policy_error'] = None
    d['llm_evaluation_on_policy_error'] = None

    if not d['response'].strip():
        return d

    d['llm_evaluation_prompt'] = eval_system_prompt

    golden_answer = d["answer"]

    if d['response']:
        eval_prompt_on_policy_error = eval_prompt_template_on_policy_error.format(
            golden_answer=golden_answer,
            response_from_model=d['response']
            )
        response_on_policy_error = evaluate_with_llm(eval_system_prompt, eval_prompt_on_policy_error, Evaluation)
        d['llm_evaluation_prompt_on_policy_error'] = eval_prompt_on_policy_error
        d['llm_evaluation_on_policy_error'] = response_on_policy_error

    return d


# with open("rebuttal/on_policy_error/on_policy_error_gsm8k_v2.jsonl") as f:
#     data = [json.loads(line) for line in f]
#     for d in tqdm(data, desc="Processing on policy error gsm8k data"):
#         d = process_data(d)
#         if d:
#             with open("rebuttal/on_policy_error/on_policy_error_gsm8k_v2_llm_eval.jsonl", "a") as f:
#                 f.write(json.dumps(d) + "\n")

# with open("rebuttal/on_policy_error/on_policy_error_math_v2.jsonl") as f:
#     data = [json.loads(line) for line in f]
#     for d in tqdm(data, desc="Processing on policy error math data"):
#         d = process_data(d)
#         if d:
#             with open("rebuttal/on_policy_error/on_policy_error_math_v2_llm_eval.jsonl", "a") as f:
#                 f.write(json.dumps(d) + "\n")

# with open("rebuttal/on_policy_error/on_policy_error_olympiadbench_v2.jsonl") as f:
#     data = [json.loads(line) for line in f]
#     for d in tqdm(data, desc="Processing on policy error olympiadbench data"):
#         d = process_data(d)
#         if d:
#             with open("rebuttal/on_policy_error/on_policy_error_olympiadbench_v2_llm_eval.jsonl", "a") as f:
#                 f.write(json.dumps(d) + "\n")


with open("rebuttal/on_policy_error/on_policy_error_omnimath_v2.jsonl") as f:    
    data = [json.loads(line) for line in f][:21]
    for d in tqdm(data, desc="Processing on policy error omnimath data"):
        d = process_data(d)
        if d:
            with open("rebuttal/on_policy_error/on_policy_error_omnimath_v2_llm_eval.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")
