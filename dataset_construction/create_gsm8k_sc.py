import os
from typing import List
import string
import re
import random
import json
from datasets import load_dataset, Dataset
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm


ds = load_dataset("openai/gsm8k", "main", split="test")
question2id = {q: i for i, q in enumerate(ds["question"])}


class ReasoningWithMistake(BaseModel):
    reasoning_steps_with_one_mistake: List[str]
    mistake_step: int
    type_of_mistake: str
    description_of_mistake: str
    incorrect_answer: str


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


system_prompt = """You are a helpful assistant that follow instructions. Output in JSON format."""


ERROR_TYPES = {
    "Problem Representation Errors": "These errors arise when the solver misunderstands or misinterprets the problem’s requirements or given information. This can involve misreading the problem statement, confusing the relationships between quantities, or failing to grasp what is being asked.",
    "Planning Errors": "These occur when the solver devises an incorrect or incomplete strategy to tackle the problem. This might include choosing the wrong operations, setting up flawed equations, or overlooking key components of the problem.",
    "Execution Errors": "These are mistakes made while carrying out the planned steps, such as errors in calculations, misapplication of mathematical rules, or procedural slip-ups, even if the plan itself is sound.",
}

question_set = set()
with open("gsm8k_sc.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        question_set.add(d['question'])

if question_set == set(ds["question"]):
    print("All questions have been processed")
    data = []
    with open("gsm8k_sc.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            d["id"] = question2id[d["question"]]
            data.append(d)

    data = sorted(data, key=lambda x: x["id"])
    ds_sc = Dataset.from_list(data)

    def create_messages(x):
        question = x['question']
        steps = [i + '.' if i[-1] not in string.punctuation else i for i in x['reasoning_steps_with_one_mistake'] if i]
        steps = [re.sub(r'\[(\d+)\]', '', step).strip() for step in steps]
        reasoning = ' '.join(steps)
        incorrect_answer = f' The answer is {x["incorrect_answer"]}.'
        return {
            "messages_error_injection_in_model_bca": [{"role": "user", "content": question},
                                                      {"role": "assistant", "content": reasoning}],
            "messages_error_injection_in_model_aca": [{"role": "user", "content": question},
                                                      {"role": "assistant", "content": reasoning + incorrect_answer}],
            "messages_error_in_user_bca": [{"role": "user", "content": question + " " + reasoning}],
            "messages_error_in_user_aca": [{"role": "user", "content": question + " " + reasoning + incorrect_answer}]
        }
    ds_sc = ds_sc.map(create_messages)
    ds_sc.push_to_hub("super-brown/gsm8k_sc", split="test")
    exit()

with open("gsm8k_sc.jsonl", "a") as f:
    for d in tqdm(ds):
        if d["question"] in question_set:
            continue

        question = d["question"]
        a = d["answer"].split("\n")
        reasoning = a[:-1]
        answer = a[-1]
        answer = re.sub(re.escape("####"), "", answer).strip()

        error_type = random.choice(list(ERROR_TYPES.keys()))
        error_description = ERROR_TYPES[error_type]
        mistake_step = random.randint(1, len(reasoning))
        prompt = f"""<question> 
{question} 
</question>

<reasoning_steps> 
{os.linesep.join([f"[{i+1}] {r}" for i, r in enumerate(reasoning)])}
</reasoning_steps>

<answer> 
{answer}
</answer>

<type_of_mistake>
{error_type}: {error_description}
</type_of_mistake>

You task is to introduce one mistake in step {mistake_step} in <reasoning_steps> and arrive at an answer different from <answer>. 
You will output: 
- <reasoning_steps> with mistake
- the step that contains the mistake
- type of the mistake
- description of the mistake
- incorrect answer"""

        response = client.responses.parse(
            model="gpt-4.1-2025-04-14",
            input=[
                {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=ReasoningWithMistake,
        temperature=0.4,
        )

        result = response.output_parsed
        result = result.model_dump()
        if result["mistake_step"] != mistake_step:
            continue
        new_d = {**d, **result, "system_prompt": system_prompt, "prompt": prompt, "generation_model": "gpt-4.1-2025-04-14", "temperature": 0.4}
        f.write(json.dumps(new_d) + "\n")
