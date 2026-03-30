import os
import json
import threading
import concurrent
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from pydantic import BaseModel
from google import genai
from google.genai import types



class Calculation(BaseModel):
    incorrect_answer: float


dataset = load_dataset("super-brown/gsm8k_sc", split="test")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
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


prompt_template = """<question>
{question}
</question>

<description_of_mistake>
{description_of_mistake}
</description_of_mistake>

<incorrect_reasoning>
{incorrect_reasoning}
</incorrect_reasoning>

You are given a question, a description of the mistake, and the resulting incorrect reasoning.
Your task is to follow the incorrect reasoning to arrive at the incorrect answer."""

eval_system_prompt = """You are a helpful assistant that follow instructions. Output in JSON format."""


def process_data(d):
    if str(d['id']) in id_set:
        return None

    prompt = prompt_template.format(
        question=d['question'],
        description_of_mistake=d['description_of_mistake'],
        incorrect_reasoning=d['messages_error_injection_in_model_bca'][1]['content'],
    )

    response = evaluate_with_llm(eval_system_prompt, prompt, Calculation)
    d['validation_system_prompt'] = eval_system_prompt
    d['validation_prompt'] = prompt
    d['validation_supporting'] = response
    return d



file_lock = threading.Lock()


id_set = set()
with open("gsm8k_sc_dataset_validation.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        id_set.add(str(d['id']))


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_data, item) for item in dataset]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset)):
        try:
            result = future.result()
            if result:
                with file_lock:
                    with open("gsm8k_sc_dataset_validation.jsonl", "a") as f:
                        f.write(json.dumps(result) + "\n")
                    id_set.add(str(result['id']))
        except concurrent.futures.TimeoutError:
            print("Future timed out")
        except Exception as e:
            print(f"Error processing future: {e}")



def clean_number(value):
    if isinstance(value, (int, float)):
        return float(value)

    if value.startswith("$"):
        value = value[1:]
    
    try:
        cleaned = value.replace(',', '').replace(' ', '')
        return float(cleaned)
    except (ValueError, TypeError):
        print(f"Could not convert {value}")
        return None


def are_close(a, b, tolerance=0.01):
    # Check if the absolute difference is less than or equal to tolerance
    return abs(a - b) <= tolerance


def validate_data(data, revalidate=False):
    for d in data:
        if d.get('validated') is not None and not revalidate:
            continue

        cleaned_number = clean_number(d['incorrect_answer'])
        if cleaned_number is None:
            def ask_human_check():
                human_check = input(f"{d['validation_supporting']['parsed']['incorrect_answer']} = {d['incorrect_answer']} (y/n)")
                if human_check == 'y':
                    d['validated'] = {"result": True, "reason": "Human check"}
                elif human_check == 'n':
                    d['validated'] = {"result": False, "reason": "Human check"}
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
                    ask_human_check()

            ask_human_check()
        else:
            d['validated'] = {"result": are_close(float(d['validation_supporting']['parsed']['incorrect_answer']), cleaned_number), "reason": "Rule-based check"}


def inspect_incorrect_data(data, revalidate=False):
    for d in data:
        if d['validated']['result'] == False and (d['validated']['reason'] != "Human check" or revalidate):
            print("id: ", d['id'])
            print("Answer from Question: ", d['incorrect_answer'])
            print("Recalculated Answer: ", d['validation_supporting']['parsed']['incorrect_answer'])
            
            def ask_human_check():
                correct_confirmed = input("Is this correct? (y/n)")
                if correct_confirmed == 'y':
                    d['validated'] = {"result": True, "reason": "Human check"}
                elif correct_confirmed == 'n':
                    d['validated'] = {"result": False, "reason": "Human check"}
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
                    ask_human_check()

            ask_human_check()


def ask_revalidate():
    revalidated = input("Re-validate? (y/n)")
    if revalidated == 'y':
        return True
    elif revalidated == 'n':
        return False
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        return ask_revalidate()


data = []
with open("gsm8k_sc_dataset_validation.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))


revalidated = ask_revalidate()
validate_data(data, revalidate=revalidated)
inspect_incorrect_data(data, revalidate=revalidated)

print(f"Frequency of validated: {Counter(d['validated']['result'] for d in data)}")
print(f"Incorrect data id: {[d['id'] for d in data if not d['validated']['result']]}")

with open("gsm8k_sc_dataset_validation.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")
