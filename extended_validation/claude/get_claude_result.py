import os
import json
from copy import deepcopy
from functools import partial
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from transformers import AutoTokenizer
from anthropic import Anthropic


MODEL_ANTHROPIC = [
    'claude-3-5-haiku-20241022',
    'claude-sonnet-4-20250514',
]


scli5_eval_data = load_scli5_eval_data()
gsm8k_eval_data = load_gsm8k_sc_eval_data()
prm800k_eval_data = load_prm800k_sc_eval_data()
print(scli5_eval_data[0].keys())
print(gsm8k_eval_data[0].keys())
print(prm800k_eval_data[0].keys())


scli5_id_hashmap = {}
gsm8k_id_hashmap = {}
prm800k_id_hashmap = {}
for d in scli5_eval_data:
    if d['model'] in MODEL_ANTHROPIC:
        scli5_id_hashmap[d['id'], d['model']] = d
for d in gsm8k_eval_data:
    if d['model'] in MODEL_ANTHROPIC:
        gsm8k_id_hashmap[d['id'], d['model']] = d
for d in prm800k_eval_data:
    if d['model'] in MODEL_ANTHROPIC:
        prm800k_id_hashmap[d['question'], d['model']] = d


print(f"Number of results that have been processed: {len(scli5_id_hashmap)}")
print(f"Number of results that have been processed: {len(gsm8k_id_hashmap)}")
print(f"Number of results that have been processed: {len(prm800k_id_hashmap)}")


scli5_id_processed = set()
if os.path.exists("rebuttal/claude/scli5_completion_results_claude.jsonl"):
    with open("rebuttal/claude/scli5_completion_results_claude.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            scli5_id_processed.add((d['id'], d['model']))

gsm8k_id_processed = set()
if os.path.exists("rebuttal/claude/gsm8k_sc_completion_results_claude.jsonl"):
    with open("rebuttal/claude/gsm8k_sc_completion_results_claude.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            gsm8k_id_processed.add((d['id'], d['model']))

prm800k_id_processed = set()
if os.path.exists("rebuttal/claude/prm800k_sc_completion_results_claude.jsonl"):
    with open("rebuttal/claude/prm800k_sc_completion_results_claude.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            prm800k_id_processed.add((d['question'], d['model']))

print(f"Number of results that have been intermediately processed: {len(scli5_id_processed)}")
print(f"Number of results that have been intermediately processed: {len(gsm8k_id_processed)}")
print(f"Number of results that have been intermediately processed: {len(prm800k_id_processed)}")


anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def anthropic_client_process_item(item_tuple, return_question=False):
    model, _, d = item_tuple

    key_model_msg = 'messages_error_injection_in_model' if 'messages_error_injection_in_model' in d else 'messages_error_injection_in_model_bca'
    key_user_msg = 'messages_error_in_user' if 'messages_error_in_user' in d else 'messages_error_in_user_bca'
    key_model_response = 'response_error_injection_in_model' if 'response_error_injection_in_model' in d else 'response_error_injection_in_model_bca'
    key_user_response = 'response_error_in_user' if 'response_error_in_user' in d else 'response_error_in_user_bca'

    response_error_injection_in_model = anthropic_client.messages.create(
        model=model,
        messages=d[key_model_msg],
        max_tokens=1024,
        temperature=0.0,
    )

    response_error_in_user = anthropic_client.messages.create(
        model=model,
        messages=d[key_user_msg],
        max_tokens=1024,
        temperature=0.0,
    )
    response_error_injection_in_model = response_error_injection_in_model.content[0].text if response_error_injection_in_model.content else ""
    response_error_in_user = response_error_in_user.content[0].text if response_error_in_user.content else ""
    if return_question:
        return {
            "question": d['question'],
            "model": model,
            f"{key_model_msg}": d[key_model_msg],
            f"{key_user_msg}": d[key_user_msg],
            f"{key_model_response}": response_error_injection_in_model,
            f"{key_user_response}": response_error_in_user,
            "temperature": 0.0,
        }
    else:
        return {
            "id": d['id'],
            "model": model,
            f"{key_model_msg}": d[key_model_msg],
            f"{key_user_msg}": d[key_user_msg],
            f"{key_model_response}": response_error_injection_in_model,
            f"{key_user_response}": response_error_in_user,
            "temperature": 0.0,
        }



scli5_data = load_dataset("super-brown/scli5", split="test")
gsm8k_data = load_dataset("super-brown/gsm8k_sc", split="test")
prm800k_data = load_dataset("super-brown/prm800k_sc", split="test")


# https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response
with open("rebuttal/claude/scli5_completion_results_claude.jsonl", "a") as f:
    for model in MODEL_ANTHROPIC:
        for d in tqdm(scli5_data, desc="Processing scli5 data"):
            if (d['id'], model) in scli5_id_processed:
                continue
            if (d['id'], model) in scli5_id_hashmap:
                r = scli5_id_hashmap[(d['id'], model)]
                result = {
                    "id": r['id'],
                    "model": r['model'],
                    "messages_error_injection_in_model": r['messages_error_injection_in_model'],
                    "messages_error_in_user": r['messages_error_in_user'],
                    "response_error_injection_in_model": r['response_error_injection_in_model'],
                    "response_error_in_user": r['response_error_in_user'],
                    "temperature": 0.0,
                }
            else:        
                result = anthropic_client_process_item((model, None, d))
            if result:
                f.write(json.dumps(result) + "\n")


with open("rebuttal/claude/gsm8k_sc_completion_results_claude.jsonl", "a") as f:
    for model in MODEL_ANTHROPIC:
        for d in tqdm(gsm8k_data, desc="Processing gsm8k data"):
            if (d['id'], model) in gsm8k_id_processed:
                continue
            if (d['id'], model) in gsm8k_id_hashmap:
                r = gsm8k_id_hashmap[(d['id'], model)]
                result = {
                    "id": r['id'],
                    "model": r['model'],
                    "messages_error_injection_in_model_bca": r['messages_error_injection_in_model_bca'],
                    "messages_error_in_user_bca": r['messages_error_in_user_bca'],
                    "response_error_injection_in_model_bca": r['response_error_injection_in_model_bca'],
                    "response_error_in_user_bca": r['response_error_in_user_bca'],
                    "temperature": 0.0,
                }
            else:        
                result = anthropic_client_process_item((model, None, d))
            if result:
                f.write(json.dumps(result) + "\n")


with open("rebuttal/claude/prm800k_sc_completion_results_claude.jsonl", "a") as f:
    for model in MODEL_ANTHROPIC:
        for d in tqdm(prm800k_data, desc="Processing prm800k data"):
            if (d['question'], model) in prm800k_id_processed:
                continue
            if (d['question'], model) in prm800k_id_hashmap:
                r = prm800k_id_hashmap[(d['question'], model)]
                result = {
                    "question": r['question'],
                    "model": r['model'],
                    "messages_error_injection_in_model_bca": r['messages_error_injection_in_model_bca'],
                    "messages_error_in_user_bca": r['messages_error_in_user_bca'],
                    "response_error_injection_in_model_bca": r['response_error_injection_in_model_bca'],
                    "response_error_in_user_bca": r['response_error_in_user_bca'],
                    "temperature": 0.0,
                }
            else:        
                result = anthropic_client_process_item((model, None, d), return_question=True)
            if result:
                f.write(json.dumps(result) + "\n")
