import argparse
import os
import re
import json
import re
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
    eval_prompt_template_bca,
    eval_prompt_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate steering results with Gemini LLM judge")
    parser.add_argument("--input", required=True, help="Path to input JSON file (e.g. steering_llama-8b_gsm8k_sc_internal_24.json)")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    parser.add_argument("--thinking_budget", type=int, default=0, help="Gemini thinking budget (default: 0)")
    return parser.parse_args()


def get_eval_prompt_template():
    return {
        "scli5": eval_prompt_template,
        "gsm8k_sc": eval_prompt_template_bca,
        "prm800k_sc": eval_prompt_template_bca,
    }


def get_question_error_and_correct_answer_all():
    messages_error_injection_in_model_all = {}
    correct_answer_all = {}

    dataset = load_dataset("super-brown/scli5", split="test")
    messages_error_injection_in_model_all['scli5'] = {d['id']: d['messages_error_injection_in_model'] for d in dataset}
    correct_answer_all['scli5'] = {d['id']: d['correct_answer'] for d in dataset}
    del dataset

    dataset = load_dataset("super-brown/gsm8k_sc", split="test")
    messages_error_injection_in_model_all['gsm8k_sc'] = {d['id']: d['messages_error_injection_in_model_bca'] for d in dataset}
    correct_answer = {}
    for d in dataset:
        a = d["answer"].split("\n")
        answer = a[-1]
        answer = re.sub(re.escape("####"), "", answer).strip()
        correct_answer[d['id']] = answer
    correct_answer_all['gsm8k_sc'] = correct_answer
    del dataset

    dataset = load_dataset("super-brown/prm800k_sc", split="test")
    messages_error_injection_in_model_all['prm800k_sc'] = {i: d['messages_error_injection_in_model_bca'] for i, d in enumerate(dataset)}
    correct_answer_all['prm800k_sc'] = {i: d['ground_truth_answer'] for i, d in enumerate(dataset)}
    return messages_error_injection_in_model_all, correct_answer_all


def make_evaluate_with_llm(client, thinking_budget):
    def evaluate_with_llm(prompt, response_schema):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=eval_system_prompt,
                temperature=0.0,
                response_mime_type='application/json',
                response_schema=response_schema,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            )
        )
        return response.model_dump()
    return evaluate_with_llm


def make_process_data(dataset, messages_error_injection_in_model, correct_answer,
                      eval_prompt_tmpl, evaluate_fn):
    def process_data(d, alpha):
        d['alpha'] = alpha

        if not d['generated'].strip():
            d['llm_evaluation'] = None
            return d

        d['llm_evaluation_system_prompt'] = eval_system_prompt

        msg_error_model = messages_error_injection_in_model[d['id']]
        question = msg_error_model[0]['content']
        given_wrong = msg_error_model[1]['content']
        correct_ans = correct_answer[d['id']]

        # scli5
        if dataset == 'scli5':
            eval_prompt = eval_prompt_tmpl.format(
                question=question,
                golden_answer=correct_ans,
                given_wrong_answer=given_wrong,
                completion_from_model=re.sub("assistant", "", d['generated']).strip()
            )
        # gsm8k_sc / prm800k_sc
        elif dataset in ['gsm8k_sc', 'prm800k_sc']:
            eval_prompt = eval_prompt_tmpl.format(
                question=question,
                golden_answer=correct_ans,
                given_wrong_reasoning=given_wrong,
                completion_from_model=re.sub("assistant", "", d['generated']).strip()
            )

        d['llm_evaluation_prompt_bca'] = eval_prompt

        response_bca = evaluate_fn(eval_prompt, Evaluation)
        d['llm_evaluation'] = response_bca
        return d
    return process_data


def main():
    args = parse_args()

    if os.path.exists(args.output):
        print(f"File {args.output} already exists.")
    else:
        with open(args.output, "w") as f:
            pass

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
        http_options=types.HttpOptions(api_version='v1alpha', timeout=60 * 10 * 1000)
    )

    with open(args.input, "r") as f:
        data_in = json.load(f)

    dataset = data_in["dataset"]
    assert dataset in ["scli5", "gsm8k_sc", "prm800k_sc"], \
        "dataset must be in 'scli5', 'gsm8k_sc', 'prm800k_sc'"

    eval_prompt_tmpl = get_eval_prompt_template()[dataset]
    print(f"[INFO] Loading reference datasets...")
    messages_error_injection_in_model_all, correct_answer_all = get_question_error_and_correct_answer_all()
    messages_error_injection_in_model = messages_error_injection_in_model_all[dataset]
    correct_answer = correct_answer_all[dataset]

    evaluate_fn = make_evaluate_with_llm(client, args.thinking_budget)
    process_data = make_process_data(
        dataset, messages_error_injection_in_model, correct_answer, eval_prompt_tmpl, evaluate_fn
    )

    all_alphas = list(data_in["internal_sweep"].keys())
    all_alphas = [a for a in all_alphas if float(a) != 0]

    data = []
    for a in all_alphas:
        for e in data_in["internal_sweep"][a]["per_example"]:
            data.append((e, a))

    print(f"[INFO] Evaluating {len(data)} examples from '{dataset}' (alphas: {all_alphas})")

    file_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_data, item, alpha) for item, alpha in data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
            try:
                result = future.result()
                if result:
                    with file_lock:
                        with open(args.output, "a") as f:
                            f.write(json.dumps(result) + "\n")
            except concurrent.futures.TimeoutError:
                print("Future timed out")
            except Exception as e:
                print(f"Error processing future: {e}")

    print(f"[INFO] Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
