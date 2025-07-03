import json
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from plot.constants import NON_REASONING_MODELS
from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from llm_inference.constants import MODEL_LIST


def get_input_tokens(dataset_name, temperature=0.0, supplement=True):
    data_list = {m: [] for m in NON_REASONING_MODELS}

    if dataset_name == "scli5":
        data = load_scli5_eval_data(temperature=temperature)
    elif dataset_name == "gsm8k_sc":
        data = load_gsm8k_sc_eval_data(temperature=temperature)
    elif dataset_name == "prm800k_sc":
        data = load_prm800k_sc_eval_data(temperature=temperature, supplement=supplement)

    for d in data:
        # excluding reasoning models and thinking mode
        if d['model'] not in NON_REASONING_MODELS or d.get('enable_thinking'):
            continue

        if dataset_name == "scli5":
            text = {
                "prompt_error_in_user": d["prompt_error_injection_in_model"],
                "prompt_error_in_model": d["prompt_error_in_user"],
                "prompt_error_in_model_wait": d["prompt_error_injection_in_model_wait"],
            }
        elif dataset_name in ["gsm8k_sc", "prm800k_sc"]:
            text = {
                "prompt_error_in_user": d["prompt_error_injection_in_model_bca"],
                "prompt_error_in_model": d["prompt_error_in_user_bca"],
                "prompt_error_in_model_wait": d["prompt_error_injection_in_model_bca_wait"],
            }
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        data_list[d['model']].append(text)

    for m in tqdm(NON_REASONING_MODELS):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_LIST[m])
        data_list[m] = Dataset.from_list(data_list[m])
        
        data_list[m] = data_list[m].map(lambda x: {"prompt_error_in_user": tokenizer(x['prompt_error_in_user'], add_special_tokens=False)['input_ids']})
        data_list[m] = data_list[m].map(lambda x: {"prompt_error_in_model": tokenizer(x['prompt_error_in_model'], add_special_tokens=False)['input_ids']})
        data_list[m] = data_list[m].map(lambda x: {"prompt_error_in_model_wait": tokenizer(x['prompt_error_in_model_wait'], add_special_tokens=False)['input_ids']})


    return data_list


if __name__ == "__main__":
    import pandas as pd

    for dataset_name in ["scli5", "gsm8k_sc", "prm800k_sc"]:
        data = get_input_tokens(dataset_name, temperature=0.0, supplement=True)
        for m in NON_REASONING_MODELS:
            data[m] = data[m].to_pandas()
            data[m]['symmetric_diff'] = data[m].apply(
                lambda row: len(set(row['prompt_error_in_user']).symmetric_difference(set(row['prompt_error_in_model']))), axis=1
            )
            data[m]['symmetric_diff_wait'] = data[m].apply(
                lambda row: len(set(row['prompt_error_in_model']).symmetric_difference(set(row['prompt_error_in_model_wait']))), axis=1
            )
        print(dataset_name)
        print(pd.concat([data[m][['symmetric_diff', 'symmetric_diff_wait']] for m in NON_REASONING_MODELS], axis=0).describe())
