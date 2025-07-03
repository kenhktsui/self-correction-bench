import json
import pandas as pd
from plot.constants import NON_REASONING_MODELS
from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from evaluation.evaluate_tool import get_is_correct_answer


def load_scli5_eval_data_markers():
    data_with_llm_eval = []
    record_hash = set()
    with open("scli5_completion_results_llm_eval_gemini2_5_flash_markers.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)

            # Ignore counting_letter and counting_digit as counting needs reasoning
            if d['question_type'] in ["counting_letter", "counting_digit"]:
                continue

            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            key = str(d['id']) + "_" + d['model']
            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash.add(key)
    return data_with_llm_eval


def load_gsm8k_sc_eval_data_markers():
    PROBLEM_ID_WITH_ERROR = [467, 361, 499, 857, 962, 1001]

    data_with_llm_eval = []
    record_hash = set()
    with open(f"gsm8k_sc_completion_results_llm_eval_gemini2_5_flash_markers.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']

            if d['id'] in PROBLEM_ID_WITH_ERROR:
                continue

            key = str(d['id']) + "_" + d['model']
            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash.add(key)
    return data_with_llm_eval


def load_prm800k_sc_eval_data_markers():
    data_with_llm_eval = []
    record_hash = set()
    with open("prm800k_sc_completion_results_llm_eval_gemini2_5_flash_markers.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            key = str(d['question']) + "_" + d['model']

            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash.add(key)

    return data_with_llm_eval


data_with_llm_eval = load_scli5_eval_data(temperature=0.0)
df_with_llm_eval = pd.DataFrame([[d["model"], 
                                get_is_correct_answer(d, "llm_evaluation"), 
                                get_is_correct_answer(d, "llm_evaluation_error_in_user"), 
                                get_is_correct_answer(d, "llm_evaluation_wait"), 
                                ] 
                                for d in data_with_llm_eval],
                                columns=["model", 
                                        "error_in_model",
                                        "error_in_user",
                                        "error_in_model_wait"])

data_with_llm_eval_markers = load_scli5_eval_data_markers()
df_with_llm_eval_markers = pd.DataFrame([[d["model"], 
                                get_is_correct_answer(d, "llm_evaluation_but"), 
                                get_is_correct_answer(d, "llm_evaluation_however"), 
                                ] 
                                for d in data_with_llm_eval_markers],
                                columns=["model", 
                                        "error_in_model_but", 
                                        "error_in_model_however"])
scli5_summary = pd.concat([df_with_llm_eval.groupby("model").mean(), df_with_llm_eval_markers.groupby("model").mean()], axis=1).reindex(NON_REASONING_MODELS)
print("scli5_summary")
print(pd.concat([scli5_summary.mean(axis=0), scli5_summary.mean(axis=0)/scli5_summary.mean(axis=0).loc['error_in_model'] - 1], axis=1))


data_with_llm_eval = load_gsm8k_sc_eval_data(temperature=0.0)
df_with_llm_eval = pd.DataFrame([[d["model"], 
                                get_is_correct_answer(d, "llm_evaluation_bca"), 
                                get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                ] 
                                for d in data_with_llm_eval],
                                columns=["model", 
                                        "error_in_model_bca",
                                        "error_in_user_bca",
                                        "error_in_model_bca_wait"])
data_with_llm_eval_markers = load_gsm8k_sc_eval_data_markers()
df_with_llm_eval_markers = pd.DataFrame([[d["model"], 
                                get_is_correct_answer(d, "llm_evaluation_bca_but"),
                                get_is_correct_answer(d, "llm_evaluation_bca_however"),
                                ] 
                                for d in data_with_llm_eval_markers],
                                columns=["model", 
                                        "error_in_model_bca_but",
                                        "error_in_model_bca_however"])
gsm8k_summary = pd.concat([df_with_llm_eval.groupby("model").mean(), df_with_llm_eval_markers.groupby("model").mean()], axis=1).reindex(NON_REASONING_MODELS)
print("gsm8k_sc_summary")
print(pd.concat([gsm8k_summary.mean(axis=0), gsm8k_summary.mean(axis=0)/gsm8k_summary.mean(axis=0).loc['error_in_model_bca'] - 1], axis=1))


data_with_llm_eval = load_prm800k_sc_eval_data(temperature=0.0)
df_with_llm_eval = pd.DataFrame([[d["model"], 
                                get_is_correct_answer(d, "llm_evaluation_bca"), 
                                get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                ] 
                                for d in data_with_llm_eval],
                                columns=["model", 
                                        "error_in_model_bca",
                                        "error_in_user_bca",
                                        "error_in_model_bca_wait"])
data_with_llm_eval_markers = load_prm800k_sc_eval_data_markers()
df_with_llm_eval_markers = pd.DataFrame([[d["model"], 
                                get_is_correct_answer(d, "llm_evaluation_bca_but"),
                                get_is_correct_answer(d, "llm_evaluation_bca_however"),
                                ] 
                                for d in data_with_llm_eval_markers],
                                columns=["model", 
                                        "error_in_model_bca_but",
                                        "error_in_model_bca_however"])
prm800k_summary = pd.concat([df_with_llm_eval.groupby("model").mean(), df_with_llm_eval_markers.groupby("model").mean()], axis=1).reindex(NON_REASONING_MODELS)
print("prm800k_sc_summary")
print(pd.concat([prm800k_summary.mean(axis=0), prm800k_summary.mean(axis=0)/prm800k_summary.mean(axis=0).loc['error_in_model_bca'] - 1], axis=1))

