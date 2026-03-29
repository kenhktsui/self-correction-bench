import json
import pandas as pd
from evaluation.evaluate_tool import get_is_correct_answer


def load_scli5_eval_data(temperature=0.0):
    data_with_llm_eval = []
    record_hash = set()
    with open(f"rebuttal/claude/scli5_completion_results_claude_llm_eval.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)

            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            key = str(d['id']) + "_" + d['model']
            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash.add(key)
    return data_with_llm_eval


def load_gsm8k_sc_eval_data(temperature=0.0):
    PROBLEM_ID_WITH_ERROR = [467, 361, 499, 857, 962, 1001]

    data_with_llm_eval = []
    record_hash = set()
    with open(f"rebuttal/claude/gsm8k_sc_completion_results_claude_llm_eval.jsonl", "r") as f:
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


def load_prm800k_sc_eval_data(temperature=0.0, supplement=False):
    data_with_llm_eval = []
    record_hash = {}
    with open("rebuttal/claude/prm800k_sc_completion_results_claude_llm_eval.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            key = str(d['question']) + "_" + d['model']

            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash[key] = d
       
    return data_with_llm_eval



if __name__ == "__main__":
    data_with_llm_eval = load_scli5_eval_data()
    df_with_llm_eval_scli5 = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user"), 
                                    ] 
                                    for d in data_with_llm_eval],
                                    columns=["model", 
                                            "error_in_model",
                                            "error_in_user",
                                            ])
    df_with_llm_eval_scli5['dataset'] = 'SCLI5'

    data_with_llm_eval = load_gsm8k_sc_eval_data(temperature=0.0)
    df_with_llm_eval_gsm8k_sc = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    ] 
                                    for d in data_with_llm_eval],
                                    columns=["model", 
                                            "error_in_model",
                                            "error_in_user",
                                            ])
    df_with_llm_eval_gsm8k_sc['dataset'] = 'GSM8k-SC'

    data_with_llm_eval = load_prm800k_sc_eval_data()
    df_with_llm_eval_prm800k_sc = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"),
                                  ] 
                                  for d in data_with_llm_eval],
                                  columns=["model", 
                                           "error_in_model",
                                           "error_in_user",
                                           ])
    df_with_llm_eval_prm800k_sc['dataset'] = 'PRM800K-SC'

    df_with_llm_eval_summary = pd.concat([df_with_llm_eval_scli5, df_with_llm_eval_gsm8k_sc, df_with_llm_eval_prm800k_sc], axis=0)

    df_with_llm_eval_summary = df_with_llm_eval_summary.groupby(["dataset", "model"])[['error_in_model', 'error_in_user']].mean()
    df_with_llm_eval_summary['Blind Spot'] = (df_with_llm_eval_summary['error_in_user'] - df_with_llm_eval_summary['error_in_model'])/df_with_llm_eval_summary['error_in_user']

    df_with_llm_eval_summary.reset_index(inplace=True)
    df_with_llm_eval_summary.set_index(['model', 'dataset'], inplace=True)
    df_with_llm_eval_summary = df_with_llm_eval_summary.reindex(['claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514'], level=0)
    df_with_llm_eval_summary = df_with_llm_eval_summary.reindex(['SCLI5', 'GSM8k-SC', 'PRM800K-SC'], level=1)
    df_with_llm_eval_summary = df_with_llm_eval_summary.reset_index()
    df_with_llm_eval_summary = df_with_llm_eval_summary[['model', 'dataset', 'error_in_user', 'error_in_model', 'Blind Spot']]
    print((df_with_llm_eval_summary).round(4).astype(str).to_markdown(index=False))
    print((df_with_llm_eval_summary).round(3).astype(str).to_latex(index=False))
    print(df_with_llm_eval_summary.groupby("model")['Blind Spot'].mean())
