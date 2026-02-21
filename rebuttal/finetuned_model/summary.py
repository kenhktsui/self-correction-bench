import json
import pandas as pd
from evaluation.evaluate_tool import get_is_correct_answer


def load_eval_data(path):
    PROBLEM_ID_WITH_ERROR = [467, 361, 499, 857, 962, 1001]

    data_with_llm_eval = []
    record_hash = set()
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']

            if d['dataset'] == "gsm8k" and d['id'] in PROBLEM_ID_WITH_ERROR:
                continue

            data_with_llm_eval.append(d)
    return data_with_llm_eval


if __name__ == "__main__":
    data_with_llm_eval = load_eval_data("rebuttal/finetuned_model/finetuned_model_deepinfra_llm_eval.jsonl") + load_eval_data("rebuttal/finetuned_model/finetuned_model_featherless_llm_eval.jsonl")
    df_with_llm_eval_summary = pd.DataFrame([[d["dataset"], d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation") or get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user") or get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    ] 
                                    for d in data_with_llm_eval],
                                    columns=["dataset", "model", 
                                            "error_in_model",
                                            "error_in_user",
                                            ])

    df_with_llm_eval_summary = df_with_llm_eval_summary.groupby(["model", "dataset"])[['error_in_model', 'error_in_user']].mean()
    df_with_llm_eval_summary['Blind Spot'] = (df_with_llm_eval_summary['error_in_user'] - df_with_llm_eval_summary['error_in_model'])/df_with_llm_eval_summary['error_in_user']

    df_with_llm_eval_summary = df_with_llm_eval_summary.reindex(['deepseek-ai/DeepSeek-R1-Distill-Llama-8B_thinking', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B_thinking'], level=0)
    df_with_llm_eval_summary = df_with_llm_eval_summary.reindex(['scli5', 'gsm8k', 'prm800k'], level=1)
    df_with_llm_eval_summary['Blind Spot (Base Model)'] = [0.828947,  0.970794,  0.888889, 0.457746,  0.688525, 0.316770]
    df_with_llm_eval_summary.reset_index(inplace=True)
    df_with_llm_eval_summary['model'] = df_with_llm_eval_summary['model'].str.split('/').str[1]
    print(df_with_llm_eval_summary)
    print((df_with_llm_eval_summary.reset_index()[['model', 'dataset', 'error_in_user', 'error_in_model',  'Blind Spot', 'Blind Spot (Base Model)']]).round(4).astype(str).to_markdown(index=False))
    print((df_with_llm_eval_summary.reset_index()[['model', 'dataset', 'error_in_user', 'error_in_model',  'Blind Spot', 'Blind Spot (Base Model)']]).round(3).astype(str).to_latex(index=False))
    print(df_with_llm_eval_summary['Blind Spot'].mean()/df_with_llm_eval_summary['Blind Spot (Base Model)'].mean()-1)
