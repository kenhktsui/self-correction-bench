import json
import pandas as pd
from evaluate_tool import get_is_correct_answer


def load_scli5_eval_data():
    data_with_llm_eval = []
    record_hash = set()
    with open("scli5_completion_results_llm_eval_gemini2_5_flash.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            key = str(d['id']) + "_" + d['model']
            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash.add(key)
    return data_with_llm_eval


all_types = [
    "get_add_one",
    "get_sub_one",
    "get_next_character",
    "get_previous_character",
    "get_larger_number",
    "get_smaller_number",
    "counting_letter",
    "counting_digit"
]

if __name__ == "__main__":
    data_with_llm_eval = load_scli5_eval_data()
    all_models = sorted(list(set([d['model'] for d in data_with_llm_eval])))
    df_with_llm_eval = pd.DataFrame([[d["model"], 
                                    d["question_type"],
                                    get_is_correct_answer(d, "llm_evaluation"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user"), 
                                    get_is_correct_answer(d, "llm_evaluation_wait"), 
                                    get_is_correct_answer(d, "llm_evaluation_cot"), 
                                    ] 
                                    for d in data_with_llm_eval],
                                    columns=["model", 
                                            "question_type",
                                            "error_in_model",
                                            "error_in_user",
                                            "error_in_model_wait",
                                            "error_in_model_cot"])


    print('Size of each model')
    print(df_with_llm_eval.groupby("model").size())


    print('Accuracy of each model')
    print((df_with_llm_eval.groupby("model").mean(numeric_only=True)).round(2).to_markdown())
    print((df_with_llm_eval.groupby("model").mean(numeric_only=True)).round(2).mean().to_markdown())


    print((df_with_llm_eval.groupby(["model", "question_type"]).mean()).round(2))

    df_with_llm_eval_bs = df_with_llm_eval.groupby("model").mean(numeric_only=True)[['error_in_model', 'error_in_user']]
    df_with_llm_eval_bs['Blind Spot'] = (df_with_llm_eval_bs['error_in_user'] - df_with_llm_eval_bs['error_in_model'])/df_with_llm_eval_bs['error_in_user']
    print((df_with_llm_eval_bs[['error_in_model', 'error_in_user', 'Blind Spot']]).round(4).astype(str).to_latex())
