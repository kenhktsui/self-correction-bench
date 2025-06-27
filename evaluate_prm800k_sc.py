import json
import pandas as pd
from evaluate_tool import get_is_correct_answer


def load_prm800k_sc_eval_data():
    data_with_llm_eval = []
    record_hash = set()
    with open("prm800k_sc_completion_results_llm_eval_gemini2_5_flash.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            key = str(d['question']) + "_" + d['model']

            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash.add(key)

    return data_with_llm_eval

if __name__ == "__main__":
    data_with_llm_eval = load_prm800k_sc_eval_data()
    all_models = sorted(list(set([d['model'] for d in data_with_llm_eval])))
    df_with_llm_eval = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_aca"),
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"),
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),                                   
                                    get_is_correct_answer(d, "llm_evaluation_aca_wait"), 
                                    get_is_correct_answer(d, "llm_evaluation_bca_cot"),
                                    get_is_correct_answer(d, "llm_evaluation_aca_cot"),
                                  ] 
                                  for d in data_with_llm_eval],
                                  columns=["model", 
                                           "error_in_model_bca",
                                           "error_in_model_aca",
                                           "error_in_user_bca",
                                           "error_in_user_aca",
                                           "error_in_model_wait_bca",
                                           "error_in_model_wait_aca",
                                           "error_in_model_cot_bca",
                                           "error_in_model_cot_aca",
                                           ])


    print('Size of each model')
    print(df_with_llm_eval.groupby("model").size())

    print('Accuracy of each model')
    print((df_with_llm_eval.groupby("model").mean(numeric_only=True)).round(2).to_markdown())


    print('Accuracy')
    print((df_with_llm_eval.groupby("model").mean(numeric_only=True).mean(numeric_only=True)).round(2).to_markdown())

    df_with_llm_eval_bs = df_with_llm_eval.groupby("model").mean(numeric_only=True)[['error_in_model_bca', 'error_in_user_bca', 'error_in_model_aca', 'error_in_user_aca']]
    df_with_llm_eval_bs['Blind Spot bca'] = (df_with_llm_eval_bs['error_in_user_bca'] - df_with_llm_eval_bs['error_in_model_bca'])/df_with_llm_eval_bs['error_in_user_bca']
    df_with_llm_eval_bs['Blind Spot aca'] = (df_with_llm_eval_bs['error_in_user_aca'] - df_with_llm_eval_bs['error_in_model_aca'])/df_with_llm_eval_bs['error_in_user_aca']

    print((df_with_llm_eval_bs[['error_in_model_bca', 'error_in_user_bca', 'Blind Spot bca']]).round(4).astype(str).to_latex())
    print((df_with_llm_eval_bs[['error_in_model_aca', 'error_in_user_aca', 'Blind Spot aca']]).round(4).astype(str).to_latex())
