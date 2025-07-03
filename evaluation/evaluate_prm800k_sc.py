import json
import pandas as pd
from evaluation.evaluate_tool import get_is_correct_answer


def load_prm800k_sc_eval_data(temperature=0.0, supplement=False):
    data_with_llm_eval = []
    record_hash = {}
    with open(f"prm800k_sc_completion_results_llm_eval_gemini2_5_flash_{str(temperature).replace('.', '_')}.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            key = str(d['question']) + "_" + d['model']

            if key not in record_hash:
                data_with_llm_eval.append(d)
                record_hash[key] = d

    # load supplement data (with limit of 4,096 output token) and override the original data with a limit of output token length >= 1024
    if supplement:
        with open(f"prm800k_sc_completion_results_llm_eval_gemini2_5_flash_supplement_{str(temperature).replace('.', '_')}.jsonl", "r") as f:
            for line in f:
                d = json.loads(line)
                d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
                key = str(d['question']) + "_" + d['model']
                # retrieve the original record
                orginal_d = record_hash[key]

                for k in  [
                    "response_error_injection_in_model_bca",
                    "response_error_injection_in_model_aca",
                    "response_error_in_user_bca",
                    "response_error_in_user_aca",
                    "response_error_injection_in_model_bca_wait",
                    "response_error_injection_in_model_aca_wait",
                    "llm_evaluation_system_prompt",
                    "llm_evaluation_prompt_bca",
                    "llm_evaluation_bca",
                    "llm_evaluation_prompt_aca",
                    "llm_evaluation_aca",
                    "llm_evaluation_prompt_error_in_user_bca",
                    "llm_evaluation_error_in_user_bca",
                    "llm_evaluation_prompt_error_in_user_aca",
                    "llm_evaluation_error_in_user_aca",
                    "llm_evaluation_prompt_bca_wait",
                    "llm_evaluation_bca_wait",
                    "llm_evaluation_prompt_aca_wait",
                    "llm_evaluation_aca_wait",                    
                ]:
                    if d[k] is not None:
                        orginal_d[k] = d[k]                 
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
