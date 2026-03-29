import json
import pandas as pd
from evaluation.evaluate_tool import get_is_correct_answer



if __name__ == "__main__":
    data_with_llm_eval = []
    with open(f"rebuttal/domain/tracking_shuffled_objects_completion_results_api_llm_eval.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)

            d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
            data_with_llm_eval.append(d)

    df_with_llm_eval_domain = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                    ] 
                                    for d in data_with_llm_eval],
                                    columns=["model", 
                                            "error_in_model",
                                            "error_in_user",
                                            "error_in_model_wait",
                                            ])

    df_with_llm_eval_domain, size = df_with_llm_eval_domain.groupby(["model"])[['error_in_model', 'error_in_user', 'error_in_model_wait']].mean(), df_with_llm_eval_domain.groupby(["model"])[['error_in_model', 'error_in_user', 'error_in_model_wait']].size()
    df_with_llm_eval_domain['Blind Spot'] = (df_with_llm_eval_domain['error_in_user'] - df_with_llm_eval_domain['error_in_model'])/df_with_llm_eval_domain['error_in_user']
    df_with_llm_eval_domain['Size'] = size.reindex(df_with_llm_eval_domain.index)
    print(df_with_llm_eval_domain)
    print((df_with_llm_eval_domain.reset_index()[['model', 'error_in_model', 'error_in_user', 'error_in_model_wait', 'Blind Spot']]).round(4).astype(str).to_latex())


    def summary(path):
        data_with_llm_eval = []
        with open(path, "r") as f:
            for line in f:
                d = json.loads(line)

                d['model'] = d['model'] + '_thinking' if d.get('enable_thinking') else d['model']
                data_with_llm_eval.append(d)

        df_with_llm_eval_domain = pd.DataFrame([[d["model"], 
                                        get_is_correct_answer(d, "llm_evaluation_bca"), 
                                        get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                        get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                        ] 
                                        for d in data_with_llm_eval],
                                        columns=["model", 
                                                "error_in_model",
                                                "error_in_user",
                                                "error_in_model_wait",
                                                ])

        df_with_llm_eval_domain, size = df_with_llm_eval_domain.groupby(["model"])[['error_in_model', 'error_in_user']].mean(), df_with_llm_eval_domain.groupby(["model"])[['error_in_model', 'error_in_user']].size()
        df_with_llm_eval_domain['Blind Spot'] = (df_with_llm_eval_domain['error_in_user'] - df_with_llm_eval_domain['error_in_model'])/df_with_llm_eval_domain['error_in_user']
        df_with_llm_eval_domain['Size'] = size.reindex(df_with_llm_eval_domain.index)

        df_with_llm_eval_domain = df_with_llm_eval_domain.reindex(['meta-llama/Meta-Llama-3.1-8B-Instruct', 'Qwen/Qwen3-14B', 'mistralai/Mistral-Small-24B-Instruct-2501', 'meta-llama/Llama-3.3-70B-Instruct', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'], level=0)
        df_with_llm_eval_domain.reset_index(inplace=True)
        df_with_llm_eval_domain = df_with_llm_eval_domain[df_with_llm_eval_domain['model'] != 'Qwen/Qwen2.5-7B-Instruct']
        return df_with_llm_eval_domain

    
tracking_shuffled_objects_summary = summary("rebuttal/domain/tracking_shuffled_objects_completion_results_api_llm_eval.jsonl")
logic_deduction_summary = summary("rebuttal/domain/logic_deduction_completion_results_api_llm_eval.jsonl")

tracking_shuffled_objects_summary['dataset'] = 'Tracking shuffled objects'
logic_deduction_summary['dataset'] = 'Logical deduction'

all_summary = pd.concat([tracking_shuffled_objects_summary, logic_deduction_summary]).round(4)[['model', 'dataset', 'error_in_user',  'error_in_model', 'Blind Spot', 'Size']]
all_summary.set_index(['model', 'dataset'], inplace=True)
all_summary = all_summary.reindex(['Logical deduction', 'Tracking shuffled objects'], level=1)
all_summary = all_summary.reindex(['meta-llama/Meta-Llama-3.1-8B-Instruct', 'Qwen/Qwen3-14B', 'mistralai/Mistral-Small-24B-Instruct-2501', 'meta-llama/Llama-3.3-70B-Instruct', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'], level=0)
all_summary = all_summary.reset_index()
all_summary['model'] = all_summary['model'].str.split('/').str[1]

print(all_summary.to_markdown(index=False))
print(all_summary.round(3).astype(str).to_latex(index=False))
print(all_summary[all_summary['model'] != 'DeepSeek-R1-Distill-Llama-70B']['Blind Spot'].mean())
print(all_summary[all_summary['model'] == 'DeepSeek-R1-Distill-Llama-70B']['Blind Spot'].mean()/all_summary[all_summary['model'] == 'Llama-3.3-70B-Instruct']['Blind Spot'].mean()-1)
