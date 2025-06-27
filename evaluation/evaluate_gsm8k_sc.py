import json
import pandas as pd
from evaluate_tool import get_is_correct_answer


def load_gsm8k_sc_eval_data():
    PROBLEM_ID_WITH_ERROR = [467, 361, 499, 857, 962, 1001]

    data_with_llm_eval = []
    record_hash = set()
    with open("gsm8k_sc_completion_results_llm_eval_gemini2_5_flash.jsonl", "r") as f:
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


all_types = [
    "Problem Representation Errors",
    "Planning Errors",
    "Execution Errors"
]


error_type_map = {
    'Calculation Error': 'Execution Errors', 
    'Planning Errors': 'Planning Errors', 
    'Execution Error': 'Execution Errors', 
    'Execution Errors': 'Execution Errors', 
    'Problem Representation Errors': 'Problem Representation Errors',
    'Computation Error': 'Execution Errors',
    'Problem Representation Error': 'Problem Representation Errors',
    'Planning Error': 'Planning Errors',
    'Arithmetic Error': 'Execution Errors',
    'Sign Error': 'Execution Errors',
    'Execution Error (Calculation Error)': 'Execution Errors',
    'Unit Conversion Error': 'Execution Errors',
    'Execution Error (Calculation Mistake)': 'Execution Errors',
    'Omission Error': 'Problem Representation Errors',
    'Problem Representation Error (Misapplication of Percentage Increases)': 'Problem Representation Errors',
    'Planning Error: Incorrect Operation': 'Planning Errors',
    'Arithmetic Sign Error': 'Execution Errors',
    'Arithmetic/Logical Error': 'Execution Errors',
    'Operation Error': 'Execution Errors',
    'Planning Errors: Incorrect operation': 'Planning Errors',
    'Execution Error (Subtraction Order)': 'Execution Errors'
}

if __name__ == "__main__":
    data_with_llm_eval = load_gsm8k_sc_eval_data()
    all_models = sorted(list(set([d['model'] for d in data_with_llm_eval])))
    df_with_llm_eval = pd.DataFrame([[d["model"], 
                                    d["mistake_type"],
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_aca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                    get_is_correct_answer(d, "llm_evaluation_aca_wait"),
                                    ] 
                                    for d in data_with_llm_eval],
                                    columns=["model", 
                                            "error_type",
                                            "error_in_model_bca",
                                            "error_in_model_aca", 
                                            "error_in_user_bca",
                                            "error_in_user_aca", 
                                            "error_in_model_bca_wait",
                                            "error_in_model_aca_wait"])
    df_with_llm_eval["error_type"] = df_with_llm_eval["error_type"].map(error_type_map)


    print('Size of each model')
    print(df_with_llm_eval.groupby("model").size())

    print('Accuracy of each model')
    print((df_with_llm_eval.groupby("model").mean(numeric_only=True)).to_markdown())
    print((df_with_llm_eval.groupby("model").mean(numeric_only=True)).mean().to_markdown())


    print((df_with_llm_eval.groupby("error_type").mean(numeric_only=True)).to_markdown())


    print((df_with_llm_eval.groupby("model").mean(numeric_only=True)[['error_in_model_bca', 'error_in_model_aca', 'error_in_user_bca', 'error_in_user_aca']]).round(4).astype(str).to_latex())


    df_with_llm_eval_bs = df_with_llm_eval.groupby("model").mean(numeric_only=True)[['error_in_model_bca', 'error_in_user_bca', 'error_in_model_aca', 'error_in_user_aca']]
    df_with_llm_eval_bs['Blind Spot bca'] = (df_with_llm_eval_bs['error_in_user_bca'] - df_with_llm_eval_bs['error_in_model_bca'])/df_with_llm_eval_bs['error_in_user_bca']
    df_with_llm_eval_bs['Blind Spot aca'] = (df_with_llm_eval_bs['error_in_user_aca'] - df_with_llm_eval_bs['error_in_model_aca'])/df_with_llm_eval_bs['error_in_user_aca']

    print((df_with_llm_eval_bs[['error_in_model_bca', 'error_in_user_bca', 'Blind Spot bca']]).round(4).astype(str).to_latex())
    print((df_with_llm_eval_bs[['error_in_model_aca', 'error_in_user_aca', 'Blind Spot aca']]).round(4).astype(str).to_latex())
