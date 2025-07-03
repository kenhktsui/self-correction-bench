from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from evaluation.evaluate_tool import get_is_correct_answer
import pandas as pd
from plot.constants import NON_REASONING_MODELS
    

def get_scli5_eval_data_with_diff_temperature():
    data_with_llm_eval_00 = load_scli5_eval_data(temperature=0.0)
    df_with_llm_eval_00 = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user"), 
                                    get_is_correct_answer(d, "llm_evaluation_wait"), 
                                    ] 
                                    for d in data_with_llm_eval_00],
                                    columns=["model", 
                                            "error_in_model",
                                            "error_in_user",
                                            "error_in_model_wait"])

    data_with_llm_eval_06 = load_scli5_eval_data(temperature=0.6)
    df_with_llm_eval_06 = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user"), 
                                    get_is_correct_answer(d, "llm_evaluation_wait"), 
                                    ] 
                                    for d in data_with_llm_eval_06],
                                    columns=["model", 
                                            "error_in_model",
                                            "error_in_user",
                                            "error_in_model_wait"])

    # compare the mean accuracy of 0.6 and 0.0
    df_with_llm_eval_bs_00 = df_with_llm_eval_00.groupby("model").mean(numeric_only=True)[['error_in_model', 'error_in_user', 'error_in_model_wait']]
    df_with_llm_eval_bs_06 = df_with_llm_eval_06.groupby("model").mean(numeric_only=True)[['error_in_model', 'error_in_user', 'error_in_model_wait']]
    df_with_llm_eval_bs_00.columns = ['error_in_model_00', 'error_in_user_00', 'error_in_model_wait_00']
    df_with_llm_eval_bs_06.columns = ['error_in_model_06', 'error_in_user_06', 'error_in_model_wait_06']
    df_with_llm_eval_bs_compare = pd.concat([df_with_llm_eval_bs_00, df_with_llm_eval_bs_06], axis=1)
    df_with_llm_eval_bs_compare = df_with_llm_eval_bs_compare[['error_in_model_00', 'error_in_model_06', 'error_in_user_00', 'error_in_user_06', 'error_in_model_wait_00', 'error_in_model_wait_06']]
    df_with_llm_eval_bs_compare = df_with_llm_eval_bs_compare[df_with_llm_eval_bs_compare.index.isin(NON_REASONING_MODELS)].reindex(NON_REASONING_MODELS)
    return df_with_llm_eval_bs_compare


def get_gsm8k_sc_eval_data_with_diff_temperature():
    data_with_llm_eval_00 = load_gsm8k_sc_eval_data(temperature=0.0)
    df_with_llm_eval_00 = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_aca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                    get_is_correct_answer(d, "llm_evaluation_aca_wait"),
                                    ] 
                                    for d in data_with_llm_eval_00],
                                    columns=["model", 
                                            "error_in_model_bca",
                                            "error_in_model_aca", 
                                            "error_in_user_bca",
                                            "error_in_user_aca", 
                                            "error_in_model_bca_wait",
                                            "error_in_model_aca_wait"])

    data_with_llm_eval_06 = load_gsm8k_sc_eval_data(temperature=0.6)
    df_with_llm_eval_06 = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_aca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                    get_is_correct_answer(d, "llm_evaluation_aca_wait"),
                                    ] 
                                    for d in data_with_llm_eval_06],
                                    columns=["model", 
                                            "error_in_model_bca",
                                            "error_in_model_aca", 
                                            "error_in_user_bca",
                                            "error_in_user_aca", 
                                            "error_in_model_bca_wait",
                                            "error_in_model_aca_wait"])

    # compare the mean accuracy of 0.6 and 0.0
    df_with_llm_eval_bs_00 = df_with_llm_eval_00.groupby("model").mean(numeric_only=True)[['error_in_model_bca', 'error_in_user_bca', 'error_in_model_aca', 'error_in_user_aca', 'error_in_model_bca_wait', 'error_in_model_aca_wait']]
    df_with_llm_eval_bs_06 = df_with_llm_eval_06.groupby("model").mean(numeric_only=True)[['error_in_model_bca', 'error_in_user_bca', 'error_in_model_aca', 'error_in_user_aca', 'error_in_model_bca_wait', 'error_in_model_aca_wait']]

    df_with_llm_eval_bs_00.columns = ['error_in_model_bca_00', 'error_in_user_bca_00', 'error_in_model_aca_00', 'error_in_user_aca_00', 'error_in_model_bca_wait_00', 'error_in_model_aca_wait_00']
    df_with_llm_eval_bs_06.columns = ['error_in_model_bca_06', 'error_in_user_bca_06', 'error_in_model_aca_06', 'error_in_user_aca_06', 'error_in_model_bca_wait_06', 'error_in_model_aca_wait_06']

    df_with_llm_eval_bs_compare = pd.concat([df_with_llm_eval_bs_00, df_with_llm_eval_bs_06], axis=1)
    df_with_llm_eval_bs_compare = df_with_llm_eval_bs_compare[['error_in_model_bca_00', 'error_in_model_bca_06', 'error_in_user_bca_00', 'error_in_user_bca_06', 'error_in_model_bca_wait_00', 'error_in_model_bca_wait_06']]
    df_with_llm_eval_bs_compare = df_with_llm_eval_bs_compare[df_with_llm_eval_bs_compare.index.isin(NON_REASONING_MODELS)].reindex(NON_REASONING_MODELS)
    return df_with_llm_eval_bs_compare


def get_prm800k_sc_eval_data_with_diff_temperature():
    data_with_llm_eval_00 = load_prm800k_sc_eval_data(temperature=0.0)
    df_with_llm_eval_00 = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_aca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                    get_is_correct_answer(d, "llm_evaluation_aca_wait"),
                                    ] 
                                    for d in data_with_llm_eval_00],
                                    columns=["model", 
                                            "error_in_model_bca",
                                            "error_in_model_aca", 
                                            "error_in_user_bca",
                                            "error_in_user_aca", 
                                            "error_in_model_bca_wait",
                                            "error_in_model_aca_wait"])

    data_with_llm_eval_06 = load_prm800k_sc_eval_data(temperature=0.6)
    df_with_llm_eval_06 = pd.DataFrame([[d["model"], 
                                    get_is_correct_answer(d, "llm_evaluation_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_aca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), 
                                    get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                                    get_is_correct_answer(d, "llm_evaluation_aca_wait"),
                                    ] 
                                    for d in data_with_llm_eval_06],
                                    columns=["model", 
                                            "error_in_model_bca",
                                            "error_in_model_aca", 
                                            "error_in_user_bca",
                                            "error_in_user_aca", 
                                            "error_in_model_bca_wait",
                                            "error_in_model_aca_wait"])

    # compare the mean accuracy of 0.6 and 0.0
    df_with_llm_eval_bs_00 = df_with_llm_eval_00.groupby("model").mean(numeric_only=True)[['error_in_model_bca', 'error_in_user_bca', 'error_in_model_aca', 'error_in_user_aca', 'error_in_model_bca_wait', 'error_in_model_aca_wait']]
    df_with_llm_eval_bs_06 = df_with_llm_eval_06.groupby("model").mean(numeric_only=True)[['error_in_model_bca', 'error_in_user_bca', 'error_in_model_aca', 'error_in_user_aca', 'error_in_model_bca_wait', 'error_in_model_aca_wait']]

    df_with_llm_eval_bs_00.columns = ['error_in_model_bca_00', 'error_in_user_bca_00', 'error_in_model_aca_00', 'error_in_user_aca_00', 'error_in_model_bca_wait_00', 'error_in_model_aca_wait_00']
    df_with_llm_eval_bs_06.columns = ['error_in_model_bca_06', 'error_in_user_bca_06', 'error_in_model_aca_06', 'error_in_user_aca_06', 'error_in_model_bca_wait_06', 'error_in_model_aca_wait_06']

    df_with_llm_eval_bs_compare = pd.concat([df_with_llm_eval_bs_00, df_with_llm_eval_bs_06], axis=1)
    df_with_llm_eval_bs_compare = df_with_llm_eval_bs_compare[['error_in_model_bca_00', 'error_in_model_bca_06', 'error_in_user_bca_00', 'error_in_user_bca_06', 'error_in_model_bca_wait_00', 'error_in_model_bca_wait_06']]
    df_with_llm_eval_bs_compare = df_with_llm_eval_bs_compare[df_with_llm_eval_bs_compare.index.isin(NON_REASONING_MODELS)].reindex(NON_REASONING_MODELS)
    return df_with_llm_eval_bs_compare


scli5_df = get_scli5_eval_data_with_diff_temperature()
gsm8k_df = get_gsm8k_sc_eval_data_with_diff_temperature()
prm800k_df = get_prm800k_sc_eval_data_with_diff_temperature()

print("scli5")
print(scli5_df)
print("gsm8k")
print(gsm8k_df)
print("prm800k")
print(prm800k_df)


summary_00 = pd.concat([scli5_df['error_in_model_00'], gsm8k_df['error_in_model_bca_00'], prm800k_df['error_in_model_bca_00']], axis=1, keys=['scli5_00', 'gsm8k_00', 'prm800k_00'])
summary_06 = pd.concat([scli5_df['error_in_model_06'], gsm8k_df['error_in_model_bca_06'], prm800k_df['error_in_model_bca_06']], axis=1, keys=['scli5_06', 'gsm8k_06', 'prm800k_06'])
summary_00_blindspot  = pd.concat([
     scli5_df['error_in_user_00'] - scli5_df['error_in_model_00'],
     gsm8k_df['error_in_user_bca_00'] - gsm8k_df['error_in_model_bca_00'],
     prm800k_df['error_in_user_bca_00'] - prm800k_df['error_in_model_bca_00']
     ], axis=1, keys=['scli5_00_blindspot', 'gsm8k_00_blindspot', 'prm800k_00_blindspot'])
summary_06_blindspot  = pd.concat([
     scli5_df['error_in_user_06'] - scli5_df['error_in_model_06'],
     gsm8k_df['error_in_user_bca_06'] - gsm8k_df['error_in_model_bca_06'],
     prm800k_df['error_in_user_bca_06'] - prm800k_df['error_in_model_bca_06']
     ], axis=1, keys=['scli5_06_blindspot', 'gsm8k_06_blindspot', 'prm800k_06_blindspot'])
summary_00_wait_increase  = pd.concat([
     scli5_df['error_in_model_wait_00'] - scli5_df['error_in_model_00'],
     gsm8k_df['error_in_model_bca_wait_00'] - gsm8k_df['error_in_model_bca_00'],
     prm800k_df['error_in_model_bca_wait_00'] - prm800k_df['error_in_model_bca_00']
     ], axis=1, keys=['scli5_00_wait_increase', 'gsm8k_00_wait_increase', 'prm800k_00_wait_increase'])
summary_06_wait_increase  = pd.concat([
     scli5_df['error_in_model_wait_06'] - scli5_df['error_in_model_06'],
     gsm8k_df['error_in_model_bca_wait_06'] - gsm8k_df['error_in_model_bca_06'],
     prm800k_df['error_in_model_bca_wait_06'] - prm800k_df['error_in_model_bca_06']
     ], axis=1, keys=['scli5_06_wait_increase', 'gsm8k_06_wait_increase', 'prm800k_06_wait_increase'])

print('='*100)
print("Blindspot between Temperature=0.0")
print(summary_00_blindspot)
print('='*100)
print("Blindspot between Temperature=0.6")
print(summary_06_blindspot)
print('='*100)
print("Wait increase in Temperature=0.0")
print(summary_00_wait_increase)
print('='*100)
print("Wait increase in Temperature=0.6")
print(summary_06_wait_increase)


summary_diff = pd.concat([
    summary_06['scli5_06'] - summary_00['scli5_00'],
    summary_06['gsm8k_06'] - summary_00['gsm8k_00'],
    summary_06['prm800k_06'] - summary_00['prm800k_00']
    ], axis=1, keys=['scli5_diff', 'gsm8k_diff', 'prm800k_diff'])

print('='*100)
print("Summary of Temperature=0.0")
print(summary_00.round(3).astype(str).to_latex())
print('='*100)
print("Summary of Temperature=0.6")
print(summary_06.round(3).astype(str).to_latex())
print('='*100)
print("Difference between Temperature=0.6 and Temperature=0.0")
print(summary_diff.round(3))
