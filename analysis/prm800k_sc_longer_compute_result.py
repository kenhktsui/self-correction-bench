import pandas as pd
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from evaluation.evaluate_tool import get_is_correct_answer
from plot.constants import NON_REASONING_MODELS


def short_vs_long_compute_result():
    data_short_compute = load_prm800k_sc_eval_data(supplement=False)
    data_long_compute = load_prm800k_sc_eval_data(supplement=True)

    prm800k_short_compute_df = pd.DataFrame([[d["model"], get_is_correct_answer(d, "llm_evaluation_bca"), get_is_correct_answer(d, "llm_evaluation_aca"),
                                get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                get_is_correct_answer(d, "llm_evaluation_bca_wait"), get_is_correct_answer(d, "llm_evaluation_aca_wait")] 
                              for d in data_short_compute],
                             columns=["model", "error_in_model_bca", "error_in_model_aca", "error_in_user_bca", "error_in_user_aca",
                                      "error_in_model_bca_wait", "error_in_model_aca_wait"])

    prm800k_long_compute_df = pd.DataFrame([[d["model"], get_is_correct_answer(d, "llm_evaluation_bca"), get_is_correct_answer(d, "llm_evaluation_aca"),
                                get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"), get_is_correct_answer(d, "llm_evaluation_error_in_user_aca"),
                                get_is_correct_answer(d, "llm_evaluation_bca_wait"), get_is_correct_answer(d, "llm_evaluation_aca_wait")] 
                              for d in data_long_compute],
                             columns=["model", "error_in_model_bca", "error_in_model_aca", "error_in_user_bca", "error_in_user_aca",
                                      "error_in_model_bca_wait", "error_in_model_aca_wait"])

    prm800k_short_compute_df = prm800k_short_compute_df.groupby("model")[["error_in_user_bca", "error_in_model_bca", "error_in_model_bca_wait"]].mean()
    prm800k_long_compute_df = prm800k_long_compute_df.groupby("model")[["error_in_user_bca", "error_in_model_bca", "error_in_model_bca_wait"]].mean()

    prm800k_short_compute_df.columns = ["error_in_user_bca_short", "error_in_model_bca_short", "error_in_model_bca_wait_short"]
    prm800k_long_compute_df.columns = ["error_in_user_bca_long", "error_in_model_bca_long", "error_in_model_bca_wait_long"]

    summary_df = pd.concat([prm800k_short_compute_df, prm800k_long_compute_df], axis=1)
    summary_df = summary_df.reindex(NON_REASONING_MODELS)

    summary_df = summary_df[["error_in_user_bca_short", "error_in_user_bca_long",
                             "error_in_model_bca_short", "error_in_model_bca_long",
                             "error_in_model_bca_wait_short", "error_in_model_bca_wait_long"]]
    return summary_df


if __name__ == "__main__":
    summary_df = short_vs_long_compute_result()
    print(summary_df[["error_in_user_bca_short", "error_in_user_bca_long"]])
    print(summary_df[["error_in_model_bca_short", "error_in_model_bca_long"]])
    print(summary_df[["error_in_model_bca_wait_short", "error_in_model_bca_wait_long"]])


    print(summary_df[["error_in_user_bca_short", "error_in_user_bca_long", "error_in_model_bca_short", "error_in_model_bca_long", "error_in_model_bca_wait_short", "error_in_model_bca_wait_long"]].round(3).astype(str).to_latex())
