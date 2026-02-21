import json
import pandas as pd
import numpy as np
from evaluation.evaluate_tool import get_is_correct_answer



def summary(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame([[d["model"], get_is_correct_answer(d, "llm_evaluation_on_policy_error")] for d in data], columns=["model", "acc_in_on_policy_error"])
    summary = pd.concat([df.groupby("model")['acc_in_on_policy_error'].mean(), df.groupby("model")['acc_in_on_policy_error'].std()/np.sqrt(df.groupby("model")['acc_in_on_policy_error'].size()), df.groupby("model")['acc_in_on_policy_error'].size()], axis=1)
    summary.columns = ["accuracy", "standard_error", "size"]
    return summary.reset_index()

def summary_v2(path_list):
    data = []
    for path in path_list:
        with open(path, "r") as f:
            data.extend([json.loads(line) for line in f])
    df = pd.DataFrame([[d["model"], get_is_correct_answer(d, "llm_evaluation_on_policy_error")] for d in data], columns=["model", "acc_in_on_policy_error"])
    summary = pd.concat([df.groupby("model")['acc_in_on_policy_error'].mean(), df.groupby("model")['acc_in_on_policy_error'].std()/np.sqrt(df.groupby("model")['acc_in_on_policy_error'].size()), df.groupby("model")['acc_in_on_policy_error'].size()], axis=1)
    summary.columns = ["accuracy", "standard_error", "size"]
    return summary.reset_index()



# print(gsm8k_df["checked"].value_counts())
# print(math_df["checked"].value_counts())
# print(olympiadbench_df["checked"].value_counts())
# print(omnimath_df["checked"].value_counts())

# gsm8k_df["checked"].replace({"Y": 1, "N": 0}, inplace=True)
# math_df["checked"].replace({"Y": 1, "N": 0}, inplace=True)
# olympiadbench_df["checked"].replace({"Y": 1, "N": 0}, inplace=True)
# omnimath_df["checked"].replace({"Y": 1, "N": 0}, inplace=True)

# gsm8k_df = gsm8k_df[gsm8k_df["checked"].isin([0, 1])]
# math_df = math_df[math_df["checked"].isin([0, 1])]
# olympiadbench_df = olympiadbench_df[olympiadbench_df["checked"].isin([0, 1])]
# omnimath_df = omnimath_df[omnimath_df["checked"].isin([0, 1])]

gsm8k_summary = summary("rebuttal/on_policy_error/on_policy_error_gsm8k_v2_llm_eval.jsonl")
gsm8k_summary['dataset'] = 'GSM8K'
math_summary = summary("rebuttal/on_policy_error/on_policy_error_math_v2_llm_eval.jsonl")
math_summary['dataset'] = 'MATH'
olympiadbench_summary = summary("rebuttal/on_policy_error/on_policy_error_olympiadbench_v2_llm_eval.jsonl")
olympiadbench_summary['dataset'] = 'OlympiadBench'
omnimath_summary = summary("rebuttal/on_policy_error/on_policy_error_omnimath_v2_llm_eval.jsonl")
omnimath_summary['dataset'] = 'Omni-Math'

all_summary = pd.concat([
    gsm8k_summary,
    math_summary,
    olympiadbench_summary,
    omnimath_summary
    ]).round(4)[['model', 'dataset', 'accuracy', 'standard_error', 'size']].round(4)



all_summary.set_index(['model', 'dataset'], inplace=True)
all_summary = all_summary.reindex(['GSM8K', 'MATH', 'OlympiadBench', 'Omni-Math'], level=1)
all_summary = all_summary.reindex(['Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'], level=0)
all_summary = all_summary.reset_index()
all_summary['model'] = all_summary['model'].str.split('/').str[1]

print(all_summary.groupby('model')['accuracy'].mean())


def format_standard_error(mean, sem, m=1.96):
    sem = m * sem
    return f"{mean:.3f}" + " ± " + f"{sem:.3f}"

# all_summary['accuracy'] = all_summary.apply(lambda r: format_standard_error(r['accuracy'], r['standard_error']), axis=1)

# del all_summary['standard_error']

print(all_summary.round(3).to_markdown(index=False))
print(all_summary.round(3).astype(str).to_latex(index=False))


all_summary = summary_v2([
    "rebuttal/on_policy_error/on_policy_error_gsm8k_v2_llm_eval.jsonl",
    "rebuttal/on_policy_error/on_policy_error_math_v2_llm_eval.jsonl",
    "rebuttal/on_policy_error/on_policy_error_olympiadbench_v2_llm_eval.jsonl",
    "rebuttal/on_policy_error/on_policy_error_omnimath_v2_llm_eval.jsonl"
])
all_summary['Accuracy (95% CI)'] = all_summary.apply(lambda r: format_standard_error(r['accuracy'], r['standard_error'], m=1.96), axis=1)
del all_summary['accuracy']
del all_summary['standard_error']
all_summary['model'] = all_summary['model'].str.split('/').str[1]
all_summary = all_summary[['model', 'Accuracy (95% CI)', 'size']]


all_summary_subset = summary_v2([
    "rebuttal/on_policy_error/on_policy_error_gsm8k_v2_llm_eval.jsonl",
    "rebuttal/on_policy_error/on_policy_error_math_v2_llm_eval.jsonl",
])
all_summary_subset['Accuracy (95% CI) subset'] = all_summary_subset.apply(lambda r: format_standard_error(r['accuracy'], r['standard_error'], m=1.96), axis=1)
del all_summary_subset['accuracy']
del all_summary_subset['standard_error']
all_summary_subset['model'] = all_summary_subset['model'].str.split('/').str[1]
all_summary_subset = all_summary_subset.rename(columns={'size': 'size subset'})
all_summary_subset = all_summary_subset[['model', 'Accuracy (95% CI) subset', 'size subset']]

all_summary = all_summary.merge(all_summary_subset, on='model', how='left')


print(all_summary.to_markdown(index=False))
print(all_summary.to_latex(index=False))
    