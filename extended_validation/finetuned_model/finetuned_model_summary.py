import json
import pandas as pd
import numpy as np


with open("finetuned_model.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

with open("finetuned_model_llama.jsonl", "r") as f:
    data.extend([json.loads(line) for line in f])


df = pd.DataFrame.from_dict(data)
print(df["checked_error_injection_in_model"].value_counts())
print(df["checked_error_in_user"].value_counts())

df["checked_error_injection_in_model"].replace({"Y": 1, "N": 0}, inplace=True)
df["checked_error_in_user"].replace({"Y": 1, "N": 0}, inplace=True)

df = df[df["checked_error_injection_in_model"].isin([1, 0])]
df = df[df["checked_error_in_user"].isin([1, 0])]


df_summary = pd.concat([df.groupby(["model", "dataset"])[["checked_error_injection_in_model", "checked_error_in_user"]].mean(),
                        df.groupby(["model", "dataset"])[["checked_error_injection_in_model", "checked_error_in_user"]].std()/np.sqrt(df.groupby(["model", "dataset"])[["checked_error_injection_in_model", "checked_error_in_user"]].count()),
                        df.groupby(["model", "dataset"])[["checked_error_injection_in_model", "checked_error_in_user"]].count()], axis=1)
df_summary.columns = ["mean_error_injection_in_model", "mean_error_in_user", "sem_error_injection_in_model", "sem_error_in_user", "size_error_injection_in_model", "size_error_in_user"]


df_summary["self-correction_blind_spot"] = 1 - df_summary["mean_error_injection_in_model"]/df_summary["mean_error_in_user"]


# error_in_model_sem = df_summary["sem_error_injection_in_model"] / np.sqrt(df_summary["size_error_injection_in_model"])
# error_in_user_sem = df_summary["sem_error_in_user"] / np.sqrt(df_summary["size_error_in_user"])
# d_dx = df_summary["mean_error_injection_in_model"] / (df_summary["mean_error_in_user"] ** 2)
# d_dy = -1 / df_summary["mean_error_in_user"]

print(df_summary)
