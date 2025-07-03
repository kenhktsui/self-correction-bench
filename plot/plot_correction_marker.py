import json
import re
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evaluation.evaluate_tool import get_is_correct_answer
from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from plot.constants import MODEL_LIST, NON_REASONING_MODELS


KEYWORDS = r"(?:^|[.!?]\s+)(Wait|No|But|Hold on|Hang on|Alternatively|However|Hmm)[,\s!]"
KEYWORDS_RE = re.compile(KEYWORDS)

def check_correction_marker(d, key):
    return len(KEYWORDS_RE.findall(d[key])) > 0

assert check_correction_marker({'response_error_injection_in_model': 'Wait, I think I made a mistake. Let me think again.'}, 'response_error_injection_in_model')
assert not check_correction_marker({'response_error_injection_in_model': 'Please wait for me.'}, 'response_error_injection_in_model')


data_scli5 = load_scli5_eval_data()
data_gsm8k = load_gsm8k_sc_eval_data()
data_prm800k = load_prm800k_sc_eval_data()


data_dict = {
    "scli5": data_scli5,
    "gsm8k": data_gsm8k,
    "prm800k": data_prm800k,
}


for dataset_name, data in data_dict.items():
    if dataset_name == "scli5":
        for d in tqdm(data, desc=dataset_name):
            m_name = d['model'] + '_thinking' if d.get('enable_thinking', False) else d['model']
            for k in [
                "response_error_injection_in_model",
                "response_error_in_user",
                "response_error_injection_in_model_wait"
                ]:
                cd = check_correction_marker(d, k)
                d[k + '_marker'] = cd
    else:
        for d in tqdm(data, desc=dataset_name):
            m_name = d['model'] + '_thinking' if d.get('enable_thinking', False) else d['model']
            id_key = d['id'] if dataset_name == 'gsm8k' else d['question']
            for k in [
                "response_error_injection_in_model_bca",
                "response_error_injection_in_model_aca",
                "response_error_in_user_bca",
                "response_error_in_user_aca",
                "response_error_injection_in_model_bca_wait",
                "response_error_injection_in_model_aca_wait"
                ]:
                cd = check_correction_marker(d, k)
                d[k + '_marker'] = cd

marker_df = {}
for dataset_name, data in data_dict.items():
    if dataset_name == "scli5":
        df = []
        for d in tqdm(data, desc=dataset_name):
            d_model_name = d['model'] + '_thinking' if d.get('enable_thinking', False) else d['model']
            if d_model_name not in NON_REASONING_MODELS:
                continue
            df.append([MODEL_LIST[d_model_name]] + [d[k + '_marker'] for k in [
                "response_error_injection_in_model",
                "response_error_injection_in_model_wait",
                "response_error_in_user"
                ]] + [
                    get_is_correct_answer(d, "llm_evaluation"),
                    get_is_correct_answer(d, "llm_evaluation_wait"),
                    get_is_correct_answer(d, "llm_evaluation_error_in_user")
                ])
        df = pd.DataFrame(df, columns=['model'] + [
                "marker_response_error_injection_in_model",
                "marker_response_error_injection_in_model_wait",
                "marker_response_error_in_user",
                "accuracy_response_error_injection_in_model",
                "accuracy_response_error_injection_in_model_wait",
                "accuracy_response_error_in_user"
                ])
        df = df.groupby('model').mean()
        df['marker_change'] = (df['marker_response_error_injection_in_model_wait'] - df['marker_response_error_injection_in_model'])
        df['accuracy_change'] = (df['accuracy_response_error_injection_in_model_wait'] - df['accuracy_response_error_injection_in_model'])
        df['marker_diff_user'] = (df['marker_response_error_in_user'] - df['marker_response_error_injection_in_model'])
        df['accuracy_diff_user'] = (df['accuracy_response_error_in_user'] - df['accuracy_response_error_injection_in_model'])
        marker_df[dataset_name] = df
    else:
        df = []
        for d in tqdm(data, desc=dataset_name):
            d_model_name = d['model'] + '_thinking' if d.get('enable_thinking', False) else d['model']
            if d_model_name not in NON_REASONING_MODELS:
                continue
            df.append([MODEL_LIST[d_model_name]] + [d[k + '_marker'] for k in [
                "response_error_injection_in_model_bca",
                "response_error_injection_in_model_bca_wait",
                "response_error_in_user_bca",
                ]] + [
                    get_is_correct_answer(d, "llm_evaluation_bca"),
                    get_is_correct_answer(d, "llm_evaluation_bca_wait"),
                    get_is_correct_answer(d, "llm_evaluation_error_in_user_bca"),
                ])
        df = pd.DataFrame(df, columns=['model'] + [
                "marker_response_error_injection_in_model",
                "marker_response_error_injection_in_model_wait",
                "marker_response_error_in_user",
                "accuracy_response_error_injection_in_model",
                "accuracy_response_error_injection_in_model_wait",
                "accuracy_response_error_in_user"
                ])
        df = df.groupby('model').mean()
        df['marker_change'] = (df['marker_response_error_injection_in_model_wait'] - df['marker_response_error_injection_in_model'])
        df['accuracy_change'] = (df['accuracy_response_error_injection_in_model_wait'] - df['accuracy_response_error_injection_in_model'])
        df['marker_diff_user'] = (df['marker_response_error_in_user'] - df['marker_response_error_injection_in_model'])
        df['accuracy_diff_user'] = (df['accuracy_response_error_in_user'] - df['accuracy_response_error_injection_in_model'])
        marker_df[dataset_name] = df


for d, df in marker_df.items():
    print(d)
    print(df[['marker_response_error_in_user', 'marker_response_error_injection_in_model']])
    print("User / Model Error Marker Increase", df['marker_response_error_in_user'].mean(axis=0)/ df['marker_response_error_injection_in_model'].mean(axis=0) - 1)
    print(df[['marker_change', 'accuracy_change', 'marker_diff_user', 'accuracy_diff_user']])
    print(df[['marker_change', 'accuracy_change']].corr())
    print(df[['marker_diff_user', 'accuracy_diff_user']].corr())
    print('-'*100)

# Create correlation plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Correlation: change in correction marker presence vs change in accuracy after appending Wait', fontsize=16)

dataset_names = ['scli5', 'gsm8k', 'prm800k']
dataset_titles = ['SCLI5', 'GSM8K_SC', 'PRM800K_SC']


for i, (dataset_name, title) in enumerate(zip(dataset_names, dataset_titles)):
    df = marker_df[dataset_name]
    
    # Create scatter plot
    axes[i].scatter(df['marker_change'], df['accuracy_change'], alpha=0.7, s=60)
    
    # Add trend line
    z = np.polyfit(df['marker_change'], df['accuracy_change'], 1)
    p = np.poly1d(z)
    axes[i].plot(df['marker_change'], p(df['marker_change']), "r--", alpha=0.8)
    
    # Add correlation coefficient
    corr = df[['marker_change', 'accuracy_change']].corr().iloc[0, 1]
    axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=axes[i].transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    
    axes[i].set_xlabel('Absolute Change in Correction Marker Presence')
    axes[i].set_ylabel('Absolute Change in Accuracy')
    axes[i].set_title(title)
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)


plt.tight_layout()
plt.savefig('output/correlation_plots_marker_presence_vs_accuracy_after_wait.png', dpi=300, bbox_inches='tight')
plt.show()
