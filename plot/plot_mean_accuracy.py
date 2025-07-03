import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evaluation.evaluate_tool import get_is_correct_answer
from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from plot.constants import MODEL_LIST, NON_REASONING_MODELS, REASONING_MODELS


def get_error_injection_model_data():
    """
    Extract error injection model BCA and ACA data from all three datasets.
    
    Returns:
        dict: Dictionary with dataset names as keys and DataFrames as values
    """
    results = {}
    
    # SCLI5 - uses single evaluation field (no BCA/ACA distinction)
    print("Loading SCLI5 data...")
    scli5_data = load_scli5_eval_data()
    scli5_df = pd.DataFrame([[d["model"], get_is_correct_answer(d, "llm_evaluation"), not bool(d["response_error_injection_in_model"]),
                              get_is_correct_answer(d, "llm_evaluation_wait")]
                            for d in scli5_data],
                           columns=["model", "error_in_model", "is_empty", "error_in_model_wait"])
    scli5_by_model = scli5_df.groupby("model").agg({
        'error_in_model': ['mean', 'std', 'count'],
        'is_empty': ['mean', 'std', 'count'],
        'error_in_model_wait': ['mean', 'std', 'count']
    }).round(4)
    results['scli5'] = scli5_by_model
    
    # GSM8K-SC - uses BCA and ACA fields
    print("Loading GSM8K-SC data...")
    gsm8k_data = load_gsm8k_sc_eval_data()
    gsm8k_df = pd.DataFrame([[d["model"], get_is_correct_answer(d, "llm_evaluation_bca"), get_is_correct_answer(d, "llm_evaluation_aca"),
                              not bool(d["response_error_injection_in_model_bca"]), not bool(d["response_error_injection_in_model_aca"]),
                              get_is_correct_answer(d, "llm_evaluation_bca_wait"), get_is_correct_answer(d, "llm_evaluation_aca_wait")] 
                            for d in gsm8k_data],
                           columns=["model", "error_in_model_bca", "error_in_model_aca", "is_empty_bca", "is_empty_aca",
                                    "error_in_model_bca_wait", "error_in_model_aca_wait"])
    gsm8k_by_model = gsm8k_df.groupby("model").agg({
        'error_in_model_bca': ['mean', 'std', 'count'],
        'error_in_model_aca': ['mean', 'std', 'count'],
        'is_empty_bca': ['mean', 'std', 'count'],
        'is_empty_aca': ['mean', 'std', 'count'],
        'error_in_model_bca_wait': ['mean', 'std', 'count'],
        'error_in_model_aca_wait': ['mean', 'std', 'count']
    }).round(4)
    results['gsm8k_sc'] = gsm8k_by_model
    
    # PRM800K-SC - uses BCA and ACA fields
    print("Loading PRM800K-SC data...")
    prm800k_data = load_prm800k_sc_eval_data(supplement=True)
    prm800k_df = pd.DataFrame([[d["model"], get_is_correct_answer(d, "llm_evaluation_bca"), get_is_correct_answer(d, "llm_evaluation_aca"),
                                not bool(d["response_error_injection_in_model_bca"]), not bool(d["response_error_injection_in_model_aca"]),
                                get_is_correct_answer(d, "llm_evaluation_bca_wait"), get_is_correct_answer(d, "llm_evaluation_aca_wait")] 
                              for d in prm800k_data],
                             columns=["model", "error_in_model_bca", "error_in_model_aca", "is_empty_bca", "is_empty_aca",
                                      "error_in_model_bca_wait", "error_in_model_aca_wait"])
    prm800k_by_model = prm800k_df.groupby("model").agg({
        'error_in_model_bca': ['mean', 'std', 'count'],
        'error_in_model_aca': ['mean', 'std', 'count'],
        'is_empty_bca': ['mean', 'std', 'count'],
        'is_empty_aca': ['mean', 'std', 'count'],
        'error_in_model_bca_wait': ['mean', 'std', 'count'],
        'error_in_model_aca_wait': ['mean', 'std', 'count']
    }).round(4)
    results['prm800k_sc'] = prm800k_by_model
    
    return results

def construct_data_matrix(data, model_list, config="default"):
    """
    Construct a data matrix from the given data.
    
    Args:
        data (dict): Dictionary with dataset data
        metric (str): 'bca' or 'aca'
    
    Returns:
        dict: Dictionary with model names as keys and macro averages as values
    """
    macro_averages = {}
    key_mapping = {
            "scli5": {
                "default": "error_in_model",
                "wait": "error_in_model_wait",
            },
            "gsm8k_sc": {
                "default": "error_in_model_bca",
                "wait": "error_in_model_bca_wait",
            },
            "prm800k_sc": {
                "default": "error_in_model_bca",
                "wait": "error_in_model_bca_wait",
            }
    }

    for model in model_list:
        model_scores = []
        
        for dataset_name, dataset_data in data.items():
            if dataset_name == 'scli5':
                mean_val = dataset_data.loc[model, (key_mapping[dataset_name][config], 'mean')]
            else:
                mean_val = dataset_data.loc[model, (key_mapping[dataset_name][config], 'mean')]
                
            model_scores.append(mean_val)
        macro_averages[model] = model_scores

    data = pd.DataFrame.from_dict(macro_averages, orient='index', columns=list(data.keys()))
    return data

def construct_sem_matrix(data, model_list, config="default"):
    """
    Construct a standard error of the mean (SEM) matrix from the given data.
    
    Args:
        data (dict): Dictionary with dataset data
    
    Returns:
        pd.DataFrame: DataFrame with model names as keys and standard errors of the mean as values
    """
    sem_averages = {}
    key_mapping = {
            "scli5": {
                "default": "error_in_model",
                "wait": "error_in_model_wait",
            },
            "gsm8k_sc": {
                "default": "error_in_model_bca",
                "wait": "error_in_model_bca_wait",
            },
            "prm800k_sc": {
                "default": "error_in_model_bca",
                "wait": "error_in_model_bca_wait",
            }
    }
    for model in model_list:
        model_sems = []
        
        for dataset_name, dataset_data in data.items():
            if dataset_name == 'scli5':
                std_val = dataset_data.loc[model, (key_mapping[dataset_name][config], 'std')]
                count_val = dataset_data.loc[model, (key_mapping[dataset_name][config], 'count')]
            else:
                std_val = dataset_data.loc[model, (key_mapping[dataset_name][config], 'std')]
                count_val = dataset_data.loc[model, (key_mapping[dataset_name][config], 'count')]
            
            # Calculate standard error of the mean: SEM = σ / √n
            sem_val = std_val / np.sqrt(count_val) if count_val > 0 else 0
            model_sems.append(sem_val)
        sem_averages[model] = model_sems

    sem_data = pd.DataFrame.from_dict(sem_averages, orient='index', columns=list(data.keys()))
    return sem_data

def construct_empty_matrix(data, model_list):
    """
    Construct a matrix of empty values for each model across all three datasets.
    
    Args:
        data (dict): Dictionary with dataset data
    
    Returns:
        pd.DataFrame: DataFrame with model names as keys and empty values as values
    """
    empty_averages = {}
    
    for model in model_list:
        model_scores = []
        
        for dataset_name, dataset_data in data.items():
            if dataset_name == 'scli5':
                mean_val = dataset_data.loc[model, (f'is_empty', 'mean')]
            else:
                mean_val = dataset_data.loc[model, (f'is_empty_bca', 'mean')]
                
            model_scores.append(mean_val)
        empty_averages[model] = model_scores

    data = pd.DataFrame.from_dict(empty_averages, orient='index', columns=list(data.keys()))
    return data

def plot_error_injection_model_macro_averages(model_list, output_file, config="default"):
    """
    Plot macro average error injection model for each model across all three datasets.
    """
    # Get data
    data = get_error_injection_model_data()
    
    # Calculate macro averages
    df = construct_data_matrix(data, model_list=model_list, config=config)
    df_sem = construct_sem_matrix(data, model_list=model_list, config=config)

    sorted_models = model_list
    
    # Reorder the data based on sorted models
    df_sorted = df.loc[sorted_models]
    df_sem_sorted = df_sem.loc[sorted_models]
    print(sorted_models)
    df_sorted['macro_average'] = df_sorted.mean(axis=1)
    print(df_sorted.round(3).astype(str).to_latex())
    del df_sorted['macro_average']
    
    # Prepare data for plotting
    macro_values = [df_sorted.loc[model].mean() for model in sorted_models]
    scli5_values = [df_sorted.loc[model, 'scli5'] for model in sorted_models]
    gsm8k_values = [df_sorted.loc[model, 'gsm8k_sc'] for model in sorted_models]
    prm800k_values = [df_sorted.loc[model, 'prm800k_sc'] for model in sorted_models]
    
    # Prepare error bar data (standard error of the mean)
    scli5_errors = [df_sem_sorted.loc[model, 'scli5'] * 1.96 for model in sorted_models]
    gsm8k_errors = [df_sem_sorted.loc[model, 'gsm8k_sc'] * 1.96 for model in sorted_models]
    prm800k_errors = [df_sem_sorted.loc[model, 'prm800k_sc'] * 1.96 for model in sorted_models]
    # Create figure
    _, ax = plt.subplots(figsize=(20, 10))
    
    # Set up bar positions
    x = np.arange(len(sorted_models))
    width = 0.2  # Reduced width to accommodate 4 bars per group
    
    # Colors for different datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot bars for each dataset with error bars
    ax.bar(x - 1.5*width, scli5_values, width, label='SCLI5', color=colors[0], alpha=0.3, 
           yerr=scli5_errors, capsize=3)
    ax.bar(x - 0.5*width, gsm8k_values, width, label='GSM8K-SC', color=colors[1], alpha=0.3,
           yerr=gsm8k_errors, capsize=3)
    ax.bar(x + 0.5*width, prm800k_values, width, label='PRM800K-SC', color=colors[2], alpha=0.3,
           yerr=prm800k_errors, capsize=3)
    bars4 = ax.bar(x + 1.5*width, macro_values, width, label='Macro Average', color=colors[3], alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Mean accuracy and macro average (95% confidence intervals) after injection of external error', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Use shortened model names for x-axis labels
    model_labels = [MODEL_LIST.get(model, model.split('/')[-1]) for model in sorted_models]
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars (only for macro average to avoid clutter)
    for bar in bars4:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    return macro_values

def plot_error_in_error_injection_model_macro_averages(model_list, output_file):
    """
    Plot macro average error injection model for each model across all three datasets.
    """
    # Get data
    data = get_error_injection_model_data()
    
    # Calculate macro averages
    df = construct_data_matrix(data, model_list)
    df_empty = construct_empty_matrix(data, model_list)
    
    # Sort models by macro average in descending order
    sorted_models = model_list
    
    # Reorder the data based on sorted models
    df_sorted = df.loc[sorted_models]
    df_empty_sorted = df_empty.loc[sorted_models]
    print(sorted_models)
    print(df_sorted.round(3).astype(str).to_latex())
    
    # Prepare empty values
    scli5_empty = [df_empty_sorted.loc[model, 'scli5'] for model in sorted_models]
    gsm8k_empty = [df_empty_sorted.loc[model, 'gsm8k_sc'] for model in sorted_models]
    prm800k_empty = [df_empty_sorted.loc[model, 'prm800k_sc'] for model in sorted_models]

    # Prepare data for plotting
    scli5_values = [1 - df_sorted.loc[model, 'scli5'] - df_empty_sorted.loc[model, 'scli5'] for model in sorted_models]
    gsm8k_values = [1 - df_sorted.loc[model, 'gsm8k_sc'] - df_empty_sorted.loc[model, 'gsm8k_sc'] for model in sorted_models]
    prm800k_values = [1 - df_sorted.loc[model, 'prm800k_sc'] - df_empty_sorted.loc[model, 'prm800k_sc'] for model in sorted_models]
    
    # Create figure
    _, ax = plt.subplots(figsize=(20, 10))
    
    # Set up bar positions
    x = np.arange(len(sorted_models))
    width = 0.2  # Reduced width to accommodate 4 bars per group
    
    # Colors for different datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot bars for each dataset with error bars
    ax.bar(x - 1.5*width, scli5_values, width, label='SCLI5', color=colors[0], alpha=0.3, capsize=3)
    ax.bar(x - 0.5*width, gsm8k_values, width, label='GSM8K-SC', color=colors[1], alpha=0.3, capsize=3)
    ax.bar(x + 0.5*width, prm800k_values, width, label='PRM800K-SC', color=colors[2], alpha=0.3, capsize=3)

    # Plot empty values (stacked on top with hatched patterns)
    ax.bar(x - 1.5*width, scli5_empty, width, color=colors[0], alpha=0.3, hatch='xxx', bottom=scli5_values)
    ax.bar(x - 0.5*width, gsm8k_empty, width, color=colors[1], alpha=0.3, hatch='xxx', bottom=gsm8k_values )
    ax.bar(x + 0.5*width, prm800k_empty, width, color=colors[2], alpha=0.3, hatch='xxx', bottom=prm800k_values)

    # Create custom legend with grouped entries
    from matplotlib.patches import Patch
    
    # Create legend handles - one for each dataset and one for all empty bars
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.3, label='SCLI5'),
        Patch(facecolor=colors[1], alpha=0.3, label='GSM8K-SC'),
        Patch(facecolor=colors[2], alpha=0.3, label='PRM800K-SC'),
        Patch(facecolor='white', edgecolor='black', hatch='xxx', label='Non-Response')
    ]
    
    # Customize plot
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error and non-response by dataset and model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Use shortened model names for x-axis labels
    model_labels = [MODEL_LIST.get(model, model.split('/')[-1]) for model in sorted_models]
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    
    ax.legend(handles=legend_elements, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_between_macro_averages_by_dataset(model_list, output_file):
    """
    Plot the correlation between macro averages of accuracy for each model across all three datasets.
    Creates a figure with 3 subplots: correlation heatmap, SCLI5 vs GSM8K_SC scatter, and GSM8K_SC vs PRM800K_SC scatter.
    """
    # Get data
    data = get_error_injection_model_data()
    
    # Calculate macro averages
    df = construct_data_matrix(data, model_list)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Subplot 1: Correlation heatmap (existing)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax1)
    ax1.set_title('Correlation matrix of \nmean accuracy across datasets', fontsize=14, fontweight='bold')
    
    # Subplot 2: SCLI5 vs GSM8K_SC scatter plot
    x_data = df['scli5']
    y_data = df['gsm8k_sc']
    
    # Create scatter plot
    ax2.scatter(x_data, y_data, alpha=0.7, s=80, color='blue')
    
    # Add fitted line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax2.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2, label='Fitted line')
    
    # Add model labels
    for model in model_list:
        model_name = MODEL_LIST.get(model, model.split('/')[-1])
        # Shorten model names for better fit
        if len(model_name) > 20:
            model_name = model_name[:17] + "..."
        ax2.annotate(model_name, 
                    (df.loc[model, 'scli5'], df.loc[model, 'gsm8k_sc']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', alpha=0.7))
    
    # Add ideal line (perfect correlation)
    min_val = min(x_data.min(), y_data.min())
    max_val = max(x_data.max(), y_data.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, linewidth=2, label='Ideal line')
    
    # Calculate correlation coefficient
    corr_coef = corr_matrix.loc['scli5', 'gsm8k_sc']
    
    ax2.set_xlabel('SCLI5 macro average', fontsize=12)
    ax2.set_ylabel('GSM8K-SC macro average', fontsize=12)
    ax2.set_title(f'SCLI5 vs GSM8K-SC\n(r = {corr_coef:.3f})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Subplot 3: GSM8K_SC vs PRM800K_SC scatter plot
    x_data = df['gsm8k_sc']
    y_data = df['prm800k_sc']
    
    # Create scatter plot
    ax3.scatter(x_data, y_data, alpha=0.7, s=80, color='green')
    
    # Add fitted line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax3.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2, label='Fitted line')
    
    # Add model labels
    for model in df.index:
        model_name = MODEL_LIST.get(model, model.split('/')[-1])
        # Shorten model names for better fit
        if len(model_name) > 20:
            model_name = model_name[:17] + "..."
        ax3.annotate(model_name, 
                    (df.loc[model, 'gsm8k_sc'], df.loc[model, 'prm800k_sc']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', alpha=0.7))
    
    # Add ideal line (perfect correlation)
    min_val = min(x_data.min(), y_data.min())
    max_val = max(x_data.max(), y_data.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, linewidth=2, label='Ideal line')
    
    # Calculate correlation coefficient
    corr_coef = corr_matrix.loc['gsm8k_sc', 'prm800k_sc']
    
    ax3.set_xlabel('GSM8K-SC macro average', fontsize=12)
    ax3.set_ylabel('PRM800K-SC macro average', fontsize=12)
    ax3.set_title(f'GSM8K-SC vs PRM800K-SC\n(r = {corr_coef:.3f})', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_no_wait_vs_wait_macro_averages(model_list, output_file):
    """
    Plot the macro averages of accuracy for each model across all three datasets for no wait and wait conditions.
    """
    # Get data

    data = get_error_injection_model_data()
    
    # Calculate macro averages
    df = construct_data_matrix(data, model_list=model_list, config='default')
    df_wait = construct_data_matrix(data, model_list=model_list, config='wait')

    sorted_models = model_list
    df_sorted = df.loc[sorted_models]
    df_wait_sorted = df_wait.loc[sorted_models]

    # Prepare data for plotting
    macro_values = [df_sorted.loc[model].mean() for model in sorted_models]
    macro_values_wait = [df_wait_sorted.loc[model].mean() for model in sorted_models]

    print("Macro average accuracy improvement from original to wait: ", np.array(macro_values_wait).mean() / np.array(macro_values).mean() - 1)

    # Create figure
    _, ax = plt.subplots(figsize=(20, 10))
    
    x = np.arange(len(sorted_models))
    width = 0.4 

    # Plot macro averages   
    bar1 = ax.bar(x - width/2, macro_values, width, label='Original', color='#d62728', alpha=0.8)
    bar2 = ax.bar(x + width/2, macro_values_wait, width, label='Appended Wait', color='#5c0505', alpha=0.8)

    # Add value labels on bars (only for macro average to avoid clutter)
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)


    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Macro average accuracy', fontsize=12)
    ax.set_title('Macro average accuracy increases from original to appended Wait', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Use shortened model names for x-axis labels
    model_labels = [MODEL_LIST.get(model, model.split('/')[-1]) for model in sorted_models]
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)    

    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    data = get_error_injection_model_data()
    df = construct_data_matrix(data, model_list=NON_REASONING_MODELS)
    macro_avgs = df.mean(axis=1)
    # Sort models by macro average in descending order
    sorted_non_reasoning_models = macro_avgs.sort_values(ascending=False).index.tolist()

    plot_error_injection_model_macro_averages(sorted_non_reasoning_models,
                                              'output/error_injection_model_macro_averages_non_reasoning.png')
    # plot_error_injection_model_macro_averages(sorted_non_reasoning_models,
    #                                           'output/error_injection_model_macro_averages_non_reasoning_wait.png',
    #                                           config="wait")
    
    plot_no_wait_vs_wait_macro_averages(sorted_non_reasoning_models,
                                       'output/error_injection_model_macro_averages_non_reasoning_no_wait_vs_wait.png')

    plot_error_in_error_injection_model_macro_averages(sorted_non_reasoning_models,
                                                       'output/error_in_error_injection_model_macro_averages_non_reasoning.png')
    plot_correlation_between_macro_averages_by_dataset(sorted_non_reasoning_models,
                                                       'output/error_injection_model_correlation_matrix_non_reasoning.png')

    df = construct_data_matrix(data, model_list=REASONING_MODELS)
    macro_avgs = df.mean(axis=1)
    # Sort models by macro average in descending order
    sorted_reasoning_models = macro_avgs.sort_values(ascending=False).index.tolist()


    plot_error_injection_model_macro_averages(sorted_reasoning_models,
                                              'output/error_injection_model_macro_averages_reasoning.png')
    plot_error_in_error_injection_model_macro_averages(sorted_reasoning_models,
                                                       'output/error_in_error_injection_model_macro_averages_reasoning.png')
