import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from evaluation.evaluate_tool import get_is_correct_answer
from evaluation.evaluate_scli5 import load_scli5_eval_data
from evaluation.evaluate_gsm8k_sc import load_gsm8k_sc_eval_data
from evaluation.evaluate_prm800k_sc import load_prm800k_sc_eval_data
from plot.constants import MODEL_LIST, NON_REASONING_MODELS, REASONING_MODELS
from plot.plot_mean_accuracy import get_error_injection_model_data, construct_data_matrix


def calculate_blind_spot(dataset, question_types=None, field_mapping=None):
    """
    Calculate blind spot for a given dataset.
    
    Args:
        dataset (str): Dataset name - 'scli5', 'gsm8k', or 'prm800k'
        question_types (list, optional): For SCLI5, filter by question types
        field_mapping (dict, optional): Custom field mapping for different settings.
            For SCLI5: {'error_in_model': 'field_name', 'error_in_user': 'field_name'}
            For GSM8K/PRM800K: {
                'error_in_model_bca': 'field_name', 
                'error_in_user_bca': 'field_name',
                'error_in_model_aca': 'field_name', 
                'error_in_user_aca': 'field_name'
            }
            If None, uses default field mappings.
    
    Returns:
        dict: Blind spot results with mean and standard error
    """
    # Default field mappings
    default_mappings = {
        'scli5': {
            'error_in_model': 'llm_evaluation',
            'error_in_user': 'llm_evaluation_error_in_user'
        },
        'gsm8k': {
            'error_in_model_bca': 'llm_evaluation_bca',
            'error_in_user_bca': 'llm_evaluation_error_in_user_bca',
            'error_in_model_aca': 'llm_evaluation_aca',
            'error_in_user_aca': 'llm_evaluation_error_in_user_aca'
        },
        'prm800k': {
            'error_in_model_bca': 'llm_evaluation_bca',
            'error_in_user_bca': 'llm_evaluation_error_in_user_bca',
            'error_in_model_aca': 'llm_evaluation_aca',
            'error_in_user_aca': 'llm_evaluation_error_in_user_aca'
        }
    }
    
    if field_mapping is None:
        field_mapping = default_mappings.get(dataset, {})
    
    if dataset == 'scli5':
        data_with_llm_eval = load_scli5_eval_data()
        
        if question_types:
            filtered_data = []
            for d in data_with_llm_eval:
                question_type = d.get("question_type", "")
                if question_type in question_types:
                    filtered_data.append(d)
            data_with_llm_eval = filtered_data
        
        df = pd.DataFrame([[d["model"], 
                           get_is_correct_answer(d, field_mapping.get('error_in_model', 'llm_evaluation')), 
                           get_is_correct_answer(d, field_mapping.get('error_in_user', 'llm_evaluation_error_in_user'))] 
                          for d in data_with_llm_eval],
                         columns=["model", "error_in_model", "error_in_user"])
        
        blind_spot_by_model = df.groupby("model").agg({
            'error_in_model': ['mean', 'std'],
            'error_in_user': ['mean', 'std']
        })
        
        sample_sizes = df.groupby("model").size()
        
        blind_spot_mean = {}
        blind_spot_sem = {}
        
        for model in blind_spot_by_model.index:
            error_in_model_mean = blind_spot_by_model.loc[model, ('error_in_model', 'mean')]
            error_in_user_mean = blind_spot_by_model.loc[model, ('error_in_user', 'mean')]
            error_in_model_std = blind_spot_by_model.loc[model, ('error_in_model', 'std')]
            error_in_user_std = blind_spot_by_model.loc[model, ('error_in_user', 'std')]
            n = sample_sizes[model]
            
            if error_in_user_mean > 0:
                blind_spot_mean[model] = (error_in_user_mean - error_in_model_mean) / error_in_user_mean
            else:
                blind_spot_mean[model] = 0
            
            # Calculate standard error
            if error_in_user_mean > 0 and n > 0:
                error_in_model_sem = error_in_model_std / np.sqrt(n)
                error_in_user_sem = error_in_user_std / np.sqrt(n)
                
                d_dx = error_in_model_mean / (error_in_user_mean ** 2)
                d_dy = -1 / error_in_user_mean
                
                blind_spot_sem[model] = np.sqrt((d_dx ** 2) * (error_in_user_sem ** 2) + (d_dy ** 2) * (error_in_model_sem ** 2))
            else:
                blind_spot_sem[model] = 0
        
        return {'mean': blind_spot_mean, 'sem': blind_spot_sem}
    
    elif dataset in ['gsm8k', 'prm800k']:
        if dataset == 'gsm8k':
            data_with_llm_eval = load_gsm8k_sc_eval_data()
        else:
            data_with_llm_eval = load_prm800k_sc_eval_data()
        
        # GSM8K-SC and PRM800K-SC use BCA/ACA fields
        df = pd.DataFrame([[d["model"], 
                           get_is_correct_answer(d, field_mapping.get('error_in_model_bca', 'llm_evaluation_bca')), 
                           get_is_correct_answer(d, field_mapping.get('error_in_model_aca', 'llm_evaluation_aca')),
                           get_is_correct_answer(d, field_mapping.get('error_in_user_bca', 'llm_evaluation_error_in_user_bca')), 
                           get_is_correct_answer(d, field_mapping.get('error_in_user_aca', 'llm_evaluation_error_in_user_aca'))] 
                          for d in data_with_llm_eval],
                         columns=["model", "error_in_model_bca", "error_in_model_aca", 
                                 "error_in_user_bca", "error_in_user_aca"])
        
        blind_spot_by_model = df.groupby("model").agg({
            'error_in_model_bca': ['mean', 'std'],
            'error_in_model_aca': ['mean', 'std'],
            'error_in_user_bca': ['mean', 'std'],
            'error_in_user_aca': ['mean', 'std']
        })
        
        sample_sizes = df.groupby("model").size()
        
        blind_spot_mean_bca = {}
        blind_spot_sem_bca = {}
        blind_spot_mean_aca = {}
        blind_spot_sem_aca = {}
        
        for model in blind_spot_by_model.index:
            error_in_model_bca_mean = blind_spot_by_model.loc[model, ('error_in_model_bca', 'mean')]
            error_in_user_bca_mean = blind_spot_by_model.loc[model, ('error_in_user_bca', 'mean')]
            error_in_model_bca_std = blind_spot_by_model.loc[model, ('error_in_model_bca', 'std')]
            error_in_user_bca_std = blind_spot_by_model.loc[model, ('error_in_user_bca', 'std')]
            
            error_in_model_aca_mean = blind_spot_by_model.loc[model, ('error_in_model_aca', 'mean')]
            error_in_user_aca_mean = blind_spot_by_model.loc[model, ('error_in_user_aca', 'mean')]
            error_in_model_aca_std = blind_spot_by_model.loc[model, ('error_in_model_aca', 'std')]
            error_in_user_aca_std = blind_spot_by_model.loc[model, ('error_in_user_aca', 'std')]
            
            n = sample_sizes[model]
            
            if error_in_user_bca_mean > 0:
                blind_spot_mean_bca[model] = (error_in_user_bca_mean - error_in_model_bca_mean) / error_in_user_bca_mean
            else:
                blind_spot_mean_bca[model] = 0
            
            if error_in_user_aca_mean > 0:
                blind_spot_mean_aca[model] = (error_in_user_aca_mean - error_in_model_aca_mean) / error_in_user_aca_mean
            else:
                blind_spot_mean_aca[model] = 0
            
            if error_in_user_bca_mean > 0 and n > 0:
                error_in_model_bca_sem = error_in_model_bca_std / np.sqrt(n)
                error_in_user_bca_sem = error_in_user_bca_std / np.sqrt(n)
                
                d_dx_bca = error_in_model_bca_mean / (error_in_user_bca_mean ** 2)
                d_dy_bca = -1 / error_in_user_bca_mean
                blind_spot_sem_bca[model] = np.sqrt((d_dx_bca ** 2) * (error_in_user_bca_sem ** 2) + (d_dy_bca ** 2) * (error_in_model_bca_sem ** 2))
            else:
                blind_spot_sem_bca[model] = 0
            
            if error_in_user_aca_mean > 0 and n > 0:
                error_in_model_aca_sem = error_in_model_aca_std / np.sqrt(n)
                error_in_user_aca_sem = error_in_user_aca_std / np.sqrt(n)
                
                d_dx_aca = error_in_model_aca_mean / (error_in_user_aca_mean ** 2)
                d_dy_aca = -1 / error_in_user_aca_mean
                blind_spot_sem_aca[model] = np.sqrt((d_dx_aca ** 2) * (error_in_user_aca_sem ** 2) + (d_dy_aca ** 2) * (error_in_model_aca_sem ** 2))
            else:
                blind_spot_sem_aca[model] = 0
        
        return {
            'BCA': {'mean': blind_spot_mean_bca, 'sem': blind_spot_sem_bca},
            'ACA': {'mean': blind_spot_mean_aca, 'sem': blind_spot_sem_aca}
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'scli5', 'gsm8k', or 'prm800k'")


def calculate_blind_spot_scli5(question_types=None):
    """Calculate blind spot for SCLI5 dataset with optional question type filtering"""
    return calculate_blind_spot('scli5', question_types)

def calculate_blind_spot_gsm8k():
    """Calculate blind spot for GSM8K-SC dataset"""
    return calculate_blind_spot('gsm8k')

def calculate_blind_spot_prm800k():
    """Calculate blind spot for PRM800K-SC dataset"""
    return calculate_blind_spot('prm800k')

def calculate_blind_spot_scli5_wait(question_types=None):
    """Calculate blind spot for SCLI5 dataset using wait fields"""
    field_mapping = {
        'error_in_model': 'llm_evaluation_wait',
        'error_in_user': 'llm_evaluation_error_in_user'
    }
    return calculate_blind_spot('scli5', question_types, field_mapping)

def calculate_blind_spot_scli5_cot(question_types=None):
    """Calculate blind spot for SCLI5 dataset using CoT fields"""
    field_mapping = {
        'error_in_model': 'llm_evaluation_cot',
        'error_in_user': 'llm_evaluation_error_in_user'
    }
    return calculate_blind_spot('scli5', question_types, field_mapping)

def calculate_blind_spot_gsm8k_wait():
    """Calculate blind spot for GSM8K-SC dataset using wait fields"""
    field_mapping = {
        'error_in_model_bca': 'llm_evaluation_bca_wait',
        'error_in_user_bca': 'llm_evaluation_error_in_user_bca',
        'error_in_model_aca': 'llm_evaluation_aca_wait',
        'error_in_user_aca': 'llm_evaluation_error_in_user_aca'
    }
    return calculate_blind_spot('gsm8k', field_mapping=field_mapping)

def calculate_blind_spot_gsm8k_cot():
    """Calculate blind spot for GSM8K-SC dataset using CoT fields"""
    field_mapping = {
        'error_in_model_bca': 'llm_evaluation_bca_cot',
        'error_in_user_bca': 'llm_evaluation_error_in_user_bca',
        'error_in_model_aca': 'llm_evaluation_aca_cot',
        'error_in_user_aca': 'llm_evaluation_error_in_user_aca'
    }
    return calculate_blind_spot('gsm8k', field_mapping=field_mapping)

def calculate_blind_spot_prm800k_wait():
    """Calculate blind spot for PRM800K-SC dataset using wait fields"""
    field_mapping = {
        'error_in_model_bca': 'llm_evaluation_bca_wait',
        'error_in_user_bca': 'llm_evaluation_error_in_user_bca',
        'error_in_model_aca': 'llm_evaluation_aca_wait',
        'error_in_user_aca': 'llm_evaluation_error_in_user_aca'
    }
    return calculate_blind_spot('prm800k', field_mapping=field_mapping)

def calculate_blind_spot_prm800k_cot():
    """Calculate blind spot for PRM800K-SC dataset using CoT fields"""
    field_mapping = {
        'error_in_model_bca': 'llm_evaluation_bca_cot',
        'error_in_user_bca': 'llm_evaluation_error_in_user_bca',
        'error_in_model_aca': 'llm_evaluation_aca_cot',
        'error_in_user_aca': 'llm_evaluation_error_in_user_aca'
    }
    return calculate_blind_spot('prm800k', field_mapping=field_mapping)

def plot_blind_spot_summary_generic(model_list, filename_suffix, setting='default'):
    """
    Generic function to create vertical bar chart summarizing blind spot across datasets.
    
    Args:
        setting (str): Setting to use - 'default', 'wait', or 'cot'
    
    Returns:
        tuple: (data_matrix_mean, data_matrix_sem, all_models, datasets)
    """
    # Define setting configurations
    setting_configs = {
        'default': {
            'scli5_func': calculate_blind_spot_scli5,
            'gsm8k_func': calculate_blind_spot_gsm8k,
            'prm800k_func': calculate_blind_spot_prm800k,
            'dataset_labels': [
                'SCLI5',
                'GSM8K-SC (Before commit answer)',
                'GSM8K-SC (After commit answer)',
                'PRM800K-SC (Before commit answer)',
                'PRM800K-SC (After commit answer)'
            ],
            'title_suffix': '',
        },
        'wait': {
            'scli5_func': calculate_blind_spot_scli5_wait,
            'gsm8k_func': calculate_blind_spot_gsm8k_wait,
            'prm800k_func': calculate_blind_spot_prm800k_wait,
            'dataset_labels': [
                'SCLI5 (Wait)',
                'GSM8K-SC (Before commit answer, Wait)',
                'GSM8K-SC (After commit answer, Wait)',
                'PRM800K-SC (Before commit answer, Wait)',
                'PRM800K-SC (After commit answer, Wait)'
            ],
            'title_suffix': ' (Appending "Wait")',
        },
        'cot': {
            'scli5_func': calculate_blind_spot_scli5_cot,
            'gsm8k_func': calculate_blind_spot_gsm8k_cot,
            'prm800k_func': calculate_blind_spot_prm800k_cot,
            'dataset_labels': [
                'SCLI5 (CoT)',
                'GSM8K-SC (Before commit answer, CoT)',
                'GSM8K-SC (After commit answer, CoT)',
                'PRM800K-SC (Before commit answer, CoT)',
                'PRM800K-SC (After commit answer, CoT)'
            ],
            'title_suffix': ' (CoT Fields)',
        }
    }
    
    if setting not in setting_configs:
        raise ValueError(f"Unknown setting: {setting}. Must be 'default', 'wait', or 'cot'")
    
    config = setting_configs[setting]
    
    scli5_result = config['scli5_func']()
    scli5_blind_spot_mean, scli5_blind_spot_sem = scli5_result['mean'], scli5_result['sem']
    gsm8k_blind_spot = config['gsm8k_func']()
    prm800k_blind_spot = config['prm800k_func']()
    
    all_models = model_list
    datasets = config['dataset_labels']
    
    data_matrix_mean = []
    data_matrix_sem = []
    
    for model in all_models:
        row_mean = []
        row_sem = []
        
        # SCLI5
        scli5_val = scli5_blind_spot_mean.get(model, np.nan)
        scli5_sem_val = scli5_blind_spot_sem.get(model, np.nan)
        row_mean.append(scli5_val)
        row_sem.append(scli5_sem_val)
        
        # GSM8K-SC BCA
        gsm8k_bca_val = gsm8k_blind_spot['BCA']['mean'].get(model, np.nan)
        gsm8k_bca_sem_val = gsm8k_blind_spot['BCA']['sem'].get(model, np.nan)
        row_mean.append(gsm8k_bca_val)
        row_sem.append(gsm8k_bca_sem_val)
        
        # GSM8K-SC ACA
        gsm8k_aca_val = gsm8k_blind_spot['ACA']['mean'].get(model, np.nan)
        gsm8k_aca_sem_val = gsm8k_blind_spot['ACA']['sem'].get(model, np.nan)
        row_mean.append(gsm8k_aca_val)
        row_sem.append(gsm8k_aca_sem_val)
        
        # PRM800K-SC BCA
        prm800k_bca_val = prm800k_blind_spot['BCA']['mean'].get(model, np.nan)
        prm800k_bca_sem_val = prm800k_blind_spot['BCA']['sem'].get(model, np.nan)
        row_mean.append(prm800k_bca_val)
        row_sem.append(prm800k_bca_sem_val)
        
        # PRM800K-SC ACA
        prm800k_aca_val = prm800k_blind_spot['ACA']['mean'].get(model, np.nan)
        prm800k_aca_sem_val = prm800k_blind_spot['ACA']['sem'].get(model, np.nan)
        row_mean.append(prm800k_aca_val)
        row_sem.append(prm800k_aca_sem_val)
        
        data_matrix_mean.append(row_mean)
        data_matrix_sem.append(row_sem)
    
    data_matrix_mean = np.array(data_matrix_mean)
    data_matrix_sem = np.array(data_matrix_sem)
    
    _, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(all_models))
    width = 0.15  # Width of bars
    multiplier = 0
    
    colors = ['#1f77b4', '#ff7f0e', '#fccea4', '#2ca02c', '#80ad80']
    
    # Create bars for each dataset
    for i, dataset in enumerate(datasets):
        offset = width * multiplier
        values = data_matrix_mean[:, i]
        errors = data_matrix_sem[:, i]
        
        valid_indices = ~np.isnan(values)
        valid_x = x[valid_indices] + offset
        valid_values = values[valid_indices]
        valid_errors = errors[valid_indices]
        
        ax.bar(valid_x, valid_values, width, label=dataset, 
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5,
               yerr=1.96 * valid_errors, capsize=3)
        
        multiplier += 1
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Self-Correction Blind Spot', fontsize=12)
    ax.set_title(f'Blind Spot summary across datasets{config["title_suffix"]} - 95% Confidence Intervals', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)  # Center the x-ticks
    ax.set_xticklabels([MODEL_LIST[model] for model in all_models], rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_ylim(data_matrix_mean.min() - 1.96 * data_matrix_sem.max() - 0.02, data_matrix_mean.max() + 1.96 * data_matrix_sem.max() + 0.02)
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'output/blind_spot_summary_{setting}_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Blind Spot Summary Statistics{config['title_suffix']} - 95% Confidence Intervals:")
    print("=" * 80)
    
    for i, dataset in enumerate(datasets):
        values = data_matrix_mean[:, i]
        errors = data_matrix_sem[:, i]
        valid_indices = ~np.isnan(values)
        valid_values = values[valid_indices]
        valid_errors = errors[valid_indices]
        
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            mean_error = np.mean(valid_errors)
            print(f"{dataset:35} | Mean: {mean_val:.3f}±{1.96*mean_error:.3f} | N: {len(valid_values)}")
    
    return data_matrix_mean, data_matrix_sem, all_models, datasets

def plot_blind_spot_summary(model_list, filename_suffix):
    """Create vertical bar chart summarizing blind spot across datasets (default fields)"""
    return plot_blind_spot_summary_generic(model_list, filename_suffix,'default')

def plot_blind_spot_summary_wait(model_list, filename_suffix):
    """Create vertical bar chart summarizing blind spot across datasets using wait fields"""
    return plot_blind_spot_summary_generic(model_list, filename_suffix, 'wait')

def plot_blind_spot_summary_cot(model_list, filename_suffix):
    """Create vertical bar chart summarizing blind spot across datasets using CoT fields"""
    return plot_blind_spot_summary_generic(model_list, filename_suffix,'cot')

def plot_blind_spot_correlation(model_list, filename_suffix):
    """Create correlation plot of blind spot scores across datasets for each model"""
    
    scli5_result = calculate_blind_spot('scli5')
    scli5_blind_spot_mean, _ = scli5_result['mean'], scli5_result['sem']
    gsm8k_blind_spot = calculate_blind_spot('gsm8k')
    prm800k_blind_spot = calculate_blind_spot('prm800k')
    
    all_models = model_list
    
    # Use BCA (Before commit answer) for GSM8K and PRM800K to match SCLI5
    correlation_data = []
    
    for model in all_models:
        scli5_val = scli5_blind_spot_mean.get(model, np.nan)
        gsm8k_val = gsm8k_blind_spot['BCA']['mean'].get(model, np.nan)
        prm800k_val = prm800k_blind_spot['BCA']['mean'].get(model, np.nan)
    
        if not (np.isnan(scli5_val) or np.isnan(gsm8k_val) or np.isnan(prm800k_val)):
            correlation_data.append({
                'model': model,
                'SCLI5': scli5_val,
                'GSM8K-SC (BCA)': gsm8k_val,
                'PRM800K-SC (BCA)': prm800k_val
            })
    
    if not correlation_data:
        print("No models have complete data across all three datasets")
        return
    
    df_corr = pd.DataFrame(correlation_data)
    corr_matrix = df_corr[['SCLI5', 'GSM8K-SC (BCA)', 'PRM800K-SC (BCA)']].corr()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Correlation heatmap
    sns.heatmap(corr_matrix, ax=ax1, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True)
    ax1.set_title('Blind Spot correlation matrix', fontsize=14, fontweight='bold')

    # Plot 2: SCLI5 vs GSM8K-SC (BCA) scatter plot
    dataset1, dataset2 = 'SCLI5', 'GSM8K-SC (BCA)'
    corr_coef = corr_matrix.loc[dataset1, dataset2]
    
    x_data = df_corr[dataset1]
    y_data = df_corr[dataset2]
    
    ax2.scatter(x_data, y_data, alpha=0.7, s=80, color='blue')
    
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax2.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
    
    for _, row_data in df_corr.iterrows():
        model_name = MODEL_LIST.get(row_data['model'], row_data['model'])
        # Shorten model names for better fit
        if len(model_name) > 20:
            model_name = model_name[:17] + "..."
        ax2.annotate(model_name, 
                    (row_data[dataset1], row_data[dataset2]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', alpha=0.7))
    
    ax2.set_xlabel(f'{dataset1} Blind Spot Score', fontsize=12)
    ax2.set_ylabel(f'{dataset2} Blind Spot Score', fontsize=12)
    ax2.set_title(f'Blind Spot correlation: {dataset1} vs {dataset2}\n(r = {corr_coef:.3f})', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    min_val = min(x_data.min(), y_data.min())
    max_val = max(x_data.max(), y_data.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, label='Perfect correlation')
    ax2.legend(loc='best')
    
    # Plot 3: Scatter plot for the strongest correlation (excluding SCLI5 vs GSM8K-SC)
    # Find the pair with highest correlation (excluding diagonal and SCLI5 vs GSM8K-SC)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            # Skip SCLI5 vs GSM8K-SC as it's already plotted
            if not ((pair[0] == 'SCLI5' and pair[1] == 'GSM8K-SC (BCA)') or 
                   (pair[0] == 'GSM8K-SC (BCA)' and pair[1] == 'SCLI5')):
                corr_pairs.append(pair)
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if corr_pairs:
        dataset1, dataset2, corr_coef = corr_pairs[0]
        
        # Get data for this pair
        x_data = df_corr[dataset1]
        y_data = df_corr[dataset2]
        
        # Create scatter plot
        ax3.scatter(x_data, y_data, alpha=0.7, s=80, color='green')
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax3.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
        
        # Add model labels with better positioning
        for _, row_data in df_corr.iterrows():
            model_name = MODEL_LIST.get(row_data['model'], row_data['model'])
            if len(model_name) > 20:
                model_name = model_name[:17] + "..."
            ax3.annotate(model_name, 
                        (row_data[dataset1], row_data[dataset2]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', 
                                                       facecolor='white', alpha=0.7))
        
        ax3.set_xlabel(f'{dataset1} Blind Spot Score', fontsize=12)
        ax3.set_ylabel(f'{dataset2} Blind Spot Score', fontsize=12)
        ax3.set_title(f'Blind Spot correlation: {dataset1} vs {dataset2}\n(r = {corr_coef:.3f})', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add perfect correlation line for reference
        min_val = min(x_data.min(), y_data.min())
        max_val = max(x_data.max(), y_data.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, label='Perfect correlation')
        ax3.legend(loc='best')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'output/blind_spot_correlation_bca_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Blind Spot Correlation Analysis:")
    print("=" * 50)
    print(f"Number of models with complete data: {len(correlation_data)}")
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Calculate average correlation
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper_triangle.stack().mean()
    print(f"\nAverage correlation across datasets: {avg_corr:.3f}")
    
    return df_corr, corr_matrix

def plot_blind_spot_correlation_aca(model_list, filename_suffix):
    """Create correlation plot of blind spot scores across datasets for each model using ACA"""
    
    scli5_result = calculate_blind_spot('scli5')
    scli5_blind_spot_mean, _ = scli5_result['mean'], scli5_result['sem']
    gsm8k_blind_spot = calculate_blind_spot('gsm8k')
    prm800k_blind_spot = calculate_blind_spot('prm800k')
    
    all_models = model_list
    
    # Prepare data for correlation analysis
    # Use ACA (After commit answer) for GSM8K and PRM800K to match SCLI5
    correlation_data = []
    
    for model in all_models:
        scli5_val = scli5_blind_spot_mean.get(model, np.nan)
        gsm8k_val = gsm8k_blind_spot['ACA']['mean'].get(model, np.nan)
        prm800k_val = prm800k_blind_spot['ACA']['mean'].get(model, np.nan)

        if not (np.isnan(scli5_val) or np.isnan(gsm8k_val) or np.isnan(prm800k_val)):
            correlation_data.append({
                'model': model,
                'SCLI5': scli5_val,
                'GSM8K-SC (ACA)': gsm8k_val,
                'PRM800K-SC (ACA)': prm800k_val
            })
    
    if not correlation_data:
        print("No models have complete data across all three datasets")
        return
    
    df_corr = pd.DataFrame(correlation_data)
    corr_matrix = df_corr[['SCLI5', 'GSM8K-SC (ACA)', 'PRM800K-SC (ACA)']].corr()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Correlation heatmap
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(corr_matrix.columns)))
    ax1.set_yticks(range(len(corr_matrix.columns)))
    ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax1.set_yticklabels(corr_matrix.columns)
    ax1.set_title('Blind Spot correlation matrix (ACA)', fontsize=14, fontweight='bold')
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    # Plot 2: SCLI5 vs GSM8K-SC (ACA) scatter plot
    dataset1, dataset2 = 'SCLI5', 'GSM8K-SC (ACA)'
    corr_coef = corr_matrix.loc[dataset1, dataset2]
    
    x_data = df_corr[dataset1]
    y_data = df_corr[dataset2]
    
    ax2.scatter(x_data, y_data, alpha=0.7, s=80, color='blue')
    
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax2.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
    
    for _, row_data in df_corr.iterrows():
        model_name = MODEL_LIST.get(row_data['model'], row_data['model'])
        # Shorten model names for better fit
        if len(model_name) > 20:
            model_name = model_name[:17] + "..."
        ax2.annotate(model_name, 
                    (row_data[dataset1], row_data[dataset2]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', alpha=0.7))
    
    ax2.set_xlabel(f'{dataset1} Self-Correction Blind Spot', fontsize=12)
    ax2.set_ylabel(f'{dataset2} Self-Correction Blind Spot', fontsize=12)
    ax2.set_title(f'Blind Spot correlation: {dataset1} vs {dataset2}\n(r = {corr_coef:.3f})', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add perfect correlation line for reference
    min_val = min(x_data.min(), y_data.min())
    max_val = max(x_data.max(), y_data.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, label='Perfect correlation')
    ax2.legend(loc='best')
    
    # Plot 3: Scatter plot for the strongest correlation (excluding SCLI5 vs GSM8K-SC)
    # Find the pair with highest correlation (excluding diagonal and SCLI5 vs GSM8K-SC)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            # Skip SCLI5 vs GSM8K-SC as it's already plotted
            if not ((pair[0] == 'SCLI5' and pair[1] == 'GSM8K-SC (ACA)') or 
                   (pair[0] == 'GSM8K-SC (ACA)' and pair[1] == 'SCLI5')):
                corr_pairs.append(pair)
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if corr_pairs:
        dataset1, dataset2, corr_coef = corr_pairs[0]
        
        # Get data for this pair
        x_data = df_corr[dataset1]
        y_data = df_corr[dataset2]
        
        # Create scatter plot
        ax3.scatter(x_data, y_data, alpha=0.7, s=80, color='green')
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax3.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
        
        # Add model labels with better positioning
        for _, row_data in df_corr.iterrows():
            model_name = MODEL_LIST.get(row_data['model'], row_data['model'])
            # Shorten model names for better fit
            if len(model_name) > 20:
                model_name = model_name[:17] + "..."
            ax3.annotate(model_name, 
                        (row_data[dataset1], row_data[dataset2]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', 
                                                       facecolor='white', alpha=0.7))
        
        ax3.set_xlabel(f'{dataset1} Self-Correction Blind Spot', fontsize=12)
        ax3.set_ylabel(f'{dataset2} Self-Correction Blind Spot', fontsize=12)
        ax3.set_title(f'Blind Spot correlation: {dataset1} vs {dataset2}\n(r = {corr_coef:.3f}', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add perfect correlation line for reference
        min_val = min(x_data.min(), y_data.min())
        max_val = max(x_data.max(), y_data.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, label='Perfect correlation')
        ax3.legend(loc='best')
    
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(f'output/blind_spot_correlation_aca_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Blind Spot Correlation Analysis (ACA):")
    print("=" * 50)
    print(f"Number of models with complete data: {len(correlation_data)}")
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Calculate average correlation
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper_triangle.stack().mean()
    print(f"\nAverage correlation across datasets: {avg_corr:.3f}")
    
    return df_corr, corr_matrix


if __name__ == "__main__":
    data = get_error_injection_model_data()

    df = construct_data_matrix(data, model_list=NON_REASONING_MODELS)
    macro_avgs = df.mean(axis=1)
    # Sort models by macro average in descending order
    sorted_non_reasoning_models = macro_avgs.sort_values(ascending=False).index.tolist()

    # Default blind spot summary
    data_matrix_mean, data_matrix_sem, models, datasets = plot_blind_spot_summary(sorted_non_reasoning_models, 'non_reasoning')
    print("Blind spot summary of non-reasoning models:")
    print(data_matrix_mean[:,[0, 1, 3]].mean())
    print("\n" + "="*80 + "\n")
    
    # Correlation analysis
    correlation_df, correlation_matrix = plot_blind_spot_correlation(sorted_non_reasoning_models, 'non_reasoning')
    print("\n" + "="*80 + "\n")
    correlation_aca_df, correlation_aca_matrix = plot_blind_spot_correlation(sorted_non_reasoning_models, 'non_reasoning')
    print("\n" + "="*80 + "\n")
    
    # Wait fields blind spot summary
    try:
        data_matrix_mean_wait, data_matrix_sem_wait, models_wait, datasets_wait = plot_blind_spot_summary_wait(sorted_non_reasoning_models, 'non_reasoning')
        print("Result of blind spot reduction acorss dataset:")
        print(data_matrix_mean_wait[:,[0, 1, 3]].mean(axis=1) / data_matrix_mean[:,[0, 1, 3]].mean(axis=1) - 1)
        print("Result of blind spot reduction acorss dataset across all models:")
        print((data_matrix_mean_wait[:,[0, 1, 3]].mean() / data_matrix_mean[:,[0, 1, 3]].mean() - 1))

        print("\n" + "="*80 + "\n")
    except Exception as e:
        print(f"Wait fields analysis failed: {e}")
        print("\n" + "="*80 + "\n")


    df = construct_data_matrix(data, model_list=REASONING_MODELS)
    macro_avgs = df.mean(axis=1)
    # Sort models by macro average in descending order
    sorted_reasoning_models = macro_avgs.sort_values(ascending=False).index.tolist()
    data_matrix_mean, data_matrix_sem, models, datasets = plot_blind_spot_summary(sorted_reasoning_models, 'reasoning')
    print("\n" + "="*80 + "\n")
