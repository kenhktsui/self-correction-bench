import re
import json
from collections import Counter
import pandas as pd
from plot.constants import REASONING_MODELS
from evaluation.evaluate_tool import get_is_correct_answer


PROBLEM_ID_WITH_ERROR = [467, 361, 499, 857, 962, 1001]


def get_most_common_first_word_in_correct_answer(data_file,model_name, eval_key="llm_evaluation_bca", response_key="response_error_injection_in_model_bca"):
    data = []
    with open(data_file, "r") as f:
        for line in f:
            d = json.loads(line)
            if model_name.endswith("_thinking"):
                model_name = model_name[:-9]
                enable_thinking = True
            else:
                enable_thinking = False

            if data_file.startswith("scli5") and d['question_type'] in ["counting_letter", "counting_digit"]:
                continue

            if data_file.startswith("gsm8k") and d['id'] in PROBLEM_ID_WITH_ERROR:
                continue

            if not get_is_correct_answer(d, eval_key):
                continue

            if d['model'] == model_name and enable_thinking == d.get('enable_thinking', False):
                t = [re.sub(r'\n', ' ', w).strip() for w in d[response_key].split(" ") if w.strip()]
                if t:
                    data.append(t[0])

    most_common_words = [(word, round(freq/len(data), 3)) for word, freq in Counter(data).most_common(10)]
    return most_common_words[0] if most_common_words else ["", ""]


df = []
for m in REASONING_MODELS:
    df.append([
        m,
        get_most_common_first_word_in_correct_answer("scli5_completion_results_llm_eval_gemini2_5_flash_0_0.jsonl", m, eval_key="llm_evaluation", response_key="response_error_injection_in_model"),
        get_most_common_first_word_in_correct_answer("gsm8k_sc_completion_results_llm_eval_gemini2_5_flash_0_0.jsonl", m, eval_key="llm_evaluation_bca", response_key="response_error_injection_in_model_bca"),
        get_most_common_first_word_in_correct_answer("prm800k_sc_completion_results_llm_eval_gemini2_5_flash_0_0.jsonl", m, eval_key="llm_evaluation_bca", response_key="response_error_injection_in_model_bca")
    ])

print(df[0])
df = pd.DataFrame(df, columns=["model", "scli5", "gsm8k_sc", "prm800k_sc"])
print(df.to_latex(index=False))
