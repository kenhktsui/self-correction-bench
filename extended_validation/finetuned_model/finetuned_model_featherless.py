from collections import Counter
import json
import re
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from ollama import generate, chat
from tqdm import tqdm
from openai import OpenAI
from llm_inference.prompt import get_prompt_eos_token



MODEL_LIST = {
    # "open-thoughts/OpenThinker3-7B": "open-thoughts/OpenThinker3-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}



client = OpenAI(
  base_url="https://api.featherless.ai/v1",
  api_key=os.environ.get("FEATHERLESS_API_KEY"),
  timeout=180,
  max_retries=3,
)

TEMPERATURE = 0.0

def get_prompt_and_response(cliemt, model, hf_tokenizer_name, messages, add_generation_prompt, continue_final_message, enable_thinking):
    prompt, eos_token = get_prompt_eos_token(model,
                                             hf_tokenizer_name,
                                             messages,
                                             add_generation_prompt,
                                             continue_final_message,
                                             enable_thinking
                                            )
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1024,
        temperature=TEMPERATURE,
        stop=[eos_token]
    )
    return prompt, response


def check_answer(prediction, answer):
    response = chat(model='gpt-oss:20b', messages=[
        {
            'role': 'user',
            'content': f"Is the following two answers equal? Answer Y/N \n\nPre-generated answer: {prediction}\nGround truth answer: {answer}",
        },
    ])
    return response.message.content


dataset_scli5 = load_dataset("super-brown/scli5", split="test")
dataset_gsm8k = load_dataset("super-brown/gsm8k_sc", split="test")
dataset_gsm8k = dataset_gsm8k.map(lambda row: {"correct_answer": re.sub(re.escape("####"), "", row["answer"].split("\n")[-1]).strip()})
dataset_prm800k = load_dataset("super-brown/prm800k_sc", split="test")
dataset_prm800k = dataset_prm800k.rename_column("ground_truth_answer", "correct_answer")
dataset_scli5 = dataset_scli5.add_column("dataset", ["scli5"] * len(dataset_scli5))
dataset_gsm8k = dataset_gsm8k.add_column("dataset", ["gsm8k"] * len(dataset_gsm8k))
dataset_prm800k = dataset_prm800k.add_column("dataset", ["prm800k"] * len(dataset_prm800k))
dataset = concatenate_datasets([
    # dataset_scli5, 
    dataset_gsm8k, 
    # dataset_prm800k
    ])


if __name__ == "__main__":
    n_total = 0
    n_correct = 0
    for model, hf_tokenizer_name in MODEL_LIST.items():
        for d in tqdm(dataset, desc=model):
            key = 'messages_error_injection_in_model' if d['dataset'] == "scli5" else 'messages_error_injection_in_model_bca'
            prompt_error_injection_in_model, response_error_injection_in_model = get_prompt_and_response(client, model, hf_tokenizer_name, d[key],
                                                                                                                False,
                                                                                                                True,
                                                                                                                True)                                                                                                         
            response_error_injection_in_model = response_error_injection_in_model.choices[0].text

            key = 'messages_error_in_user' if d['dataset'] == "scli5" else 'messages_error_in_user_bca'
            prompt_error_in_user, response_error_in_user = get_prompt_and_response(client, model, hf_tokenizer_name, d[key],
                                                                                    True,
                                                                                    False,
                                                                                    True)
            response_error_in_user = response_error_in_user.choices[0].text

            checked_error_injection_in_model = check_answer(response_error_injection_in_model, d['correct_answer'])
            checked_error_in_user = check_answer(response_error_in_user, d['correct_answer'])
    
            d = {
                "dataset": d['dataset'],
                "id": d['id'],
                "question": d['question'],
                "model": MODEL_LIST[model],
                "correct_answer": d['correct_answer'],
                "prompt_error_injection_in_model": prompt_error_injection_in_model,
                "prompt_error_in_user": prompt_error_in_user,
                "response_error_injection_in_model": response_error_injection_in_model,
                "response_error_in_user": response_error_in_user,
                "enable_thinking": True,
                "temperature": TEMPERATURE,
                "checked_error_injection_in_model": checked_error_injection_in_model,
                "checked_error_in_user": checked_error_in_user
            }
            with open(f"rebuttal/finetuned_model/finetuned_model_featherless.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")
