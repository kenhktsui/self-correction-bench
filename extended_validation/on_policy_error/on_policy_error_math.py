from collections import Counter
import json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from ollama import generate, chat
from tqdm import tqdm


MODEL_LIST = {
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5:7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2:7b-instruct": "Qwen/Qwen2-7B-Instruct",
}
generation_config = {
    'temperature': 0.0,
    'num_predict': 4096
}


ds = concatenate_datasets([load_dataset("nlile/hendrycks-MATH-benchmark", split='train'), load_dataset("nlile/hendrycks-MATH-benchmark", split='test')])
problem_answer_map = {k: v for k,v in zip(ds['problem'], ds['answer'])}


# def get_question_str(x):
#     return {'question_str': x['problem'] + ' ' + ' '.join(x['steps'][:x['label']])}


def get_question_str(x):
    return {'question_str': x['problem'] + ' ' + ' '.join(x['steps'][:-1])}


# def get_question_str(x):
#     return {'question_str': x['problem'] + ' ' + ' '.join(x['steps'][:x['label']+1])}


def get_answer(x):
    return {'answer': problem_answer_map[x['problem']]}


def check_answer(prediction, answer):
    response = chat(model='gpt-oss:20b', messages=[
        {
            'role': 'user',
            'content': f"Is the following two answers equal? Answer Y/N \n\nPre-generated answer: {prediction}\nGround truth answer: {answer}",
        },
    ])
    return response.message.content



dataset = load_dataset('Qwen/ProcessBench', split='math')
dataset = dataset.filter(lambda x: x['final_answer_correct'] is False)
print(Counter(dataset['final_answer_correct']))

dataset = dataset.map(lambda x: get_question_str(x))
dataset = dataset.map(lambda x: get_answer(x))

if __name__ == "__main__":
    n_total = 0
    n_correct = 0
    for model, hf_tokenizer_name in MODEL_LIST.items():
        dataset_model = dataset.filter(lambda x: x['generator'] == hf_tokenizer_name.split('/')[1])        
        print(model, len(dataset_model))
        for d in tqdm(dataset_model, desc=model):
            tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
            prompt_error_in_user = tokenizer.apply_chat_template([{'role': 'user', 'content': d['question_str']}], 
                                                                 tokenize=False,
                                                                 add_generation_prompt=True,
                                                                 continue_final_message=False, 
                                                                 enable_thinking=False)
            response_error_in_user = generate(
                model=model,
                prompt=prompt_error_in_user,
                options=generation_config
            )
            response = response_error_in_user.response
            checked = check_answer(response, d['answer'])
            if checked == 'Y':
                n_correct += 1
            n_total += 1
            print(f"Total: {n_total}, Correct: {n_correct}, Accuracy: {n_correct/n_total * 100:.2f}%")

            d = {
                "processbench": d['id'],
                "model": MODEL_LIST[model],
                "question_str": d['question_str'],
                "answer": d['answer'],
                "response": response,
                "checked": checked
            }
            with open(f"rebuttal/on_policy_error/on_policy_error_math_v3.jsonl", "a") as f:
                f.write(json.dumps(d) + "\n")
