from datasets import load_dataset
from functools import partial
import pandas as pd
from plot.plot_correction_marker import KEYWORDS_RE


def inspect_ds(message_key, content_key, role_key, role_assistant_value, d_list):
    num_keywords = []
    length_list = []
    for d in d_list[message_key]:
        keywords_cnt = 0
        length_cnt = 0
        for m in d:
            if m[role_key] != role_assistant_value:
                continue
            keywords_cnt += len(KEYWORDS_RE.findall(m[content_key]))
            length_cnt += len(m[content_key])
        num_keywords.append(keywords_cnt)
        length_list.append(length_cnt)
    return {"num_keywords": num_keywords, "length_list": length_list}


assert inspect_ds('messages', 'content', 'role', 'assistant', {
        "messages":[
            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm good, thank you!"},
            ],
            [
                {"role": "user", "content": "Wait! I'm not ready!"},
                {"role": "assistant", "content": "Hold on, I'm coming! Alternatively, I can come later. No, let me come now."},
            ]
        ]
    })["num_keywords"] == [0, 3]


inspect_open_thought_ds = inspect_infinity_ds = inspect_oh_ds = partial(inspect_ds, 'conversations', 'value', 'from', 'gpt')
inspect_tulu_ds = inspect_mixture_of_thougts_ds = partial(inspect_ds, 'messages', 'content', 'role', 'assistant')
inspect_ultra_ds = partial(inspect_ds, 'chosen', 'content', 'role', 'assistant')


def inspect_s1k(d_list):
    num_keywords = []
    length_list = []
    for d in d_list['gemini_thinking_trajectory']:
      num_keywords.append(len(KEYWORDS_RE.findall(d)))
      length_list.append(len(d))
    return {"num_keywords": num_keywords, "length_list": length_list}

def inspect_oa(d_list):
    num_keywords = []
    length_list = []
    for d in d_list['text']:
      num_keywords.append(len(KEYWORDS_RE.findall(d)))
      length_list.append(len(d))
    return {"num_keywords": num_keywords, "length_list": length_list}


open_thought_ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split='train')
open_thought_ds = open_thought_ds.map(inspect_open_thought_ds, batched=True)
print(pd.DataFrame.from_dict(
    {
        "OpenThoughts3-1.2M_keywords": open_thought_ds["num_keywords"],
        "OpenThoughts3-1.2M_length": open_thought_ds["length_list"],
    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())


mixture_of_thougts_ds = load_dataset("open-r1/Mixture-of-Thoughts", "all", split='train')
mixture_of_thougts_ds = mixture_of_thougts_ds.map(inspect_mixture_of_thougts_ds, batched=True)
print(pd.DataFrame.from_dict(
    {
        "Mixture-of-Thoughts_keywords": mixture_of_thougts_ds["num_keywords"],
        "Mixture-of-Thoughts_length": mixture_of_thougts_ds["length_list"],

    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())


tulu_ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture", split='train', cache_dir=CACHE_DIR)
tulu_ds = tulu_ds.map(inspect_tulu_ds, batched=True)
print(pd.DataFrame.from_dict(
    {
        "Tulu3-sft-olmo-2-mixture_keywords": tulu_ds["num_keywords"],
        "Tulu3-sft-olmo-2-mixture_length": tulu_ds["length_list"],
    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())

oh_ds = load_dataset("teknium/OpenHermes-2.5", split='train')
oh_ds = oh_ds.map(inspect_oh_ds, batched=True)
print(pd.DataFrame.from_dict(
    {
        "OpenHermes2.5_keywords": oh_ds["num_keywords"],
        "OpenHermes2.5_length": oh_ds["length_list"],
    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())


s1k_ds = load_dataset("simplescaling/s1K-1.1", split="train")
s1k_ds = s1k_ds.map(inspect_s1k, batched=True)
print(pd.DataFrame.from_dict(
    {
        "s1K-1.1_keywords": s1k_ds["num_keywords"],
        "s1K-1.1_length": s1k_ds["length_list"],
    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())


oa_ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
oa_ds = oa_ds.map(inspect_oa, batched=True)
print(pd.DataFrame.from_dict(
    {
        "OpenAssistant_keywords": oa_ds["num_keywords"],
        "OpenAssistant_length": oa_ds["length_list"],
    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())


infinity_ds = load_dataset("BAAI/Infinity-Instruct", "7M", split='train')
infinity_ds = infinity_ds.map(inspect_infinity_ds, batched=True)
print(pd.DataFrame.from_dict(
    {
        "Infinity-Instruct-7M_keywords": infinity_ds["num_keywords"],
        "Infinity-Instruct-7M_length": infinity_ds["length_list"],
    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())


ultra_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split='train_sft')
ultra_ds = ultra_ds.map(inspect_ultra_ds, batched=True)
print(pd.DataFrame.from_dict(
    {
        "UltraFeedback_keywords": ultra_ds["num_keywords"],
        "UltraFeedback_lengtH": ultra_ds["length_list"],
    }
).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1).astype(str).to_latex())
