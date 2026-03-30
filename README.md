# Official Repo for Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs


Below are the steps to reproduce the result in the paper.
| Result| Module|
|---|---|
| Main empirical result | llm_inference: 1. prompt tested model for completions |
|  | evaluation_with_llm: 2. evaluate model completion against the ground truth |
|  | plot/ analysis: 3. produce various graph and analysis in the paper |
| Other domains, close-source model, on-policy error | extended_validation |
| Finetuning with LoRA on self-correction data | finetune (see README in submodule) |
| Mechanistical Analysis | mechinterpret (see README in submodule) |
| **Sensitivity Analysis** | |
| Test different correction markers | run llm_inference and evaluation_with_llm scripts with suffix \*\_markers.py |
| Extending to 4,096 tokens | run llm_inference and evaluation_with_llm scripts with suffix \*complete\_truncated.py |