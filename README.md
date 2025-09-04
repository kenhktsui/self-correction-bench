# Official Repo for Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs

In general, there are 3 steps to reproduce the result in the paper.
1. llm_inference: prompt tested model for completions
2. evaluation_with_llm: evaluate model completion against the ground truth
3. plot/ analysis: produce various graph and analysis in the paper.
 
Additional tests:
- Test different correction markers: run llm_inference and evaluation_with_llm scripts with suffix *_markers.py
- Extending to 4,096 tokens: run llm_inference and evaluation_with_llm scripts with suffix *complete_truncated.py
