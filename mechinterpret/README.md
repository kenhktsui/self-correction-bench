# Mechanistic Interpretability: Authorship Attribution Direction

Demonstrates that transformers encode authorship attribution as an internal representation that **causally gates** self-correction behaviour.

## Experiments
### Experiment 1 — Measure the Activation Difference
For each example run two forward passes and extract the **last-token hidden state** at every layer.  The authorship direction is `d = h_internal − h_external`.
#### Step 1: Extract activation differences
```bash
# Full sweep (run for each model × dataset combination)
for MODEL in llama-8b qwen2.5-7b; do
  for DS in scli5 gsm8k_sc prm800k_sc; do
    python -m mechinterpret.experiment1_run \
        --model $MODEL --dataset $DS \
        --output_dir mechinterpret/results
  done
done
```
#### Step 2: Analyse
```bash
python -m mechinterpret.experiment1_analysis \
    --results_dir mechinterpret/results \
    --output_dir mechinterpret/plots \
    --model_key llama-8b \
    --direction_method mean \
    --target_layer 13 && \
python -m mechinterpret.experiment1_analysis \
    --results_dir mechinterpret/results \
    --output_dir mechinterpret/plots \
    --model_key qwen2.5-7b \
    --direction_method mean \
    --target_layer 13 && \
python -m mechinterpret.experiment1_analysis \
    --results_dir mechinterpret/results \
    --output_dir mechinterpret/plots \
    --direction_method mean \
    --target_layer 13 && \
python -m mechinterpret.plot_summary --output_dir mechinterpret/plots
```

### Experiment 2 — Steer with the Direction
Adds `alpha × direction` to the last-token residual stream. **Negative alpha** (internal-error prompts): moves activations toward external-attribution state → **boosts** self-correction.
#### 2a — Layer sweep (find the most effective layer)

```bash
python -m mechinterpret.experiment2_steering --model qwen2.5-7b --dataset gsm8k_sc --direction_file mechinterpret/results/qwen2.5-7b_gsm8k_sc.npz --sweep_layers --sweep_alpha=-1 --max_examples 60 &&
python -m mechinterpret.experiment2_steering --model llama-8b --dataset gsm8k_sc --direction_file mechinterpret/results/llama-8b_gsm8k_sc.npz --sweep_layers --sweep_alpha=-1 --max_examples 60
```

#### 2b — Alpha sweep (in-distribution + cross-dataset transfer)
Uses the best layer identified by the sweep.

```bash
python -m mechinterpret.experiment2_steering  --model llama-8b  --dataset scli5  --direction_file mechinterpret/results/llama-8b_gsm8k_sc.npz  --alphas="-5,-1"  --target_layers 13 && \
python -m mechinterpret.experiment2_steering  --model qwen2.5-7b  --dataset scli5  --direction_file mechinterpret/results/qwen2.5-7b_gsm8k_sc.npz  --alphas="-5,-1"  --target_layers 13 && \
python -m mechinterpret.experiment2_steering  --model llama-8b  --dataset gsm8k_sc  --direction_file mechinterpret/results/llama-8b_gsm8k_sc.npz  --alphas="-5,-1"  --target_layers 13 && \
python -m mechinterpret.experiment2_steering  --model qwen2.5-7b  --dataset gsm8k_sc  --direction_file mechinterpret/results/qwen2.5-7b_gsm8k_sc.npz  --alphas="-5,-1"  --target_layers 13 && \
python -m mechinterpret.experiment2_steering  --model llama-8b  --dataset prm800k_sc  --direction_file mechinterpret/results/llama-8b_gsm8k_sc.npz  --alphas="-5,-1"  --target_layers 13 && \
python -m mechinterpret.experiment2_steering  --model qwen2.5-7b  --dataset prm800k_sc  --direction_file mechinterpret/results/qwen2.5-7b_gsm8k_sc.npz  --alphas="-5,-1"  --target_layers 13 
```

#### 2c - LLM Evaluation of Steering Results
After running steering experiments, evaluate each result file with the Gemini LLM judge.

```bash
python -m mechinterpret.evaluate_with_llm \
    --input  mechinterpret/results/steering_llama-8b_scli5_internal_13.json \
    --output mechinterpret/results/steering_llama-8b_scli5_internal_13_eval.jsonl
python -m mechinterpret.evaluate_with_llm \
    --input  mechinterpret/results/steering_qwen2.5-7b_scli5_internal_13.json \
    --output mechinterpret/results/steering_qwen2.5-7b_scli5_internal_13_eval.jsonl


python -m mechinterpret.evaluate_with_llm \
    --input  mechinterpret/results/steering_llama-8b_gsm8k_sc_internal_13.json \
    --output mechinterpret/results/steering_llama-8b_gsm8k_sc_internal_13_eval.jsonl
python -m mechinterpret.evaluate_with_llm \
    --input  mechinterpret/results/steering_qwen2.5-7b_gsm8k_sc_internal_13.json \
    --output mechinterpret/results/steering_qwen2.5-7b_gsm8k_sc_internal_13_eval.jsonl


python -m mechinterpret.evaluate_with_llm \
    --input  mechinterpret/results/steering_llama-8b_prm800k_sc_internal_13.json \
    --output mechinterpret/results/steering_llama-8b_prm800k_sc_internal_13_eval.jsonl
python -m mechinterpret.evaluate_with_llm \
    --input  mechinterpret/results/steering_qwen2.5-7b_prm800k_sc_internal_13.json \
    --output mechinterpret/results/steering_qwen2.5-7b_prm800k_sc_internal_13_eval.jsonl
```

Summarise the steering result.
```bash
python -m mechinterpret.summarise_steering
```
