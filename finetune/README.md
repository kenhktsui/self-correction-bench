# Finetunign with LoRA
python -m finetune.train --prepare-data --sample-fraction 0.1 --epochs 1

# Inference
python -m finetune.run_bench --dataset scli5 && \
python -m finetune.run_bench --dataset gsm8k_sc && \
python -m finetune.run_bench --dataset prm800k_sc

# Result (after running evaluation_with_llm)
python -m finetune.summarise_finetune
