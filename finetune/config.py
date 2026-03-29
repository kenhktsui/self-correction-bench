from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    dataset_name: str = "open-r1/Mixture-of-Thoughts"
    # Sub-config: "all" | "math" | "code" | "science"
    dataset_config: str = "all"
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "finetune/data"
    max_seq_length: int = 4096
    max_examples: Optional[int] = None
    val_size: float = 0.02
    seed: int = 42


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    scale: float = 10.0
    layers: int = 16


@dataclass
class TrainConfig:
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    data_dir: str = "finetune/data"
    adapter_path: str = "finetune/adapters"

    lora_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_scale: float = 10.0
    lora_dropout: float = 0.0

    batch_size: int = 2
    iters: int = 1000
    learning_rate: float = 5e-5
    min_lr_ratio: float = 0.1
    warmup: int = 100
    max_seq_length: int = 4096
    grad_checkpoint: bool = True

    steps_per_eval: int = 100
    steps_per_report: int = 10
    save_every: int = 200
    val_batches: int = 25

    seed: int = 42
