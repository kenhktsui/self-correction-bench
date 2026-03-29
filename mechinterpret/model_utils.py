"""Model loading and activation extraction utilities.

For each example, we run two forward passes and extract residual-stream activations
at the **last token of the error text** — the same content position in both prompts.

  Internal (continue_final_message=True):
      `...<assistant header>\\n\\nWRONG_ANSWER`
      Last error token = position -1  (error is the final content)

  External (add_generation_prompt=False):
      `...<user header>\\n\\nQ WRONG_ANSWER<turn_end>`
      Last error token = last position before the turn-end special/whitespace tokens
      Found by walking backwards past special tokens and whitespace.

Aligning both prompts at the same content position isolates the authorship-attribution
signal from structural differences (generation header vs turn-end tag).

The authorship-attribution difference direction is:
    d = h_internal - h_external
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechinterpret.config import DEVICE


def compute_direction(
    differences: np.ndarray,
    method: str = "mean",
    top_k: int = 1,
) -> np.ndarray:
    """Compute the authorship direction from per-example activation differences.
    """
    if top_k > 1:
        method = "pca"

    if method == "mean":
        return differences.mean(axis=0)

    if method == "pca":
        from sklearn.decomposition import PCA
        N, L, H = differences.shape
        direction = np.zeros((L, H), dtype=np.float32)
        for l in range(L):
            X = differences[:, l, :].astype(np.float32)   # (N, H)
            mean_diff = X.mean(axis=0)                     # (H,)
            pca = PCA(n_components=min(top_k, min(N, H)))
            pca.fit(X)
            # components_: (top_k, H), rows are PCs (from centered SVD)
            # Project mean_diff onto the top-k subspace.
            # Each PC is sign-aligned with mean_diff before projection.
            d = np.zeros(H, dtype=np.float32)
            for pc in pca.components_:
                if np.dot(pc, mean_diff) < 0:
                    pc = -pc
                d += np.dot(mean_diff, pc) * pc
            direction[l] = d
        return direction

    raise ValueError(f"Unknown direction method '{method}'. Choose 'mean' or 'pca'.")


def load_model_and_tokenizer(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = DEVICE,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def _build_prompt(
    tokenizer: AutoTokenizer,
    messages: List[dict],
    add_generation_prompt: bool,
    continue_final_message: bool,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )


def _last_error_token_pos(tokenizer: AutoTokenizer, prompt: str) -> int:
    """Return the index of the last error-content token in a no-gen prompt.

    The prompt ends with turn-end tokens that vary by model family:
      LLaMA 3.x / DeepSeek-R1-Distill-Llama:  ...<error><|eot_id|>
      Qwen 2.x:                                ...<error><|im_end|>\\n

    Walk backwards past any trailing special tokens and whitespace-only tokens
    to find the last real content token (= last token of the error text).
    """
    token_ids = tokenizer.encode(prompt)
    i = len(token_ids) - 1
    while i >= 0:
        is_special = token_ids[i] in tokenizer.all_special_ids
        is_whitespace = tokenizer.decode([token_ids[i]]).strip() == ""
        if is_special or is_whitespace:
            i -= 1
        else:
            return i
    raise ValueError("No content token found in prompt — check chat template output.")


@torch.no_grad()
def _extract_at_pos(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    pos: int,
) -> np.ndarray:
    """Forward pass; return hidden states at token `pos` for every layer.

    Returns ndarray of shape (n_layers + 1, hidden_size).
    Index 0 = embedding layer; indices 1..n_layers = transformer layers.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    stacked = torch.stack([h[0, pos, :] for h in outputs.hidden_states])
    return stacked.float().cpu().numpy()


def extract_activation_difference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    internal_messages: List[dict],
    external_messages: List[dict],
) -> np.ndarray:
    """Per-example activation difference (internal − external) at the last error token.

    Both prompts are built with add_generation_prompt=False so neither has a trailing
    generation header.  _last_error_token_pos walks backwards past turn-end special
    tokens / whitespace to find the last real content token in each, giving a
    content-aligned comparison regardless of tokenizer version.

      Internal (continue_final_message=True): ideally no closing tag → pos = last token.
        Some tokenizer versions still append <|eot_id|>; _last_error_token_pos handles both.
      External (continue_final_message=False): ends with turn-end tag → pos = token before it.

    Returns ndarray of shape (n_layers + 1, hidden_size).
    """
    prompt_internal = _build_prompt(
        tokenizer, internal_messages,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    pos_internal = _last_error_token_pos(tokenizer, prompt_internal)

    prompt_external = _build_prompt(
        tokenizer, external_messages,
        add_generation_prompt=False,
        continue_final_message=False,
    )
    pos_external = _last_error_token_pos(tokenizer, prompt_external)

    # Sanity check: both positions must resolve to the same token id.
    # A mismatch means the error text was tokenised differently across the two
    # templates (e.g. BPE boundary shift), so the comparison would be invalid.
    tok_id_internal = tokenizer.encode(prompt_internal)[pos_internal]
    tok_id_external = tokenizer.encode(prompt_external)[pos_external]
    assert tok_id_internal == tok_id_external, (
        f"Last error token mismatch: internal={repr(tokenizer.decode([tok_id_internal]))} "
        f"(id={tok_id_internal}) vs "
        f"external={repr(tokenizer.decode([tok_id_external]))} (id={tok_id_external}). "
        "The error text may be tokenised differently in the two prompt templates."
    )

    h_internal = _extract_at_pos(model, tokenizer, prompt_internal, pos=pos_internal)
    h_external = _extract_at_pos(model, tokenizer, prompt_external, pos=pos_external)

    return h_internal - h_external


def extract_dataset_differences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[dict],
    show_progress: bool = True,
) -> dict:
    """Extract per-example activation differences for a list of examples.

    Args:
        examples: list of dicts with keys 'id', 'internal_messages', 'external_messages'.

    Returns:
        dict with:
            'differences': np.ndarray (n_examples, n_layers+1, hidden_size)
            'ids':         list of example ids
    """
    all_diffs = []
    ids = []
    iterator = tqdm(examples) if show_progress else examples

    for ex in iterator:
        try:
            diff = extract_activation_difference(
                model, tokenizer,
                ex["internal_messages"],
                ex["external_messages"],
            )
            all_diffs.append(diff)
            ids.append(ex["id"])
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] Skipped example {ex.get('id')}: {e}")

    return {
        "differences": np.stack(all_diffs, axis=0),  # (N, n_layers+1, H)
        "ids": ids,
    }


# ---------------------------------------------------------------------------
# Utilities for Experiment 2 (activation steering)
# ---------------------------------------------------------------------------

class SteeringHook:
    """PyTorch forward hook that adds a direction vector to the last prompt token.

    Steering is applied as:  h += alpha * d  (unnormalised, consistent with ActAdd).
    alpha is the injection coefficient; its effective scale depends on ||d||.
    """

    def __init__(self, direction: torch.Tensor, alpha: float, prompt_len: int) -> None:
        self.direction = direction.float()
        self.alpha = alpha
        self.prompt_len = prompt_len
        self._applied = False

    def __call__(self, module, input, output):
        if self._applied:
            return output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if hidden_states.shape[1] >= self.prompt_len:
            hidden_states = hidden_states.clone()
            direction_dev = self.direction.to(hidden_states.device, hidden_states.dtype)
            hidden_states[:, self.prompt_len - 1, :] += self.alpha * direction_dev
            self._applied = True

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


@torch.no_grad()
def generate_batch_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages_batch: List[List[dict]],
    direction: np.ndarray,
    alpha: float,
    add_generation_prompt: bool = False,
    continue_final_message: bool = True,
    target_layers: Optional[List[int]] = None,
    max_new_tokens: int = 1024,
) -> List[str]:
    """Batched generate with activation steering using left-padding.

    With left-padding all sequences share the same padded length, so the last
    token (position prompt_len-1) is the last *real* token for every sequence
    in the batch — the correct position to apply the steering direction.

    For internal-error prompts (default):
        add_generation_prompt=False, continue_final_message=True
        alpha < 0  moves toward external attribution  → boosts correction
    For external-error prompts:
        add_generation_prompt=True,  continue_final_message=False
        alpha > 0  moves toward internal attribution  → suppresses correction
    """
    prompts = [
        _build_prompt(tokenizer, msgs,
                      add_generation_prompt=add_generation_prompt,
                      continue_final_message=continue_final_message)
        for msgs in messages_batch
    ]
    device = next(model.parameters()).device

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = orig_padding_side

    prompt_len = inputs["input_ids"].shape[1]

    if target_layers is None:
        n_layers = len(model.model.layers)
        target_layers = [int(0.75 * n_layers)]  # 75% depth, matches exp1 analysis layer

    handles = []
    for layer_idx in target_layers:
        dir_idx = layer_idx + 1
        if dir_idx >= direction.shape[0]:
            continue
        hook = SteeringHook(
            torch.tensor(direction[dir_idx], dtype=torch.float32),
            alpha,
            prompt_len,
        )
        handles.append(model.model.layers[layer_idx].register_forward_hook(hook))

    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        for h in handles:
            h.remove()

    return [
        tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
        for i in range(len(prompts))
    ]


@torch.no_grad()
def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    internal_messages: List[dict],
    direction: np.ndarray,
    alpha: float,
    target_layers: Optional[List[int]] = None,
    max_new_tokens: int = 256,
) -> str:
    """Generate from an internal-error prompt with activation steering.

    alpha < 0 moves activations toward external-attribution state (expected to
    boost self-correction on internal errors).
    """
    prompt = _build_prompt(
        tokenizer, internal_messages,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if target_layers is None:
        n_layers = len(model.model.layers)
        target_layers = [int(0.75 * n_layers)]  # 75% depth, matches exp1 analysis layer

    handles = []
    for layer_idx in target_layers:
        dir_idx = layer_idx + 1  # 0 = embedding, 1..n = transformer layers
        if dir_idx >= direction.shape[0]:
            continue
        hook = SteeringHook(
            torch.tensor(direction[dir_idx], dtype=torch.float32),
            alpha,
            prompt_len,
        )
        handles.append(model.model.layers[layer_idx].register_forward_hook(hook))

    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        for h in handles:
            h.remove()

    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
