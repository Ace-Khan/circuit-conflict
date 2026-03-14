"""
utils.py — model loading, device detection, logit-lens, token utilities.

All functions are deliberately stateless so they can be imported into any
notebook without side-effects.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: Optional[torch.device] = None) -> HookedTransformer:
    """
    Load GPT-2 Small via TransformerLens.

    The model is moved to `device` (defaults to best available).
    fold_ln=True and center_unembed=True apply standard pre-processing that
    makes activation patching cleaner.
    """
    if device is None:
        device = get_device()

    model = HookedTransformer.from_pretrained(
        "gpt2",
        fold_ln=True,           # fold LayerNorm weights into adjacent weights
        center_writing_weights=True,
        center_unembed=True,
        device=str(device),
    )
    model.eval()
    print(f"Loaded GPT-2 Small on {device}")
    print(f"  n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
          f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")
    return model


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def tokenize(model: HookedTransformer, text: str) -> torch.Tensor:
    """Return a (1, seq_len) int64 tensor on the model's device."""
    return model.to_tokens(text)


def get_answer_token_id(model: HookedTransformer, answer: str) -> int:
    """
    Return the single token id for `answer`.

    Tries the token with a leading space first (common in GPT-2), then without.
    Raises ValueError if the answer tokenises to more than one token under
    both conventions.
    """
    for candidate in [" " + answer.strip(), answer.strip()]:
        ids = model.to_tokens(candidate, prepend_bos=False)[0]
        if ids.shape[0] == 1:
            return ids[0].item()
    raise ValueError(
        f"Answer '{answer}' tokenises to multiple tokens. "
        "Use a single-token answer word (e.g. a pronoun or short noun)."
    )


def decode_top_k(model: HookedTransformer, logits: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
    """Return top-k (token_string, probability) pairs from a vocab-dim logit vector."""
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    return [(model.to_string(idx.item()), prob.item())
            for idx, prob in zip(top.indices, top.values)]


# ---------------------------------------------------------------------------
# Logit difference (primary scalar metric)
# ---------------------------------------------------------------------------

def logit_diff(
    logits: torch.Tensor,
    token_A: int,
    token_B: int,
    position: int = -1,
) -> float:
    """
    Compute logit_A - logit_B at `position` (default: last token).

    Parameters
    ----------
    logits : (batch, seq, vocab) or (seq, vocab)
    token_A, token_B : int  answer token ids
    position : int  sequence position to query

    Returns
    -------
    float  (positive → model prefers A, negative → model prefers B)
    """
    if logits.dim() == 3:
        logits = logits[0]  # remove batch dim
    last = logits[position]  # (vocab,)
    return (last[token_A] - last[token_B]).item()


# ---------------------------------------------------------------------------
# Logit lens (logit difference at every layer)
# ---------------------------------------------------------------------------

def logit_lens_diff(
    model: HookedTransformer,
    cache,
    token_A: int,
    token_B: int,
    position: int = -1,
) -> np.ndarray:
    """
    Apply the logit lens to compute logit_A - logit_B at each residual stream
    layer (after resid_post for layers 0..n_layers-1, then final logits).

    Returns
    -------
    np.ndarray of shape (n_layers + 1,)
        Index i is after layer i's resid_post; index n_layers is the full
        model output (identical to the real output logits).
    """
    n_layers = model.cfg.n_layers
    diffs = []

    for layer in range(n_layers):
        resid = cache["resid_post", layer]          # (batch, seq, d_model)
        # Apply the (folded) final layer norm and unembedding
        resid_ln = model.ln_final(resid)            # (batch, seq, d_model)
        logits = resid_ln @ model.W_U + model.b_U  # (batch, seq, vocab)
        diff = (logits[0, position, token_A] - logits[0, position, token_B]).item()
        diffs.append(diff)

    return np.array(diffs)


# ---------------------------------------------------------------------------
# Utility: find phase-transition layer
# ---------------------------------------------------------------------------

def find_phase_transition(layer_diffs: np.ndarray) -> Optional[int]:
    """
    Return the layer index where the logit-diff curve crosses zero or has its
    largest absolute change.  Returns None if the curve never changes sign.

    Strategy:
    1. If the curve crosses zero, return the first crossing layer.
    2. Otherwise, return the layer with the maximum absolute finite-difference.
    """
    # Zero crossing
    for i in range(len(layer_diffs) - 1):
        if layer_diffs[i] * layer_diffs[i + 1] < 0:
            return i + 1  # layer where it crossed

    # Largest single-layer jump
    deltas = np.abs(np.diff(layer_diffs))
    if deltas.max() > 0:
        return int(np.argmax(deltas)) + 1

    return None
