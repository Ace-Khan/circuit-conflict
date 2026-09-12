"""
utils.py — model loading, device detection, logit lens, token utilities.

All functions are deliberately stateless so they can be imported into any
notebook without side-effects.

Position convention
-------------------
Every position index in this project is computed on the BOS-prepended
tokenisation produced by `model.to_tokens(text)`.  Raw-string character offsets
are never used to derive token indices; positions are recovered by diffing the
token sequences of a minimal pair, which is exact.
"""

from __future__ import annotations

import numpy as np
import torch
from transformer_lens import HookedTransformer

from circuit_conflict import compat

compat.apply_all()


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

def load_model(model_name: str = "gpt2", device: torch.device | None = None,
               verbose: bool = True) -> HookedTransformer:
    """
    Load a model via TransformerLens.

    fold_ln / center_writing_weights / center_unembed apply the standard
    pre-processing that makes activation patching and direct logit attribution
    well-defined.  In particular `center_unembed` is what makes a *difference*
    of unembedding columns the correct readout direction.
    """
    if device is None:
        device = get_device()

    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        device=str(device),
    )
    model.eval()
    if verbose:
        print(f"Loaded {model_name} on {device}")
        print(f"  n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
              f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")
    return model


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def tokenize(model: HookedTransformer, text: str) -> torch.Tensor:
    """Return a (1, seq_len) int64 tensor on the model's device (BOS prepended)."""
    return model.to_tokens(text)


def is_single_token(model: HookedTransformer, word: str, with_space: bool = True) -> bool:
    """
    True if `word` is a single GPT-2 token.

    This is the predicate the dataset generators filter their vocabularies with,
    so that single-token-ness is guaranteed *by construction* rather than
    discovered at scoring time.
    """
    text = (" " + word.strip()) if with_space else word.strip()
    return model.to_tokens(text, prepend_bos=False).shape[1] == 1


def resolve_answer(model: HookedTransformer, answer: str) -> dict[str, object]:
    """
    Non-raising answer resolution.

    Returns {"ids": [...], "n_tokens": int, "first_id": int, "single": bool}.

    Prefers the leading-space form (the usual GPT-2 continuation convention) and
    falls back to the bare form.  Callers that require a single token should use
    `require_single_token`; nothing in the analysis path should silently skip an
    item, which is what the old raise-and-`continue` pattern did.
    """
    for text in (" " + answer.strip(), answer.strip()):
        ids = model.to_tokens(text, prepend_bos=False)[0].tolist()
        if len(ids) == 1:
            return {"ids": ids, "n_tokens": 1, "first_id": ids[0], "single": True}
    ids = model.to_tokens(" " + answer.strip(), prepend_bos=False)[0].tolist()
    return {"ids": ids, "n_tokens": len(ids), "first_id": ids[0], "single": False}


def require_single_token(model: HookedTransformer, answer: str) -> int:
    """Return the single token id for `answer`, or raise.  Builder-time only."""
    spec = resolve_answer(model, answer)
    if not spec["single"]:
        raise ValueError(
            f"Answer {answer!r} tokenises to {spec['n_tokens']} tokens. "
            "Generators must draw answers from a single-token-filtered vocabulary."
        )
    return int(spec["first_id"])


# Backwards-compatible alias (the old name, now delegating to the strict form).
def get_answer_token_id(model: HookedTransformer, answer: str) -> int:
    return require_single_token(model, answer)


def decode_top_k(model: HookedTransformer, logits: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
    """Return top-k (token_string, probability) pairs from a vocab-dim logit vector."""
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    return [(model.to_string(idx.item()), prob.item())
            for idx, prob in zip(top.indices, top.values)]


# ---------------------------------------------------------------------------
# Minimal-pair position recovery
# ---------------------------------------------------------------------------

def token_diff_positions(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> list[int]:
    """
    Indices where two equal-length token sequences differ.

    This is how `p_slot` is recovered for a minimal pair: exactly, from the
    tokenisation itself, with no character-offset arithmetic.
    """
    if tokens_a.shape != tokens_b.shape:
        raise ValueError(
            f"Token shapes differ: {tuple(tokens_a.shape)} vs {tuple(tokens_b.shape)}. "
            "Minimal pairs must tokenise to the same length."
        )
    a = tokens_a.flatten()
    b = tokens_b.flatten()
    return (a != b).nonzero(as_tuple=True)[0].tolist()


def find_token_position(tokens: torch.Tensor, token_id: int, last: bool = True) -> int | None:
    """
    Position of `token_id` in a (1, seq) or (seq,) tensor.

    `last=True` returns the final occurrence, which is what mention-position
    metrics want (the most recent mention before the decision site).
    """
    flat = tokens.flatten()
    hits = (flat == token_id).nonzero(as_tuple=True)[0].tolist()
    if not hits:
        return None
    return hits[-1] if last else hits[0]


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
    logit_A - logit_B at `position` (default: last token).

    Positive -> model prefers A; negative -> model prefers B.
    Accepts (batch, seq, vocab) or (seq, vocab); batch dim must be 1.
    """
    if logits.dim() == 3:
        if logits.shape[0] != 1:
            raise ValueError("logit_diff expects batch size 1; use logit_diff_batched.")
        logits = logits[0]
    last = logits[position]
    return (last[token_A] - last[token_B]).item()


def logit_diff_batched(
    logits: torch.Tensor,
    token_A: torch.Tensor,
    token_B: torch.Tensor,
    position: int = -1,
) -> torch.Tensor:
    """
    Batched logit difference.

    logits    : (batch, seq, vocab)
    token_A/B : (batch,) int64
    Returns   : (batch,) float
    """
    sel = logits[:, position, :]
    a = sel.gather(1, token_A.unsqueeze(1)).squeeze(1)
    b = sel.gather(1, token_B.unsqueeze(1)).squeeze(1)
    return a - b


# ---------------------------------------------------------------------------
# Answer-difference readout direction (for direct logit attribution)
# ---------------------------------------------------------------------------

def answer_direction(model: HookedTransformer, token_A: int, token_B: int) -> torch.Tensor:
    """
    The residual-stream direction whose dot product is (logit_A - logit_B).

    Valid because `center_unembed=True` removes the shared component that would
    otherwise make individual unembedding columns non-comparable.
    """
    return model.W_U[:, token_A] - model.W_U[:, token_B]


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
    Logit-lens logit_A - logit_B after each layer's resid_post.

    Returns
    -------
    np.ndarray of shape (n_layers,)
        Index i is the readout after layer i.  Index n_layers-1 corresponds to
        the final residual stream and matches the model's real output logits.

    Note: the residual stream is sliced to `position` *before* the unembedding,
    so this costs one (d_model,) @ (d_model,) dot per layer rather than a full
    (seq x d_model x vocab) matmul.
    """
    n_layers = model.cfg.n_layers
    direction = answer_direction(model, token_A, token_B)
    # The unembedding bias is part of the logit difference. Omitting it shifts the
    # whole curve by a constant, which moves the zero crossing that
    # detect_phase_transition keys on.
    bias = (model.b_U[token_A] - model.b_U[token_B]).item()
    diffs = []

    for layer in range(n_layers):
        resid = cache["resid_post", layer][0, position]
        resid_ln = model.ln_final(resid.unsqueeze(0).unsqueeze(0))[0, 0]
        diffs.append((resid_ln @ direction).item() + bias)

    return np.array(diffs)
