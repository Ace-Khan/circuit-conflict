"""
patching.py — activation patching and ablation hooks for TransformerLens.

Design principles
-----------------
* All patching functions return plain floats or numpy arrays — no hidden state.
* Every function documents exactly which hook name it touches.
* Batch size is always 1 (single-prompt experiments).

Hook names used (TransformerLens)
-----------------------------------
"z"           → blocks.{L}.attn.hook_z         shape (1, seq, n_heads, d_head)
"result"      → blocks.{L}.attn.hook_result    shape (1, seq, n_heads, d_model)
"resid_post"  → blocks.{L}.hook_resid_post     shape (1, seq, d_model)
"attn_out"    → blocks.{L}.hook_attn_out       shape (1, seq, d_model)
"mlp_out"     → blocks.{L}.hook_mlp_out        shape (1, seq, d_model)
"""

from __future__ import annotations

from functools import partial
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer, utils as tl_utils


# ---------------------------------------------------------------------------
# Low-level hook factories
# ---------------------------------------------------------------------------

def _make_zero_ablate_hook(head_idx: int):
    """Zero out a single attention head's z (value-weighted output) vector."""
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        # value: (batch, seq, n_heads, d_head)
        value[:, :, head_idx, :] = 0.0
        return value
    return hook_fn


def _make_patch_head_hook(clean_z: torch.Tensor, head_idx: int):
    """Replace a single head's z with activations from the clean run."""
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, :, head_idx, :] = clean_z[:, :, head_idx, :]
        return value
    return hook_fn


def _make_patch_resid_hook(clean_resid: torch.Tensor):
    """Replace the full residual stream at this position with a clean value."""
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:] = clean_resid
        return value
    return hook_fn


# ---------------------------------------------------------------------------
# Per-head causal contribution (activation patching)
# ---------------------------------------------------------------------------

def head_contribution_map(
    model: HookedTransformer,
    tokens_clean: torch.Tensor,
    tokens_corrupted: torch.Tensor,
    token_A: int,
    token_B: int,
    position: int = -1,
) -> np.ndarray:
    """
    Compute the causal contribution of every attention head (layer × head)
    by activation patching.

    For each head (l, h):
      1. Run corrupted prompt → baseline logit diff
      2. Patch head (l, h) z activations from clean run into corrupted run
      3. Δ = patched logit diff - baseline logit diff
         (positive → this head causally helps answer A win)

    Returns
    -------
    np.ndarray of shape (n_layers, n_heads)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # ---- clean run ----
    with torch.no_grad():
        _, cache_clean = model.run_with_cache(
            tokens_clean,
            names_filter=lambda n: n.endswith("hook_z"),
        )

    # ---- corrupted baseline ----
    with torch.no_grad():
        logits_corr = model(tokens_corrupted)
    baseline_diff = _ld(logits_corr, token_A, token_B, position)

    contributions = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        clean_z = cache_clean[tl_utils.get_act_name("z", layer)]  # (1, seq, n_heads, d_head)

        for head in range(n_heads):
            patch_hook = _make_patch_head_hook(clean_z, head)
            hook_name = tl_utils.get_act_name("z", layer)

            with torch.no_grad():
                logits_patched = model.run_with_hooks(
                    tokens_corrupted,
                    fwd_hooks=[(hook_name, patch_hook)],
                )
            patched_diff = _ld(logits_patched, token_A, token_B, position)
            contributions[layer, head] = patched_diff - baseline_diff

    return contributions


# ---------------------------------------------------------------------------
# Zero-ablation of a single head
# ---------------------------------------------------------------------------

def ablate_head_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    token_A: int,
    token_B: int,
    layer: int,
    head: int,
    position: int = -1,
) -> float:
    """
    Run the model with head (layer, head) zeroed out.
    Returns logit_A - logit_B at `position`.
    """
    hook_name = tl_utils.get_act_name("z", layer)
    zero_hook = _make_zero_ablate_hook(head)

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, zero_hook)],
        )
    return _ld(logits, token_A, token_B, position)


def ablate_head_layer_curve(
    model: HookedTransformer,
    tokens: torch.Tensor,
    token_A: int,
    token_B: int,
    layer_abl: int,
    head_abl: int,
    position: int = -1,
) -> np.ndarray:
    """
    Compute the logit-lens diff curve (one value per layer) with a single
    head ablated.  Returns np.ndarray of shape (n_layers,).
    """
    from circuit_conflict.utils import logit_lens_diff

    hook_name = tl_utils.get_act_name("z", layer_abl)
    zero_hook = _make_zero_ablate_hook(head_abl)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            fwd_hooks=[(hook_name, zero_hook)],
        )
    return logit_lens_diff(model, cache, token_A, token_B, position)


# ---------------------------------------------------------------------------
# Full ablation sweep (all 144 heads) → flip rate
# ---------------------------------------------------------------------------

def ablation_flip_rate(
    model: HookedTransformer,
    prompt_rows,          # iterable of dataset rows (pandas)
    layer: int,
    head: int,
) -> Dict[str, float]:
    """
    For every conflict prompt in `prompt_rows`, check whether ablating
    head (layer, head) flips the model's answer.

    Returns
    -------
    dict with keys:
        "flip_rate"      : fraction of prompts where answer flips
        "n_prompts"      : total number of prompts tested
        "n_flipped"      : absolute count
        "mean_diff_orig" : mean logit diff without ablation
        "mean_diff_abl"  : mean logit diff with ablation
    """
    from circuit_conflict.utils import get_answer_token_id

    n_flipped = 0
    n_total = 0
    diffs_orig, diffs_abl = [], []

    for _, row in prompt_rows.iterrows():
        try:
            tok_A = get_answer_token_id(model, row["answer_A"])
            tok_B = get_answer_token_id(model, row["answer_B"])
        except ValueError:
            continue

        tokens = model.to_tokens(row["prompt_text"])

        with torch.no_grad():
            logits_orig = model(tokens)
        diff_orig = _ld(logits_orig, tok_A, tok_B)

        diff_abl = ablate_head_logit_diff(model, tokens, tok_A, tok_B, layer, head)

        # Flip = sign change
        if (diff_orig > 0) != (diff_abl > 0):
            n_flipped += 1

        diffs_orig.append(diff_orig)
        diffs_abl.append(diff_abl)
        n_total += 1

    return {
        "flip_rate": n_flipped / n_total if n_total else float("nan"),
        "n_prompts": n_total,
        "n_flipped": n_flipped,
        "mean_diff_orig": float(np.mean(diffs_orig)) if diffs_orig else float("nan"),
        "mean_diff_abl": float(np.mean(diffs_abl)) if diffs_abl else float("nan"),
    }


# ---------------------------------------------------------------------------
# Activation patching: losing circuit into arbitration head position
# ---------------------------------------------------------------------------

def patch_losing_circuit_into_head(
    model: HookedTransformer,
    tokens_conflict: torch.Tensor,
    tokens_losing_unamb: torch.Tensor,
    token_A: int,
    token_B: int,
    layer: int,
    head: int,
    position: int = -1,
) -> Dict[str, float]:
    """
    Patch the z activation of head (layer, head) from the *losing* circuit's
    unambiguous run into the conflict run.

    If the conflict run prefers A, the "losing" circuit is the one that
    would produce B, so tokens_losing_unamb should be a prompt that
    strongly produces B.

    Returns
    -------
    dict:
        "diff_conflict"  : logit diff on the raw conflict run
        "diff_patched"   : logit diff after patching
        "answer_flipped" : bool
    """
    # Clean run for losing circuit activations
    with torch.no_grad():
        _, cache_losing = model.run_with_cache(
            tokens_losing_unamb,
            names_filter=lambda n: n.endswith("hook_z"),
        )

    # Normal conflict run
    with torch.no_grad():
        logits_orig = model(tokens_conflict)
    diff_orig = _ld(logits_orig, token_A, token_B, position)

    # Patched conflict run
    losing_z = cache_losing[tl_utils.get_act_name("z", layer)]
    patch_hook = _make_patch_head_hook(losing_z, head)
    hook_name = tl_utils.get_act_name("z", layer)

    with torch.no_grad():
        logits_patched = model.run_with_hooks(
            tokens_conflict,
            fwd_hooks=[(hook_name, patch_hook)],
        )
    diff_patched = _ld(logits_patched, token_A, token_B, position)

    return {
        "diff_conflict": diff_orig,
        "diff_patched": diff_patched,
        "answer_flipped": (diff_orig > 0) != (diff_patched > 0),
    }


# ---------------------------------------------------------------------------
# Head activation magnitude (for identifying arbitration candidates)
# ---------------------------------------------------------------------------

def head_activation_magnitudes(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> np.ndarray:
    """
    Compute mean absolute activation magnitude of z for every head.

    Returns np.ndarray of shape (n_layers, n_heads).
    Mean is taken over the sequence dimension.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    mags = np.zeros((n_layers, n_heads))

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n.endswith("hook_z"),
        )

    for layer in range(n_layers):
        z = cache[tl_utils.get_act_name("z", layer)]  # (1, seq, n_heads, d_head)
        # Mean over batch and seq, L2 norm over d_head
        mag = z.squeeze(0).norm(dim=-1).mean(dim=0)   # (n_heads,)
        mags[layer] = mag.cpu().numpy()

    return mags


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _ld(logits: torch.Tensor, tok_A: int, tok_B: int, position: int = -1) -> float:
    """Logit diff helper (handles batched or unbatched tensors)."""
    if logits.dim() == 3:
        logits = logits[0]
    return (logits[position, tok_A] - logits[position, tok_B]).item()
