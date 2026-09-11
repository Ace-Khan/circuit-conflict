"""
patching.py — activation patching, direct logit attribution, and ablation.

What changed and why
--------------------
The previous module measured head involvement with the mean L2 norm of `z`
averaged over the WHOLE sequence, differenced between conflict and control.
That statistic is unusable here for two independent reasons: it is confounded
with sequence length and content (the arms used to differ in both), and it is
UNSIGNED, so it cannot distinguish a head pushing toward answer A from one
pushing toward answer B.  It has been removed rather than repaired.

It is replaced by three position-specific measurements:

  1. `head_dla`          direct logit attribution at the decision site
  2. `head_attn_to`      attention from the decision site to the competing sources
  3. `patch_heads`       position-resolved activation patching (the CAUSAL one)

Head selection is driven by (3).  (1) and (2) are descriptive.

Why the decision site
---------------------
The reported logit difference is a function of `resid_post[p_end]` and nothing
else.  A head's contribution to the DECISION is therefore exactly its write into
the residual stream at `p_end`.  Activity at any other position can only matter
by being read by a later head at `p_end`, and position-resolved patching
measures that path separately.  So (1) + (3) are jointly complete, which
whole-sequence norm was neither.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer, utils as tl_utils

from circuit_conflict.utils import answer_direction, logit_diff_batched


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def _make_patch_head_hook(src_z: torch.Tensor, head: int, positions: Optional[Sequence[int]]):
    """
    Write `src_z`'s head activations into the running model.

    The shape assertion is deliberate: patching between different-length runs is
    the bug this whole redesign exists to make impossible, so it fails loudly
    rather than broadcasting something meaningless.
    """
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        if src_z.shape[1] != value.shape[1]:
            raise ValueError(
                f"sequence-length mismatch in patch hook at {hook.name}: "
                f"source {src_z.shape[1]} vs destination {value.shape[1]}. "
                "Minimal pairs must be token-aligned."
            )
        out = value.clone()
        if positions is None:
            out[:, :, head, :] = src_z[:, :, head, :]
        else:
            for p in positions:
                out[:, p, head, :] = src_z[:, p, head, :]
        return out
    return hook_fn


def _make_mean_ablate_hook(mean_z: torch.Tensor, head: int, positions: Optional[Sequence[int]]):
    """
    Replace a head's output with its mean over the control distribution.

    Zero-ablation takes the model off-distribution -- a zeroed head is not a
    state the model ever occupies -- which conflates removing the head's
    function with feeding the residual stream something corrupt.  Mean-ablation
    is the Wang et al. (2022) protocol.

    mean_z : (seq, n_heads, d_head)
    """
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        out = value.clone()
        if positions is None:
            out[:, :, head, :] = mean_z[:, head, :].unsqueeze(0)
        else:
            for p in positions:
                out[:, p, head, :] = mean_z[p, head, :]
        return out
    return hook_fn


# ---------------------------------------------------------------------------
# Batched forward helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_z(model: HookedTransformer, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Run the model, returning (logits, {layer: z}) for all layers."""
    names = {tl_utils.get_act_name("z", L): L for L in range(model.cfg.n_layers)}
    logits, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n in names
    )
    return logits, {L: cache[name] for name, L in names.items()}


@torch.no_grad()
def mean_z_over_prompts(model: HookedTransformer, tokens: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Mean `z` over a batch of equal-length prompts, per layer.

    Returns {layer: (seq, n_heads, d_head)} — the ablation distribution.
    """
    _, zs = cache_z(model, tokens)
    return {L: z.mean(dim=0) for L, z in zs.items()}


# ---------------------------------------------------------------------------
# 1. Direct logit attribution at the decision site
# ---------------------------------------------------------------------------

@torch.no_grad()
def head_dla(
    model: HookedTransformer,
    tokens: torch.Tensor,
    token_A: torch.Tensor,
    token_B: torch.Tensor,
    position: int = -1,
) -> np.ndarray:
    """
    Per-head contribution to (logit_A - logit_B), read out at `position`.

    tokens    : (batch, seq)
    token_A/B : (batch,) int64
    Returns   : (batch, n_layers, n_heads)

    Computed as z @ W_O projected onto the answer direction, after the folded
    final layer norm.  `z @ W_O` is formed explicitly rather than via
    `use_attn_result=True`, which would materialise a (batch, seq, n_heads,
    d_model) tensor for every layer.
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    names = [tl_utils.get_act_name("z", L) for L in range(n_layers)] + ["ln_final.hook_scale"]
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)

    # (batch, d_model) readout direction, one per item
    dirs = torch.stack([answer_direction(model, int(a), int(b))
                        for a, b in zip(token_A, token_B)])          # (batch, d_model)
    scale = cache["ln_final.hook_scale"][:, position]                # (batch, 1)

    out = np.zeros((tokens.shape[0], n_layers, n_heads), dtype=np.float64)
    for L in range(n_layers):
        z = cache[tl_utils.get_act_name("z", L)][:, position]        # (batch, n_heads, d_head)
        # (batch, n_heads, d_model)
        res = torch.einsum("bhd,hdm->bhm", z, model.W_O[L])
        res = res - res.mean(dim=-1, keepdim=True)                   # LN centring
        res = res / scale.unsqueeze(-1)                              # LN scaling
        out[:, L, :] = torch.einsum("bhm,bm->bh", res, dirs).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# 2. Attention from the decision site to the competing sources
# ---------------------------------------------------------------------------

@torch.no_grad()
def head_attn_to(
    model: HookedTransformer,
    tokens: torch.Tensor,
    from_position: int,
    to_positions: Sequence[int],
) -> np.ndarray:
    """
    Attention probability from `from_position` to each of `to_positions`.

    Returns (batch, n_layers, n_heads, len(to_positions)).  Length-invariant and
    directly interpretable as "is this head looking at the competing evidence?".
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    names = [tl_utils.get_act_name("pattern", L) for L in range(n_layers)]
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)

    out = np.zeros((tokens.shape[0], n_layers, n_heads, len(to_positions)), dtype=np.float64)
    for L in range(n_layers):
        patt = cache[tl_utils.get_act_name("pattern", L)]            # (b, h, q, k)
        for j, p in enumerate(to_positions):
            if p >= 0:
                out[:, L, :, j] = patt[:, :, from_position, p].cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# 3. Position-resolved activation patching — the causal measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def patch_heads(
    model: HookedTransformer,
    tokens_dst: torch.Tensor,
    tokens_src: torch.Tensor,
    token_A: torch.Tensor,
    token_B: torch.Tensor,
    position_readout: int = -1,
    patch_positions: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Patch every head from the source run into the destination run.

    Standard direction here: dst = conflict, src = control, so the effect asks
    "how much of the conflict is undone by restoring this head to its
    no-conflict value?".

    Returns
    -------
    {
      "d_dst":   (batch,)                    logit diff, unpatched destination
      "d_src":   (batch,)                    logit diff, source
      "patched": (batch, n_layers, n_heads)  logit diff after patching
      "effect":  (batch, n_layers, n_heads)  normalised (see below)
    }

    effect = (d_patch - d_dst) / (d_src - d_dst)
      0 -> head is irrelevant
      1 -> this head alone carries the entire conflict effect
     <0 -> head opposes

    Normalising per item is required: averaging raw logit deltas across items
    re-introduces an item-magnitude confound, which is the thing this redesign
    is trying to eliminate.
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    if tokens_dst.shape != tokens_src.shape:
        raise ValueError(
            f"token shape mismatch: dst {tuple(tokens_dst.shape)} vs src {tuple(tokens_src.shape)}"
        )

    logits_dst, _ = cache_z(model, tokens_dst)
    logits_src, z_src = cache_z(model, tokens_src)
    d_dst = logit_diff_batched(logits_dst, token_A, token_B, position_readout)
    d_src = logit_diff_batched(logits_src, token_A, token_B, position_readout)

    patched = np.zeros((tokens_dst.shape[0], n_layers, n_heads), dtype=np.float64)
    for L in range(n_layers):
        name = tl_utils.get_act_name("z", L)
        for h in range(n_heads):
            hook = _make_patch_head_hook(z_src[L], h, patch_positions)
            logits_p = model.run_with_hooks(tokens_dst, fwd_hooks=[(name, hook)])
            patched[:, L, h] = logit_diff_batched(
                logits_p, token_A, token_B, position_readout
            ).cpu().numpy()
        if verbose:
            print(f"    layer {L} done")

    d_dst_np = d_dst.cpu().numpy()
    denom = (d_src - d_dst).cpu().numpy()
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
    effect = (patched - d_dst_np[:, None, None]) / denom[:, None, None]

    return {"d_dst": d_dst_np, "d_src": d_src.cpu().numpy(),
            "patched": patched, "effect": effect}


# ---------------------------------------------------------------------------
# 4. Mean-ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def ablate_heads(
    model: HookedTransformer,
    tokens: torch.Tensor,
    token_A: torch.Tensor,
    token_B: torch.Tensor,
    mean_z: Dict[int, torch.Tensor],
    heads: Sequence[Tuple[int, int]],
    position_readout: int = -1,
    ablate_positions: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Mean-ablate each of `heads` in turn; report the change in logit difference.

    Returns {"baseline": (batch,), "ablated": (batch, len(heads)),
             "delta": (batch, len(heads))}.
    """
    logits = model(tokens)
    baseline = logit_diff_batched(logits, token_A, token_B, position_readout).cpu().numpy()

    ablated = np.zeros((tokens.shape[0], len(heads)), dtype=np.float64)
    for j, (L, h) in enumerate(heads):
        hook = _make_mean_ablate_hook(mean_z[L], h, ablate_positions)
        logits_a = model.run_with_hooks(
            tokens, fwd_hooks=[(tl_utils.get_act_name("z", L), hook)]
        )
        ablated[:, j] = logit_diff_batched(
            logits_a, token_A, token_B, position_readout
        ).cpu().numpy()

    return {"baseline": baseline, "ablated": ablated,
            "delta": ablated - baseline[:, None]}


@torch.no_grad()
def ablate_head_layer_curve(
    model: HookedTransformer,
    tokens: torch.Tensor,
    token_A: int,
    token_B: int,
    layer_abl: int,
    head_abl: int,
    mean_z: Dict[int, torch.Tensor],
    position: int = -1,
    ablate_positions: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Logit-lens curve with one head mean-ablated.  Returns (n_layers,).

    Note: `run_with_cache(..., fwd_hooks=...)` is NOT valid — run_with_cache
    forwards unknown kwargs to `forward()`, which takes no `fwd_hooks`, so the
    previous implementation raised TypeError on first call.  Hooks must be
    installed via the `model.hooks(...)` context manager instead.
    """
    from circuit_conflict.utils import logit_lens_diff

    hook = _make_mean_ablate_hook(mean_z[layer_abl], head_abl, ablate_positions)
    with model.hooks(fwd_hooks=[(tl_utils.get_act_name("z", layer_abl), hook)]):
        _, cache = model.run_with_cache(tokens)
    return logit_lens_diff(model, cache, token_A, token_B, position)
