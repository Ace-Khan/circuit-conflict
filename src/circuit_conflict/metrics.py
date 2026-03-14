"""
metrics.py — statistics, Jaccard similarity, and phase-transition analysis.

All functions are pure (no model, no I/O side-effects) to keep them
independently testable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Phase-transition detection
# ---------------------------------------------------------------------------

def detect_phase_transition(layer_diffs: np.ndarray) -> Optional[int]:
    """
    Find the first layer where the logit-diff curve crosses zero, or — if
    no crossing — the layer with the largest magnitude change.

    Returns None only when the curve is entirely flat.
    """
    for i in range(len(layer_diffs) - 1):
        if layer_diffs[i] * layer_diffs[i + 1] < 0:
            return i + 1
    deltas = np.abs(np.diff(layer_diffs))
    if deltas.max() > 0:
        return int(np.argmax(deltas)) + 1
    return None


def phase_transition_stats(
    transitions: List[Optional[int]],
    n_layers: int = 12,
) -> Dict[str, float]:
    """
    Summarise a collection of phase-transition layer indices.

    Parameters
    ----------
    transitions : list of int or None (None = no transition detected)
    n_layers    : total number of model layers (for normalisation)

    Returns
    -------
    dict with keys: mean, std, median, min, max, n_detected, n_total,
                    detection_rate, normalised_mean
    """
    valid = [t for t in transitions if t is not None]
    n = len(valid)
    total = len(transitions)
    if n == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "median": float("nan"), "min": float("nan"), "max": float("nan"),
            "n_detected": 0, "n_total": total,
            "detection_rate": 0.0, "normalised_mean": float("nan"),
        }
    arr = np.array(valid, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_detected": n,
        "n_total": total,
        "detection_rate": n / total,
        "normalised_mean": float(arr.mean() / n_layers),
    }


# ---------------------------------------------------------------------------
# Arbitration head delta score
# ---------------------------------------------------------------------------

def arbitration_delta_score(
    conflict_magnitudes: np.ndarray,
    unambiguous_magnitudes: np.ndarray,
) -> np.ndarray:
    """
    Compute mean activation delta per head: conflict - unambiguous.

    Parameters
    ----------
    conflict_magnitudes    : (n_prompts_conflict, n_layers, n_heads)
    unambiguous_magnitudes : (n_prompts_unamb, n_layers, n_heads)

    Returns
    -------
    (n_layers, n_heads) delta array — positive = more active during conflict
    """
    return conflict_magnitudes.mean(axis=0) - unambiguous_magnitudes.mean(axis=0)


def rank_heads_by_delta(delta: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Return a sorted list of (layer, head, delta_score) from highest to lowest.
    """
    n_layers, n_heads = delta.shape
    flat = [(layer, head, float(delta[layer, head]))
            for layer in range(n_layers)
            for head in range(n_heads)]
    flat.sort(key=lambda x: x[2], reverse=True)
    return flat


def ranked_heads_to_df(ranked: List[Tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(ranked, columns=["layer", "head", "delta_score"])


# ---------------------------------------------------------------------------
# Cross-category Jaccard similarity
# ---------------------------------------------------------------------------

def top_k_heads(delta: np.ndarray, k: int = 10) -> Set[Tuple[int, int]]:
    """Return the top-k (layer, head) tuples by delta score."""
    ranked = rank_heads_by_delta(delta)
    return {(r[0], r[1]) for r in ranked[:k]}


def jaccard(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def cross_category_jaccard(
    deltas: Dict[str, np.ndarray],
    k: int = 10,
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity between top-k head sets across
    categories.

    Parameters
    ----------
    deltas : dict of {category_label: delta_array (n_layers, n_heads)}
    k      : number of top heads to compare

    Returns
    -------
    pd.DataFrame (n_categories × n_categories) of Jaccard scores
    """
    cats = list(deltas.keys())
    top = {c: top_k_heads(deltas[c], k) for c in cats}
    data = {c_row: {c_col: jaccard(top[c_row], top[c_col]) for c_col in cats}
            for c_row in cats}
    return pd.DataFrame(data, index=cats, columns=cats)


# ---------------------------------------------------------------------------
# Ablation / patch success statistics
# ---------------------------------------------------------------------------

def summarise_flip_rates(
    flip_data: List[Dict],  # list of dicts from patching.ablation_flip_rate
) -> pd.DataFrame:
    """
    Aggregate ablation flip-rate results across a list of runs into a tidy
    DataFrame suitable for plotting.
    """
    return pd.DataFrame(flip_data)


def t_test_conflict_vs_unamb(
    conflict_vals: np.ndarray,
    unamb_vals: np.ndarray,
) -> Dict[str, float]:
    """
    Welch's t-test: is the mean activation magnitude significantly higher
    during conflict prompts than during unambiguous prompts?

    Returns dict with t_stat, p_value, cohens_d.
    """
    t_stat, p_val = stats.ttest_ind(conflict_vals, unamb_vals, equal_var=False)
    pooled_std = np.sqrt(
        (conflict_vals.std(ddof=1) ** 2 + unamb_vals.std(ddof=1) ** 2) / 2
    )
    cohens_d = (conflict_vals.mean() - unamb_vals.mean()) / (pooled_std + 1e-12)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "mean_conflict": float(conflict_vals.mean()),
        "mean_unamb": float(unamb_vals.mean()),
    }


# ---------------------------------------------------------------------------
# Compile a full metric summary table
# ---------------------------------------------------------------------------

def compile_paper_metrics(
    transition_stats_by_cat: Dict[str, Dict],
    ranked_heads: List[Tuple[int, int, float]],
    flip_results_by_head: Dict[Tuple[int, int], Dict],
    jaccard_df: pd.DataFrame,
    top_k: int = 5,
) -> str:
    """
    Return a Markdown-formatted summary table of all paper metrics.
    """
    lines = ["# Paper Metrics Summary\n"]

    lines.append("## 1. Conflict Resolution Layer (Phase 2)\n")
    lines.append("| Category | Mean Layer | Std | Median | Detection Rate |")
    lines.append("|---|---|---|---|---|")
    for cat, st in transition_stats_by_cat.items():
        lines.append(
            f"| {cat} | {st['mean']:.2f} | {st['std']:.2f} | "
            f"{st['median']:.1f} | {st['detection_rate']:.0%} |"
        )

    lines.append("\n## 2. Top Arbitration Head Candidates (Phase 3)\n")
    lines.append("| Rank | Layer | Head | Delta Score |")
    lines.append("|---|---|---|---|")
    for rank, (layer, head, delta) in enumerate(ranked_heads[:10], 1):
        lines.append(f"| {rank} | {layer} | {head} | {delta:.4f} |")

    lines.append("\n## 3. Ablation Flip Rates — Top Candidates (Phase 3)\n")
    lines.append("| Layer | Head | Flip Rate | N Prompts |")
    lines.append("|---|---|---|---|")
    for (layer, head), res in list(flip_results_by_head.items())[:top_k]:
        lines.append(
            f"| {layer} | {head} | {res['flip_rate']:.1%} | {res['n_prompts']} |"
        )

    lines.append("\n## 4. Cross-Category Jaccard Similarity (Top-10 Heads)\n")
    lines.append(jaccard_df.to_markdown())

    return "\n".join(lines)
