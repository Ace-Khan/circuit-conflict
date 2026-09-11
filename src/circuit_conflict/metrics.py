"""
metrics.py — head selection, paired statistics, and the Jaccard null models.

The headline number in this project is the cross-category overlap of causally
validated head sets.  Overlap between two k-subsets of 144 heads has a nonzero
chance baseline, so an observed Jaccard reported against ZERO is uninterpretable.
Two nulls are provided, because the project treats both high and low overlap as
informative and a single null can only bound one side:

  Null I  (chance floor)   uniform random k-subsets -> is the overlap above chance?
  Null II (shared ceiling) item-label permutation   -> is it below what one shared
                                                       mechanism would produce?

Null II is what makes a LOW overlap positive evidence for task-specific
arbitration, rather than a bare null result.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

HeadSet = set[tuple[int, int]]


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns a boolean mask of rejected nulls."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.zeros(p.shape, dtype=bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    n = order.size
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    if passed.any():
        cut = np.max(np.where(passed)[0])
        out[order[: cut + 1]] = True
    return out


# ---------------------------------------------------------------------------
# Paired per-head statistics
# ---------------------------------------------------------------------------

def paired_head_stats(effects: np.ndarray) -> pd.DataFrame:
    """
    Per-head statistics over items.

    The head column is named `head_idx`, not `head`: `head` shadows
    DataFrame.head/Series.head, so `df.head == 3` silently returns False rather
    than raising, which is a silent-wrong-answer bug rather than a loud one.

    effects : (n_items, n_layers, n_heads) normalised patching effects.

    A Wilcoxon signed-rank test against zero is used rather than the previous
    unpaired Welch's t-test: the design produces token-aligned pairs, and an
    unpaired test discards exactly the pairing the design exists to create.
    """
    _, n_layers, n_heads = effects.shape
    rows = []
    for L in range(n_layers):
        for h in range(n_heads):
            v = effects[:, L, h]
            v = v[~np.isnan(v)]
            if v.size < 3 or np.allclose(v, 0):
                p = np.nan
            else:
                try:
                    _, p = stats.wilcoxon(v)
                except ValueError:
                    p = np.nan
            rows.append({
                "layer": L, "head_idx": h,
                "mean_effect": float(np.mean(v)) if v.size else np.nan,
                "median_effect": float(np.median(v)) if v.size else np.nan,
                "se": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan,
                "n": int(v.size),
                "p_value": float(p) if not np.isnan(p) else np.nan,
            })
    df = pd.DataFrame(rows)
    df["fdr_significant"] = bh_fdr(df.p_value.values, q=0.05)
    return df


def select_head_set(
    stats_df: pd.DataFrame,
    effect_floor: float = 0.05,
    use_fdr: bool = True,
) -> HeadSet:
    """Causally validated head set: FDR-significant AND above an effect floor."""
    m = np.abs(stats_df.median_effect) >= effect_floor
    if use_fdr:
        m &= stats_df.fdr_significant
    return {(int(r.layer), int(r["head_idx"])) for _, r in stats_df[m].iterrows()}


def top_k_head_set(stats_df: pd.DataFrame, k: int = 10) -> HeadSet:
    """Fixed-k head set by |median effect| — for comparability with prior work."""
    d = stats_df.reindex(stats_df.median_effect.abs().sort_values(ascending=False).index)
    return {(int(r.layer), int(r["head_idx"])) for _, r in d.head(k).iterrows()}


# ---------------------------------------------------------------------------
# Overlap measures
# ---------------------------------------------------------------------------

def fast_top_k_selector(k: int = 10):
    """
    Top-k selector that skips the significance machinery.

    `top_k_head_set` ranks on |median_effect| alone, so the 144 Wilcoxon tests
    computed by `paired_head_stats` are pure overhead inside a resampling loop.
    Computing the median directly makes a 10,000-permutation null cheap enough to
    run, instead of ~30 minutes per category pair.  Exactly equivalent by
    construction — asserted against the slow path in the test below.
    """
    def sel(effects: np.ndarray) -> HeadSet:
        med = np.nanmedian(effects, axis=0)
        order = np.argsort(-np.abs(med), axis=None)[:k]
        ls, hs = np.unravel_index(order, med.shape)
        return {(int(l), int(h)) for l, h in zip(ls, hs)}
    return sel


def jaccard(a: HeadSet, b: HeadSet) -> float:
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def overlap_coefficient(a: HeadSet, b: HeadSet) -> float:
    """|A n B| / min(|A|,|B|) — robust when the two sets differ in size."""
    if not a or not b:
        return np.nan
    return len(a & b) / min(len(a), len(b))


# ---------------------------------------------------------------------------
# Null I — uniform random k-subsets (exact)
# ---------------------------------------------------------------------------

def jaccard_null_uniform(k_a: int, k_b: int, n_heads: int = 144) -> dict[str, float]:
    """
    Exact chance distribution of overlap between independent uniform subsets.

    |A n B| is hypergeometric, so no sampling is needed.  Returns the expected
    Jaccard and the smallest intersection size that clears p < 0.05.
    """
    if k_a == 0 or k_b == 0:
        return {"expected_intersection": np.nan, "expected_jaccard": np.nan,
                "min_intersection_p05": np.nan, "min_jaccard_p05": np.nan}
    rv = stats.hypergeom(n_heads, k_a, k_b)
    ms = np.arange(0, min(k_a, k_b) + 1)
    pmf = rv.pmf(ms)
    exp_m = float((ms * pmf).sum())
    surv = 1.0 - np.cumsum(pmf) + pmf          # P(M >= m)
    sig = ms[surv < 0.05]
    m05 = int(sig[0]) if sig.size else np.nan
    return {
        "expected_intersection": exp_m,
        "expected_jaccard": exp_m / (k_a + k_b - exp_m),
        "min_intersection_p05": m05,
        "min_jaccard_p05": (m05 / (k_a + k_b - m05)) if not np.isnan(float(m05)) else np.nan,
    }


def jaccard_p_upper(k_a: int, k_b: int, observed_intersection: int, n_heads: int = 144) -> float:
    """P(intersection >= observed) under Null I."""
    if k_a == 0 or k_b == 0:
        return np.nan
    return float(stats.hypergeom(n_heads, k_a, k_b).sf(observed_intersection - 1))


# ---------------------------------------------------------------------------
# Null II — item-label permutation (the "one shared mechanism" ceiling)
# ---------------------------------------------------------------------------

def jaccard_null_label_permutation(
    effects_by_cat: dict[str, np.ndarray],
    cat_a: str,
    cat_b: str,
    selector,
    n_perm: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """
    Distribution of Jaccard if both categories were draws from ONE mechanism.

    Pools items from the two categories and repeatedly re-splits them at the
    original sizes, re-selecting head sets each time.  Permuted pseudo-categories
    estimate the same mean, so this is the HIGH-overlap reference; an observed
    Jaccard significantly BELOW it is evidence for task-specific arbitration.

    Requires the two categories' effect arrays to share (n_layers, n_heads).
    """
    rng = np.random.default_rng(seed)
    ea, eb = effects_by_cat[cat_a], effects_by_cat[cat_b]
    pooled = np.concatenate([ea, eb], axis=0)
    n_a = ea.shape[0]
    n_total = pooled.shape[0]

    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(n_total)
        sa = selector(pooled[perm[:n_a]])
        sb = selector(pooled[perm[n_a:]])
        out[i] = jaccard(sa, sb)
    return out


def jaccard_bootstrap_ci(
    effects_by_cat: dict[str, np.ndarray],
    cat_a: str,
    cat_b: str,
    selector,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """
    Percentile bootstrap CI for the observed Jaccard.

    Necessary because top-k / FDR selection is a hard threshold and therefore
    unstable at n ~ 20 items per category.
    """
    rng = np.random.default_rng(seed)
    ea, eb = effects_by_cat[cat_a], effects_by_cat[cat_b]
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ia = rng.integers(0, ea.shape[0], ea.shape[0])
        ib = rng.integers(0, eb.shape[0], eb.shape[0])
        vals[i] = jaccard(selector(ea[ia]), selector(eb[ib]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi), vals


# ---------------------------------------------------------------------------
# Threshold-free backstop
# ---------------------------------------------------------------------------

def rank_correlation_across_categories(scores_by_cat: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Spearman rho between full 144-head effect vectors for every category pair.

    A conclusion that only holds at one choice of k is not a conclusion, so this
    threshold-free view is reported alongside the Jaccard.
    """
    cats = sorted(scores_by_cat)
    rows = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            va, vb = scores_by_cat[a].ravel(), scores_by_cat[b].ravel()
            ok = ~(np.isnan(va) | np.isnan(vb))
            rho, p = stats.spearmanr(va[ok], vb[ok])
            rows.append({"cat_a": a, "cat_b": b, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def overlap_vs_k(stats_by_cat: dict[str, pd.DataFrame], ks: Sequence[int] = (5, 10, 20, 30),
                 n_heads: int = 144) -> pd.DataFrame:
    """Jaccard at several k, so the result is not an artefact of one threshold."""
    cats = sorted(stats_by_cat)
    rows = []
    for k in ks:
        sets = {c: top_k_head_set(stats_by_cat[c], k) for c in cats}
        for i, a in enumerate(cats):
            for b in cats[i + 1:]:
                inter = len(sets[a] & sets[b])
                rows.append({
                    "k": k, "cat_a": a, "cat_b": b,
                    "intersection": inter,
                    "jaccard": jaccard(sets[a], sets[b]),
                    "expected_jaccard": jaccard_null_uniform(k, k, n_heads)["expected_jaccard"],
                    "p_upper": jaccard_p_upper(k, k, inter, n_heads),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase transition (logit-lens curves)
# ---------------------------------------------------------------------------

def detect_phase_transition(layer_diffs: np.ndarray) -> int | None:
    """
    First layer at which the logit-difference curve crosses zero, else None.

    The previous version fell back to argmax(|diff|) whenever there was no sign
    change, so a transition was reported for ANY non-flat curve and the reported
    detection rate was structurally 100%.  A curve that never crosses zero has no
    phase transition, and saying so is the informative answer.
    """
    d = np.asarray(layer_diffs, dtype=float)
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            return i + 1
    return None


def phase_transition_stats(transitions: Sequence[float | None]) -> dict[str, float]:
    """NaN-safe summary (values round-trip through CSV, where None becomes NaN)."""
    vals = [float(t) for t in transitions
            if t is not None and not (isinstance(t, float) and np.isnan(t))]
    n_total = len(transitions)
    if not vals:
        return {"n_total": n_total, "n_detected": 0, "detection_rate": 0.0,
                "mean": np.nan, "std": np.nan, "median": np.nan}
    a = np.array(vals, dtype=float)
    return {
        "n_total": n_total,
        "n_detected": int(a.size),
        "detection_rate": float(a.size / n_total) if n_total else 0.0,
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
    }


# ---------------------------------------------------------------------------
# Paper table
# ---------------------------------------------------------------------------

def compile_cross_category_table(
    head_sets: dict[str, HeadSet],
    effects_by_cat: dict[str, np.ndarray],
    selector,
    n_perm: int = 2000,
    n_boot: int = 1000,
    seed: int = 0,
    n_heads: int = 144,
) -> pd.DataFrame:
    """One row per category pair with observed overlap, both nulls, and a CI."""
    cats = sorted(head_sets)
    rows = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            sa, sb = head_sets[a], head_sets[b]
            inter = len(sa & sb)
            j = jaccard(sa, sb)
            null1 = jaccard_null_uniform(len(sa), len(sb), n_heads)
            perm = jaccard_null_label_permutation(
                effects_by_cat, a, b, selector, n_perm=n_perm, seed=seed)
            lo, hi, _ = jaccard_bootstrap_ci(
                effects_by_cat, a, b, selector, n_boot=n_boot, seed=seed)
            rows.append({
                "cat_a": a, "cat_b": b,
                "n_a": len(sa), "n_b": len(sb), "intersection": inter,
                "jaccard": j,
                "overlap_coefficient": overlap_coefficient(sa, sb),
                "ci_lo": lo, "ci_hi": hi,
                "expected_jaccard_chance": null1["expected_jaccard"],
                "p_upper_vs_chance": jaccard_p_upper(len(sa), len(sb), inter, n_heads),
                "perm_null_median": float(np.nanmedian(perm)),
                "p_lower_vs_shared": float(np.nanmean(perm <= j)),
            })
    return pd.DataFrame(rows)
