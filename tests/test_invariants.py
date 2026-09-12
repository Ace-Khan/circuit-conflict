"""
Invariant tests for the analysis pipeline.

These target the failure modes that actually bit this codebase: silently
misaligned minimal pairs, a pandas column name that shadows a DataFrame method,
and statistics that quietly return a wrong answer instead of raising.

Model-dependent tests are marked `slow` and skipped unless a model is available.
Run: uv run pytest -q          (fast only)
     uv run pytest -q -m slow  (includes model load)
"""

import numpy as np
import pandas as pd
import pytest

from circuit_conflict import metrics as M

# --------------------------------------------------------------------------
# Null models
# --------------------------------------------------------------------------

def test_hypergeometric_null_matches_simulation():
    """The exact chance floor must agree with brute-force sampling."""
    k, n = 10, 144
    rng = np.random.default_rng(0)
    sims = [len(set(rng.choice(n, k, replace=False)) & set(rng.choice(n, k, replace=False)))
            for _ in range(20000)]
    exact = M.jaccard_null_uniform(k, k, n)["expected_intersection"]
    assert abs(np.mean(sims) - exact) < 0.05


def test_chance_threshold_is_three_heads():
    """Documented in the README; if this changes the paper's threshold changes."""
    r = M.jaccard_null_uniform(10, 10, 144)
    assert r["min_intersection_p05"] == 3
    assert 0.17 < r["min_jaccard_p05"] < 0.18


def test_p_upper_is_monotone():
    from itertools import pairwise
    ps = [M.jaccard_p_upper(10, 10, m, 144) for m in range(1, 8)]
    assert all(a >= b for a, b in pairwise(ps))


def test_null_scales_with_head_count():
    """A bigger model has a lower chance floor."""
    small = M.jaccard_null_uniform(10, 10, 144)["expected_jaccard"]
    large = M.jaccard_null_uniform(10, 10, 384)["expected_jaccard"]
    assert large < small


# --------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------

def test_bh_fdr_rejects_only_small_p():
    mask = M.bh_fdr(np.array([0.001, 0.008, 0.2, 0.9]), q=0.05)
    assert mask.tolist() == [True, True, False, False]


def test_bh_fdr_handles_all_null():
    assert not M.bh_fdr(np.array([0.6, 0.7, 0.99]), q=0.05).any()


def test_bh_fdr_tolerates_nan():
    mask = M.bh_fdr(np.array([0.001, np.nan, 0.9]), q=0.05)
    assert mask[0] and not mask[2]


# --------------------------------------------------------------------------
# Phase transition — must return None rather than inventing one
# --------------------------------------------------------------------------

def test_no_crossing_returns_none():
    """Regression: this used to fall back to argmax|diff|, making detection 100%."""
    assert M.detect_phase_transition(np.array([1.0, 2.0, 5.0])) is None
    assert M.detect_phase_transition(np.array([-3.0, -2.0, -1.0])) is None


def test_crossing_is_found():
    assert M.detect_phase_transition(np.array([-1.0, -0.5, 0.5, 1.0])) == 2


def test_stats_are_nan_safe():
    """Values round-trip through CSV, where None becomes NaN."""
    s = M.phase_transition_stats([3, np.nan, 5, None])
    assert s["n_detected"] == 2
    assert s["mean"] == 4.0


# --------------------------------------------------------------------------
# Head selection
# --------------------------------------------------------------------------

def _fake_effects(n_items: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, (n_items, 12, 12))
    e[:, 10, 0] += 0.5          # one unmistakably strong head
    e[:, 11, 10] -= 0.4         # one strong in the other direction
    return e


def test_fast_selector_matches_slow_path():
    """fast_top_k_selector is used inside the resampling loops; it must be exact."""
    e = _fake_effects()
    assert M.fast_top_k_selector(10)(e) == M.top_k_head_set(M.paired_head_stats(e), 10)


def test_selection_finds_planted_heads():
    sel = M.fast_top_k_selector(5)(_fake_effects())
    assert (10, 0) in sel and (11, 10) in sel


def test_head_column_is_not_named_head():
    """
    `head` shadows DataFrame.head, so `df.head == 3` silently returns False
    instead of raising. Three bugs in this repo came from that.
    """
    cols = M.paired_head_stats(_fake_effects(5)).columns
    assert "head_idx" in cols and "head" not in cols


# --------------------------------------------------------------------------
# Overlap measures
# --------------------------------------------------------------------------

def test_jaccard_and_overlap_edges():
    a, b = {(1, 1), (2, 2)}, {(2, 2), (3, 3)}
    assert M.jaccard(a, a) == 1.0
    assert M.jaccard(a, {(9, 9)}) == 0.0
    assert M.jaccard(a, b) == pytest.approx(1 / 3)
    assert M.overlap_coefficient(a, b) == 0.5
    assert np.isnan(M.jaccard(set(), set()))


# --------------------------------------------------------------------------
# Dataset invariants (need a model)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformer_lens")
    from circuit_conflict.utils import load_model
    return load_model(verbose=False)


@pytest.mark.slow
@pytest.mark.parametrize("builder", ["a", "b", "c"])
def test_every_pair_is_token_aligned(model, builder):
    """
    The structural guarantee the whole design rests on: conflict and control
    tokenise to the same length and differ in exactly one token.
    """
    from circuit_conflict import dataset as D
    fn = {"a": D.build_category_a_df, "b": D.build_category_b_df,
          "c": D.build_category_c_df}[builder]
    df, _ = fn(model)
    assert len(df) > 0
    for item_id, g in df.groupby("item_id"):
        conf = g[g.arm == "conflict"].iloc[0]
        ctrl = g[g.arm == "control"].iloc[0]
        tc = model.to_tokens(conf.prompt_text)
        tu = model.to_tokens(ctrl.prompt_text)
        assert tc.shape == tu.shape, f"{item_id}: length mismatch"
        assert int((tc != tu).sum()) == 1, f"{item_id}: not a minimal pair"
        assert conf.token_A != conf.token_B
        assert conf.gold_control == "B"


@pytest.mark.slow
def test_misaligned_pair_is_rejected(model):
    """A bad pair must raise, never slip through."""
    from circuit_conflict.dataset import PairRejected, build_minimal_pair
    with pytest.raises(PairRejected):
        build_minimal_pair(model, "X_000", "A", "T", "cb",
                           "the cat sat on the mat today",
                           "the cat sat", "cat", "dog")


@pytest.mark.slow
def test_category_a_is_counterbalanced(model):
    """
    answer_A must be the first-mentioned name for half the items and the
    second-mentioned for the other half. Without this, answer identity is
    confounded with position and a recency head looks like an arbitration head.
    """
    from circuit_conflict import dataset as D
    df, _ = D.build_category_a_df(model, n_items=20)
    conf = df[df.arm == "conflict"]
    counts = conf.counterbalance.value_counts()
    assert set(counts.index) == {"slot_on_first", "slot_on_second"}
    assert abs(counts["slot_on_first"] - counts["slot_on_second"]) <= 1


@pytest.mark.slow
def test_category_b_crosses_template_with_counterbalance(model):
    """Deriving both from one counter made B1 always slot-on-second."""
    from circuit_conflict import dataset as D
    df, _ = D.build_category_b_df(model, n_items=24)
    conf = df[df.arm == "conflict"]
    ct = pd.crosstab(conf.template_id, conf.counterbalance)
    assert (ct.values > 0).all(), f"template and counterbalance not crossed:\n{ct}"


@pytest.mark.slow
def test_category_a_probes_are_unambiguous(model):
    """
    Each Category A probe must contain exactly ONE male name. A probe with both
    names male is ambiguous, so gating on it selects for the lexical prior --
    precisely what the gate exists to control for.
    """
    from circuit_conflict import dataset as D
    df, _ = D.build_category_a_df(model, n_items=12)
    males = set(D.MALE_NAMES)
    for _, r in df[df.arm == "conflict"].iterrows():
        for probe in (r.probe_A, r.probe_B):
            names = [w.strip(".,") for w in probe.split() if w.strip(".,") in males]
            assert len(names) == 1, f"probe has {len(names)} male names: {probe}"


def test_logit_lens_includes_unembed_bias():
    """
    Regression: the lens dropped b_U, shifting the whole curve by a constant and
    moving the zero crossing that detect_phase_transition keys on.
    """
    import inspect

    from circuit_conflict import utils
    src = inspect.getsource(utils.logit_lens_diff)
    assert "b_U" in src, "logit_lens_diff must include the unembedding bias"


def test_manipulation_check_lives_in_run_patching():
    """
    Regression: the check used to live in main(), so the notebook path (which
    calls run_patching directly) skipped it and overwrote the pipeline's
    filtered results with unfiltered ones. Two code paths, different science.
    """
    import inspect

    from circuit_conflict import pipeline as PL
    assert "MIN_SWING" in inspect.getsource(PL.run_patching)
    assert "MIN_SWING" not in inspect.getsource(PL.main)


def test_default_prompt_path_is_the_gpt2_file():
    """
    Regression: a separate prompts.csv let the notebook path and the pipeline
    path drift apart silently.
    """
    from circuit_conflict import dataset as D
    assert D.PROMPTS_CSV == D.prompts_path("gpt2")
    assert D.PROMPTS_CSV.name == "prompts_gpt2.csv"


def test_notebook_and_pipeline_use_the_same_resample_counts():
    """
    Regression: notebook 04 used n_perm=200 while the pipeline used 10,000, so
    running the notebooks overwrote the pipeline's results with lower-resolution
    p-values. Any divergence between the two paths is a correctness bug.
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    gen = (root / "scripts" / "make_notebooks.py").read_text()
    pipe = (root / "src" / "circuit_conflict" / "pipeline.py").read_text()

    def counts(text):
        return set(re.findall(r"n_(?:perm|boot)=(\d+)", text))

    assert counts(gen) == counts(pipe) == {"10000"}, (
        f"resample counts differ: notebooks {counts(gen)} vs pipeline {counts(pipe)}"
    )
