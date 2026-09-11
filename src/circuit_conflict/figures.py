"""
figures.py — publication figures.

Every figure is written as both PNG (300 dpi) and PDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circuit_conflict import dataset as D
from circuit_conflict import metrics as M

RESULTS = D.REPO_ROOT / "data" / "results"
FIGDIR = D.REPO_ROOT / "figures"
CATS = ["A", "B", "C"]
CATNAME = {"A": "A — coreference", "B": "B — instruction", "C": "C — factual override"}


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.png|pdf")


def fig_head_maps() -> None:
    """Per-category causal effect map over all 144 heads."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    grids = {}
    for c in CATS:
        s = pd.read_csv(RESULTS / "phase3" / f"head_stats_{c}.csv")
        grids[c] = s.pivot(index="layer", columns="head_idx", values="median_effect").values
    vmax = max(np.nanmax(np.abs(g)) for g in grids.values())

    for ax, c in zip(axes, CATS):
        im = ax.imshow(grids[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(CATNAME[c], fontsize=11)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.set_xticks(range(0, 12, 2)); ax.set_yticks(range(0, 12, 2))
        fig.colorbar(im, ax=ax, fraction=0.046, label="median effect")
    fig.suptitle("Causal patching effect per head (control → conflict), by conflict family",
                 fontsize=12.5, y=1.03)
    _save(fig, "fig1_head_maps")


def fig_overlap() -> None:
    """Observed Jaccard against both nulls — the headline figure."""
    t = pd.read_csv(RESULTS / "phase4" / "cross_category_topk.csv")
    labels = [f"{r.cat_a}–{r.cat_b}" for _, r in t.iterrows()]
    x = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x, t.jaccard, 0.5, color="#3B6EA5", label="observed", zorder=3)
    ax.errorbar(x, t.jaccard,
                yerr=[t.jaccard - t.ci_lo, t.ci_hi - t.jaccard],
                fmt="none", ecolor="black", capsize=5, lw=1.4, zorder=4)

    for i, r in t.iterrows():
        ax.hlines(r.expected_jaccard_chance, i - .3, i + .3,
                  color="#C0392B", lw=2, zorder=5)
        ax.hlines(r.perm_null_median, i - .3, i + .3,
                  color="#27AE60", lw=2, ls="--", zorder=5)
        # anchor labels above the CI whisker, not the bar, so they never collide
        ax.text(i, r.ci_hi + .022, f"{r.intersection}/10 heads\np={r.p_upper_vs_chance:.3g}",
                ha="center", va="bottom", fontsize=8.5)

    ax.hlines([], [], [], color="#C0392B", lw=2, label="chance floor (hypergeometric)")
    ax.hlines([], [], [], color="#27AE60", lw=2, ls="--", label="one-shared-mechanism ceiling (permutation)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Jaccard overlap of top-10 head sets")
    ax.set_title("Arbitration head overlap sits above chance but below a shared mechanism",
                 fontsize=11.5)
    ax.set_ylim(0, max(t.perm_null_median.max(), t.ci_hi.max()) * 1.15)
    # legend below the axes: at upper-left it sat on top of the A-B ceiling line
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.3, zorder=0)
    _save(fig, "fig2_cross_category_overlap")


def fig_overlap_vs_k() -> None:
    """Is the conclusion an artefact of one threshold?"""
    d = pd.read_csv(RESULTS / "phase4" / "overlap_vs_k.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for (a, b), g in d.groupby(["cat_a", "cat_b"]):
        g = g.sort_values("k")
        ax.plot(g.k, g.jaccard, "o-", lw=2, label=f"{a}–{b}")
    g0 = d.sort_values("k").drop_duplicates("k")
    ax.plot(g0.k, g0.expected_jaccard, "s--", color="#C0392B", lw=1.6, label="chance")
    ax.set_xlabel("k (heads per category)"); ax.set_ylabel("Jaccard overlap")
    ax.set_title("Overlap is stable across selection thresholds", fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    _save(fig, "fig3_overlap_vs_k")


def fig_logit_lens() -> None:
    """Mean logit-lens trajectory, conflict vs control, per category."""
    npz = np.load(RESULTS / "phase2" / "layer_curves.npz")
    summ = pd.read_csv(RESULTS / "phase2" / "phase2_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, c in zip(axes, CATS):
        for arm, col in (("conflict", "#C0392B"), ("control", "#3B6EA5")):
            ids = summ[(summ.category == c) & (summ.arm == arm)].item_id
            curves = np.stack([npz[f"{i}_{arm}"] for i in ids if f"{i}_{arm}" in npz])
            mean, se = curves.mean(0), curves.std(0, ddof=1) / np.sqrt(len(curves))
            xs = np.arange(len(mean))
            ax.plot(xs, mean, color=col, lw=2, label=f"{arm} (n={len(curves)})")
            ax.fill_between(xs, mean - se, mean + se, color=col, alpha=.22)
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.set_title(CATNAME[c], fontsize=11)
        ax.set_xlabel("layer"); ax.legend(fontsize=8.5); ax.grid(alpha=.3)
    axes[0].set_ylabel("logit diff  (A − B)")
    fig.suptitle("Logit-lens trajectory to the decision", fontsize=12.5, y=1.02)
    _save(fig, "fig4_logit_lens")


def fig_ablation() -> None:
    """Mean-ablation effect for each category's selected heads."""
    a = pd.read_csv(RESULTS / "phase3" / "ablation_results.csv")
    g = (a.groupby(["category", "layer", "head_idx"], as_index=False)
           .mean_delta.mean())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, c in zip(axes, CATS):
        s = g[g.category == c].copy()
        s = s.reindex(s.mean_delta.abs().sort_values(ascending=False).index)
        lab = [f"L{int(r['layer'])}H{int(r['head_idx'])}" for _, r in s.iterrows()]
        col = ["#C0392B" if v < 0 else "#3B6EA5" for v in s.mean_delta]
        ax.barh(range(len(s)), s.mean_delta, color=col)
        ax.set_yticks(range(len(s))); ax.set_yticklabels(lab, fontsize=8.5)
        ax.invert_yaxis(); ax.axvline(0, color="k", lw=.8)
        ax.set_title(CATNAME[c], fontsize=11); ax.set_xlabel("Δ logit diff when mean-ablated")
        ax.grid(axis="x", alpha=.3)
    fig.suptitle("Causal confirmation: mean-ablating each selected head on the conflict arm",
                 fontsize=12.5, y=1.03)
    _save(fig, "fig5_ablation")


def main() -> None:
    print("Writing figures:")
    fig_head_maps()
    fig_overlap()
    fig_overlap_vs_k()
    fig_logit_lens()
    fig_ablation()


if __name__ == "__main__":
    main()
