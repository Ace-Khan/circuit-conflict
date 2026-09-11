"""
figures.py — publication figures.

Every figure is written as both PNG (300 dpi) and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circuit_conflict import dataset as D
from circuit_conflict import pipeline as PL

RESULTS = D.REPO_ROOT / "data" / "results"
FIGDIR = D.REPO_ROOT / "figures"
MODELS = ["gpt2", "pythia-410m"]
CATS = ["A", "B", "C"]
CATNAME = {"A": "A — coreference", "B": "B — instruction", "C": "C — factual override"}


def _res(model: str = "gpt2") -> Path:
    return PL.results_dir(model)


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.png|pdf")


def fig_head_maps(model: str = "gpt2") -> None:
    """Per-category causal effect map over every head."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    grids = {}
    for c in CATS:
        s = pd.read_csv(_res(model) / "phase3" / f"head_stats_{c}.csv")
        grids[c] = s.pivot(index="layer", columns="head_idx", values="median_effect").values
    vmax = max(np.nanmax(np.abs(g)) for g in grids.values())

    for ax, c in zip(axes, CATS):
        im = ax.imshow(grids[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(CATNAME[c], fontsize=11)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.set_xticks(range(0, 12, 2)); ax.set_yticks(range(0, 12, 2))
        fig.colorbar(im, ax=ax, fraction=0.046, label="median effect")
    fig.suptitle(f"Causal patching effect per head (control → conflict) — {model}",
                 fontsize=12.5, y=1.03)
    _save(fig, f"fig1_head_maps_{model}")


def fig_overlap(model: str = "gpt2") -> None:
    """Observed Jaccard against both nulls."""
    t = pd.read_csv(_res(model) / "phase4" / "cross_category_topk.csv")
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
    ax.set_title(f"Arbitration head overlap vs both nulls — {model}", fontsize=11.5)
    ax.set_ylim(0, max(t.perm_null_median.max(), t.ci_hi.max()) * 1.15)
    # legend below the axes: at upper-left it sat on top of the A-B ceiling line
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.3, zorder=0)
    _save(fig, f"fig2_cross_category_overlap_{model}")


def fig_overlap_vs_k(model: str = "gpt2") -> None:
    """Is the conclusion an artefact of one threshold?"""
    d = pd.read_csv(_res(model) / "phase4" / "overlap_vs_k.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for (a, b), g in d.groupby(["cat_a", "cat_b"]):
        g = g.sort_values("k")
        ax.plot(g.k, g.jaccard, "o-", lw=2, label=f"{a}–{b}")
    g0 = d.sort_values("k").drop_duplicates("k")
    ax.plot(g0.k, g0.expected_jaccard, "s--", color="#C0392B", lw=1.6, label="chance")
    ax.set_xlabel("k (heads per category)"); ax.set_ylabel("Jaccard overlap")
    ax.set_title(f"Overlap across selection thresholds — {model}", fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    _save(fig, f"fig3_overlap_vs_k_{model}")


def fig_logit_lens(model: str = "gpt2") -> None:
    """Mean logit-lens trajectory, conflict vs control, per category."""
    npz = np.load(_res(model) / "phase2" / "layer_curves.npz")
    summ = pd.read_csv(_res(model) / "phase2" / "phase2_summary.csv")
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
    fig.suptitle(f"Logit-lens trajectory to the decision — {model}", fontsize=12.5, y=1.02)
    _save(fig, f"fig4_logit_lens_{model}")


def fig_ablation(model: str = "gpt2") -> None:
    """Mean-ablation effect for each category's selected heads."""
    a = pd.read_csv(_res(model) / "phase3" / "ablation_results.csv")
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
    fig.suptitle(f"Mean-ablating each selected head on the conflict arm — {model}",
                 fontsize=12.5, y=1.03)
    _save(fig, f"fig5_ablation_{model}")


def fig_model_comparison() -> None:
    """
    The result across both models — the figure that decides the paper's claim.

    The "below a shared mechanism" half replicates in both models. The "above
    chance" half does not: it is weaker throughout in Pythia-410M and absent for
    A-B.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        t = pd.read_csv(_res(model) / "phase4" / "cross_category_topk.csv")
        x = np.arange(len(t))
        sig = t.p_upper_vs_chance < 0.05
        ax.bar(x, t.jaccard, .5, color=["#3B6EA5" if s else "#AAB7C4" for s in sig], zorder=3)
        ax.errorbar(x, t.jaccard, yerr=[t.jaccard - t.ci_lo, t.ci_hi - t.jaccard],
                    fmt="none", ecolor="black", capsize=5, lw=1.3, zorder=4)
        for i, r in t.iterrows():
            ax.hlines(r.expected_jaccard_chance, i - .3, i + .3, color="#C0392B", lw=2, zorder=5)
            ax.hlines(r.perm_null_median, i - .3, i + .3, color="#27AE60", lw=2, ls="--", zorder=5)
            ax.text(i, r.ci_hi + .02,
                    f"{r.intersection}/10\n" + (f"p={r.p_upper_vs_chance:.3f}" if r.p_upper_vs_chance >= .001 else "p<0.001")
                    + ("" if r.p_upper_vs_chance < .05 else "\n(n.s.)"),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{r.cat_a}–{r.cat_b}" for _, r in t.iterrows()])
        n_heads = 144 if model == "gpt2" else 384
        ax.set_title(f"{model}  ({n_heads} heads)", fontsize=11.5)
        ax.grid(axis="y", alpha=.3, zorder=0)
    axes[0].set_ylabel("Jaccard overlap of top-10 head sets")
    axes[0].set_ylim(0, 0.82)
    # proxy artists: an empty bar() does not carry its colour into the legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#3B6EA5", label="observed, above chance"),
        Patch(facecolor="#AAB7C4", label="observed, not above chance"),
        Line2D([], [], color="#C0392B", lw=2, label="chance floor"),
        Line2D([], [], color="#27AE60", lw=2, ls="--", label="one-shared-mechanism ceiling"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .04),
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Overlap is below a shared mechanism in both models; above chance only in GPT-2 Small",
                 fontsize=12.5, y=1.02)
    _save(fig, "fig0_model_comparison")


def main() -> None:
    print("Writing figures:")
    fig_model_comparison()
    for model in MODELS:
        fig_head_maps(model)
        fig_overlap(model)
        fig_overlap_vs_k(model)
        fig_logit_lens(model)
        fig_ablation(model)


if __name__ == "__main__":
    main()
