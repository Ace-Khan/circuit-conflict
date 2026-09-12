"""
pipeline.py — the full analysis, end to end.

Grouping note: items are processed per (category, template_id).  Every item
built from one template has the same token length and the same position indices
(all fillers are single tokens by construction), so a group can be batched into
a single forward pass.  This is what makes the 144-head sweep cheap: the old
one-prompt-at-a-time design needed ~7,500 sequential forwards for Phase 1 alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from circuit_conflict import dataset as D
from circuit_conflict import metrics as M
from circuit_conflict import patching as P
from circuit_conflict.utils import load_model, logit_lens_diff

RESULTS_ROOT = D.REPO_ROOT / "data" / "results"
RESULTS = RESULTS_ROOT          # default (gpt2); rebound per-model by main()


def results_dir(model_name: str = "gpt2") -> Path:
    """Per-model results tree. gpt2 keeps the flat layout it was first written to."""
    return RESULTS_ROOT if model_name == "gpt2" else RESULTS_ROOT / D.model_slug(model_name)

# Items are batched only when they agree on template, length AND slot position.
# Grouping on template alone is not sufficient: a template with a multi-token
# filler produces items of differing length, which cannot share a forward pass.
BATCH_KEY = ["category", "template_id", "n_tokens", "p_slot"]
CATEGORIES = ["A", "B", "C"]

# Manipulation check threshold, in logits. Items whose control-minus-conflict
# swing is smaller than this are dropped before analysis.
MIN_SWING = 1.0


def _group_tensors(model, grp: pd.DataFrame):
    """Return (tokens_conflict, tokens_control, tok_A, tok_B, positions) for a group."""
    conf = grp[grp.arm == "conflict"].sort_values("item_id")
    ctrl = grp[grp.arm == "control"].sort_values("item_id")
    assert list(conf.item_id) == list(ctrl.item_id), "conflict/control items misaligned"
    t_conf = torch.cat([model.to_tokens(t) for t in conf.prompt_text])
    t_ctrl = torch.cat([model.to_tokens(t) for t in ctrl.prompt_text])
    tA = torch.tensor(conf.token_A.values, device=model.cfg.device)
    tB = torch.tensor(conf.token_B.values, device=model.cfg.device)
    pos = {
        "p_end": int(conf.p_end.iloc[0]),
        "p_slot": int(conf.p_slot.iloc[0]),
    }
    # positions must be constant within a group for batching to be valid
    assert conf.p_end.nunique() == 1 and conf.p_slot.nunique() == 1, \
        "positions vary within a template group; batching would be invalid"
    return conf, ctrl, t_conf, t_ctrl, tA, tB, pos


def run_patching(model, df: pd.DataFrame, verbose: bool = True):
    """
    Phase 1+3 core, per category:

      - position-resolved patching  (causal; drives head selection)
      - direct logit attribution    (descriptive, paired conflict - control)
      - attention to the slot       (descriptive)

    Returns (effects, behavioural, descriptive) where `effects` and each entry of
    `descriptive` are {category: (n_items, n_layers, n_heads)}.
    """
    effects: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
    dla: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
    attn: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
    behav_rows = []

    for (cat, tid, _nt, _ps), grp in df.groupby(BATCH_KEY):
        conf, _ctrl, t_conf, t_ctrl, tA, tB, pos = _group_tensors(model, grp)
        if verbose:
            print(f"  {cat}/{tid}: {len(conf)} items, seq={t_conf.shape[1]}")

        r = P.patch_heads(model, t_conf, t_ctrl, tA, tB,
                          position_readout=pos["p_end"])
        effects[cat].append(r["effect"])

        # paired within-item DLA difference: both arms are token-aligned, so
        # p_end is the same index and length cannot enter the comparison
        d_conf = P.head_dla(model, t_conf, tA, tB, position=pos["p_end"])
        d_ctrl = P.head_dla(model, t_ctrl, tA, tB, position=pos["p_end"])
        dla[cat].append(d_conf - d_ctrl)

        # attention from the decision site to the swapped token
        attn[cat].append(
            P.head_attn_to(model, t_conf, pos["p_end"], [pos["p_slot"]])[..., 0])

        for i, item in enumerate(conf.item_id):
            behav_rows.append({
                "item_id": item, "category": cat, "template_id": tid,
                "d_conflict": r["d_dst"][i], "d_control": r["d_src"][i],
                "swing": r["d_src"][i] - r["d_dst"][i],
            })

    def join(d):
        return {c: np.concatenate(v, axis=0) for c, v in d.items() if v}

    eff, dl, at = join(effects), join(dla), join(attn)
    behav = pd.DataFrame(behav_rows)

    # Manipulation check lives HERE, not in main(), so every caller gets it.
    # An item whose slot swap barely moves behaviour has no arbitration to
    # explain, and its tiny denominator inflates every normalised effect
    # computed from it. Pre-registered threshold, sign-symmetric.
    weak = behav[behav.swing.abs() < MIN_SWING]
    if len(weak):
        if verbose:
            print(f"  manipulation check: dropping {len(weak)} item(s) with "
                  f"|swing| < {MIN_SWING}: {', '.join(weak.item_id)}")
        keep = set(behav.loc[behav.swing.abs() >= MIN_SWING, "item_id"])
        for c in list(eff):
            ids = list(behav.loc[behav.category == c, "item_id"])
            mask = np.array([i in keep for i in ids])
            assert len(ids) == eff[c].shape[0], "behavioural rows misaligned with effects"
            eff[c], dl[c], at[c] = eff[c][mask], dl[c][mask], at[c][mask]
        behav = behav[behav.swing.abs() >= MIN_SWING].reset_index(drop=True)

    return eff, behav, {"dla": dl, "attn": at}


def run_logit_lens(model, df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: per-item logit-lens curve and phase-transition layer."""
    rows = []
    curves = {}
    for (cat, tid, _nt, _ps), grp in df.groupby(BATCH_KEY):
        conf, ctrl, t_conf, t_ctrl, tA, tB, pos = _group_tensors(model, grp)
        for arm, toks, meta in (("conflict", t_conf, conf), ("control", t_ctrl, ctrl)):
            for i, item in enumerate(meta.item_id):
                with torch.no_grad():
                    _, cache = model.run_with_cache(toks[i : i + 1])
                curve = logit_lens_diff(model, cache, int(tA[i]), int(tB[i]),
                                        position=pos["p_end"])
                pt = M.detect_phase_transition(curve)
                curves[f"{item}_{arm}"] = curve
                rows.append({
                    "item_id": item, "category": cat, "arm": arm,
                    "phase_transition_layer": pt,
                    "final_diff": curve[-1],
                })
    np.savez_compressed(RESULTS / "phase2" / "layer_curves.npz", **curves)
    return pd.DataFrame(rows)


def run_ablation(model, df: pd.DataFrame, head_sets: dict[str, M.HeadSet]) -> pd.DataFrame:
    """
    Causal confirmation: mean-ablate each selected head on the conflict arm.

    Mean-ablation uses the control arm as the reference distribution, so the
    counterfactual is "this head behaves as it does when there is no conflict"
    rather than "this head outputs zero", which is off-distribution.
    """
    rows = []
    for (cat, tid, _nt, _ps), grp in df.groupby(BATCH_KEY):
        heads = sorted(head_sets.get(cat, set()))
        if not heads:
            continue
        conf, _ctrl, t_conf, t_ctrl, tA, tB, pos = _group_tensors(model, grp)
        mean_z = P.mean_z_over_prompts(model, t_ctrl)
        r = P.ablate_heads(model, t_conf, tA, tB, mean_z, heads,
                           position_readout=pos["p_end"])
        for j, (L, h) in enumerate(heads):
            rows.append({
                "category": cat, "template_id": tid, "layer": L, "head_idx": h,
                "mean_baseline": r["baseline"].mean(),
                "mean_ablated": r["ablated"][:, j].mean(),
                "mean_delta": r["delta"][:, j].mean(),
                "n_items": len(conf),
            })
    return pd.DataFrame(rows)


def build_dataset(model, model_name: str) -> pd.DataFrame:
    """Generate and gate the dataset for one model (never reuse another model's)."""
    frames, rejects = [], []
    for fn in (D.build_category_a_df, D.build_category_b_df, D.build_category_c_df):
        f, r = fn(model); frames.append(f); rejects += r
    df = pd.concat(frames, ignore_index=True)
    print(f"  built {len(df)//2} items, {len(rejects)} rejected")
    df = D.run_preconditions(model, df)
    D.save_prompts(df, D.prompts_path(model_name))
    return df


def main(model_name: str = "gpt2", effect_floor: float = 0.05,
         top_k: int = 10, seed: int = 0, rebuild: bool = False) -> None:
    global RESULTS
    RESULTS = results_dir(model_name)
    for sub in ("phase1", "phase2", "phase3", "phase4"):
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)

    model = load_model(model_name)
    n_heads_total = model.cfg.n_layers * model.cfg.n_heads
    print(f"  {model.cfg.n_layers} layers x {model.cfg.n_heads} heads = {n_heads_total} heads")

    path = D.prompts_path(model_name)
    if rebuild or not path.exists():
        print("\n[0/5] Building dataset for this model")
        df = build_dataset(model, model_name)
    else:
        df = D.load_prompts(path)
    admitted = df[df.passes_precondition].copy()
    print(f"\nAdmitted {len(admitted)//2} items of {len(df)//2} generated")
    print(D.precondition_report(df).to_string(index=False))

    print("\n[1/5] Patching, direct logit attribution, attention")
    effects, behav, desc = run_patching(model, admitted)
    behav.to_csv(RESULTS / "phase1" / "behavioural.csv", index=False)
    for c, e in effects.items():
        np.save(RESULTS / "phase1" / f"effects_{c}.npy", e)
        np.save(RESULTS / "phase1" / f"dla_{c}.npy", desc["dla"][c])
        np.save(RESULTS / "phase1" / f"attn_slot_{c}.npy", desc["attn"][c])

    print("\n[2/5] Logit lens")
    lens = run_logit_lens(model, admitted)
    lens.to_csv(RESULTS / "phase2" / "phase2_summary.csv", index=False)
    pt_stats = {c: M.phase_transition_stats(
                    lens[(lens.category == c) & (lens.arm == "conflict")]
                        .phase_transition_layer.tolist())
                for c in CATEGORIES}
    (RESULTS / "phase2" / "phase_transition_stats.json").write_text(json.dumps(pt_stats, indent=2))

    print("\n[3/5] Per-head paired statistics and head selection")
    stats_by_cat, head_sets, topk_sets = {}, {}, {}
    for c in CATEGORIES:
        if c not in effects:
            continue
        s = M.paired_head_stats(effects[c])
        s.insert(0, "category", c)
        s.to_csv(RESULTS / "phase3" / f"head_stats_{c}.csv", index=False)
        stats_by_cat[c] = s
        head_sets[c] = M.select_head_set(s, effect_floor=effect_floor)
        topk_sets[c] = M.top_k_head_set(s, k=top_k)
        print(f"  {c}: {len(head_sets[c])} FDR-significant heads, "
              f"top-{top_k} set size {len(topk_sets[c])}")

    # descriptive layer: does the causally-selected head also write the answer
    # direction at the decision site, and attend to the swapped token?
    desc_rows = []
    for c in stats_by_cat:
        dla_m = np.nanmedian(desc["dla"][c], axis=0)
        attn_m = np.nanmedian(desc["attn"][c], axis=0)
        for (L, h) in sorted(topk_sets[c]):
            desc_rows.append({"category": c, "layer": L, "head_idx": h,
                              "dla_delta": dla_m[L, h], "attn_to_slot": attn_m[L, h]})
    pd.DataFrame(desc_rows).to_csv(RESULTS / "phase3" / "descriptive.csv", index=False)

    print("\n[4/5] Ablation confirmation")
    abl = run_ablation(model, admitted, topk_sets)
    abl.to_csv(RESULTS / "phase3" / "ablation_results.csv", index=False)

    print("\n[5/5] Cross-category overlap with null models")
    # fast_top_k_selector is exactly equivalent to top_k_head_set (both rank on
    # |median effect|) but skips 144 Wilcoxon tests per resample, which is what
    # makes 10,000 permutations take seconds rather than half an hour.
    selector_topk = M.fast_top_k_selector(top_k)

    tbl_topk = M.compile_cross_category_table(
        topk_sets, effects, selector_topk, n_perm=10000, n_boot=10000, seed=seed,
        n_heads=n_heads_total)
    tbl_topk.insert(0, "selection", f"top{top_k}")
    tbl_topk.to_csv(RESULTS / "phase4" / "cross_category_topk.csv", index=False)

    ovk = M.overlap_vs_k(stats_by_cat, n_heads=n_heads_total)
    ovk.to_csv(RESULTS / "phase4" / "overlap_vs_k.csv", index=False)

    scores = {c: stats_by_cat[c].pivot(index="layer", columns="head_idx",
                                       values="median_effect").values
              for c in stats_by_cat}
    rho = M.rank_correlation_across_categories(scores)
    rho.to_csv(RESULTS / "phase4" / "rank_correlation.csv", index=False)

    for name, sets in (("head_sets_fdr", head_sets), ("head_sets_topk", topk_sets)):
        (RESULTS / "phase3" / f"{name}.json").write_text(
            json.dumps({c: sorted(map(list, v)) for c, v in sets.items()}, indent=2))

    print("\n" + "=" * 72)
    print("CROSS-CATEGORY OVERLAP (top-k selection)")
    print(tbl_topk.to_string(index=False))
    print("\nRANK CORRELATION (threshold-free)")
    print(rho.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    main(model_name=a.model, rebuild=a.rebuild)
