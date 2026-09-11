"""Regenerate the phase notebooks as thin, readable wrappers over the library."""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
HEADER = """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
%load_ext autoreload
%autoreload 2"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
               for c in cells]
    n.metadata = {
        "kernelspec": {"display_name": "Circuit Conflict", "language": "python",
                       "name": "circuit-conflict"},
        "language_info": {"name": "python"},
    }
    return n


NOTEBOOKS = {
"00_dataset_construction": [
("md", """# 00 — Dataset Construction

Builds `data/prompts/prompts.csv`: token-aligned minimal pairs for all three conflict families.

**The design rule.** The control arm keeps every clause of the conflict arm and neutralises the
conflict by swapping **one token**. Both arms therefore tokenise to the same length and share the
same two answer tokens. This is what makes activation patching well-defined and makes a
conflict-minus-control difference immune to length and content confounds.

`build_minimal_pair` asserts equal length and an exact differing-token count, so a misaligned pair
cannot enter the dataset at all.

**Ground truth.** There is no `ground_truth` column. `gold_control` is the answer licensed *by
design* in the control arm — written by the generator from the rule text, a committed fact table,
or grammatical gender agreement. The conflict arm has no gold answer; it has a measured outcome."""),
("code", HEADER),
("code", """from circuit_conflict.utils import load_model
from circuit_conflict import dataset as D
import pandas as pd

model = load_model()"""),
("md", "## 1. Generate the three categories"),
("code", """frames, rejects = [], []
for fn in (D.build_category_a_df, D.build_category_b_df, D.build_category_c_df):
    df_c, rej = fn(model)
    frames.append(df_c); rejects += rej

df = pd.concat(frames, ignore_index=True)
print(f"{len(df)//2} items / {len(df)} rows built, {len(rejects)} rejected")
for r in rejects[:10]:
    print("  reject:", r)"""),
("md", """## 2. Inspect a minimal pair from each category

The two arms should differ in exactly one token."""),
("code", """for cat in ["A", "B", "C"]:
    g = df[df.category == cat]
    c = g[g.arm == "conflict"].iloc[0]; u = g[g.arm == "control"].iloc[0]
    print(f"--- {cat} ---")
    print("  conflict:", c.prompt_text)
    print("  control :", u.prompt_text)
    print(f"  A={c.answer_A!r} (slot-supported)  B={c.answer_B!r}  gold_control={c.gold_control}")
    print(f"  p_slot={c.p_slot}  p_end={c.p_end}  n_tokens={c.n_tokens}\\n")"""),
("md", """## 3. Preconditions

The one legitimate use of the model at dataset time. Measured on a probe prompt **distinct from
both experimental arms**, gating item *eligibility* rather than the outcome label.

An item is admitted only if the model's preference **reverses** when the single disambiguating
token is swapped. The reversal requirement is what rules out "answer_A is simply the more frequent
word" — the failure mode Category B is most exposed to."""),
("code", """df = D.run_preconditions(model, df)
report = D.precondition_report(df)
print(report.to_string(index=False))"""),
("md", """### The pre-registered Category B gate

If Category B's pass rate falls below 0.70, it is **not** a valid instruction-arbitration test on
this model, and the headline overlap is reported on A vs C only, with B relegated to an
exploratory arm."""),
("code", """gate = report.set_index("category").pass_rate
for cat in ["A", "B", "C"]:
    verdict = "PASS" if gate[cat] >= 0.70 else "FAIL — report separately"
    print(f"  {cat}: pass rate {gate[cat]:.2f}  ->  {verdict}")"""),
("md", "## 4. Save"),
("code", """D.save_prompts(df)
admitted = df[df.passes_precondition]
print(f"admitted {len(admitted)//2} of {len(df)//2} items")"""),
],

"01_phase1_patching": [
("md", """# 01 — Phase 1: Position-Resolved Activation Patching

For every admitted item we patch each of the 144 heads from the **control** run into the
**conflict** run and record the normalised effect

$$e = \\frac{d_{\\text{patched}} - d_{\\text{conflict}}}{d_{\\text{control}} - d_{\\text{conflict}}}$$

- `e = 0` → the head is irrelevant
- `e = 1` → this head alone carries the whole conflict effect
- `e < 0` → the head opposes

Normalising **per item** is required: averaging raw logit deltas across items re-introduces an
item-magnitude confound.

Readout is at `p_end`. The reported logit difference is a function of `resid_post[p_end]` and
nothing else, so a head's contribution to the *decision* is exactly its write there."""),
("code", HEADER),
("code", """from circuit_conflict.utils import load_model
from circuit_conflict import dataset as D, pipeline as PL
import numpy as np

model = load_model()
df = D.load_prompts()
admitted = df[df.passes_precondition]
print(f"{len(admitted)//2} admitted items")"""),
("md", """## Run the sweep

Items are batched by `(category, template, length, slot position)`. Grouping on template alone is
invalid when a template has a multi-token filler — those items differ in length and cannot share a
forward pass."""),
("code", """effects, behavioural, descriptive = PL.run_patching(model, admitted)
for c, e in effects.items():
    np.save(PL.RESULTS / "phase1" / f"effects_{c}.npy", e)
    np.save(PL.RESULTS / "phase1" / f"dla_{c}.npy", descriptive["dla"][c])
    np.save(PL.RESULTS / "phase1" / f"attn_slot_{c}.npy", descriptive["attn"][c])
    print(f"  {c}: effects {e.shape}")
behavioural.to_csv(PL.RESULTS / "phase1" / "behavioural.csv", index=False)"""),
("md", """`run_patching` also returns two descriptive measures alongside the causal one: direct logit
attribution at the decision site (paired conflict − control) and attention from the decision site
to the swapped token. Head selection uses only the causal effect; these corroborate it."""),
("md", """## Behavioural summary

`d_conflict` > 0 means the slot/context channel wins the conflict; < 0 means the other channel does."""),
("code", """print(behavioural.groupby("category")[["d_conflict", "d_control", "swing"]].mean().round(2))"""),
],

"02_phase2_logit_lens": [
("md", """# 02 — Phase 2: Logit-Lens Trajectories

Per-item logit-difference curve across layers, for both arms.

**Phase transition** = the first layer at which the curve crosses zero, or `None`. The previous
implementation fell back to `argmax|diff|` whenever there was no sign change, which made the
reported detection rate 100% by construction. A curve that never crosses zero has no phase
transition, and reporting that is the informative answer."""),
("code", HEADER),
("code", """from circuit_conflict.utils import load_model
from circuit_conflict import dataset as D, metrics as M, pipeline as PL
import json

model = load_model()
admitted = D.load_prompts().query("passes_precondition")
lens = PL.run_logit_lens(model, admitted)
lens.to_csv(PL.RESULTS / "phase2" / "phase2_summary.csv", index=False)
lens.head()"""),
("md", "## Phase-transition statistics (note the honest detection rate)"),
("code", """stats = {c: M.phase_transition_stats(
             lens.query("category == @c and arm == 'conflict'").phase_transition_layer.tolist())
         for c in ["A", "B", "C"]}
for c, s in stats.items():
    print(f"  {c}: detected {s['n_detected']}/{s['n_total']} "
          f"({s['detection_rate']:.0%})  mean layer {s['mean']}")
(PL.RESULTS / "phase2" / "phase_transition_stats.json").write_text(json.dumps(stats, indent=2))"""),
],

"03_phase3_arbitration_heads": [
("md", """# 03 — Phase 3: Selecting the Arbitration Heads

Per-head **paired** Wilcoxon signed-rank against zero, Benjamini–Hochberg FDR across all 144 heads
within category, and an effect-size floor.

Paired, not unpaired: the design produces token-aligned pairs, and an unpaired test discards
exactly the pairing the design exists to create."""),
("code", HEADER),
("code", """from circuit_conflict.utils import load_model
from circuit_conflict import dataset as D, metrics as M, pipeline as PL
import numpy as np, json

effects = {c: np.load(PL.RESULTS / "phase1" / f"effects_{c}.npy") for c in ["A", "B", "C"]}
stats_by_cat, head_sets, topk_sets = {}, {}, {}

for c, e in effects.items():
    s = M.paired_head_stats(e); s.insert(0, "category", c)
    s.to_csv(PL.RESULTS / "phase3" / f"head_stats_{c}.csv", index=False)
    stats_by_cat[c] = s
    head_sets[c] = M.select_head_set(s, effect_floor=0.05)
    topk_sets[c] = M.top_k_head_set(s, k=10)
    print(f"  {c}: {len(head_sets[c])} FDR-significant, top-10 selected")"""),
("md", "## The selected head sets"),
("code", """for c, s in topk_sets.items():
    print(f"  {c}: " + " ".join(f"L{l}H{h}" for l, h in sorted(s)))

core = topk_sets["A"] & topk_sets["B"] & topk_sets["C"]
print("\\nShared by all three:", " ".join(f"L{l}H{h}" for l, h in sorted(core)) or "(none)")"""),
("md", """## Validation anchor — Ortu et al. (2024)

Category C reproduces their design, so its head set should recover the heads they report for
GPT-2 Small. **If it does not, the pipeline is wrong** — this is the primary end-to-end check on
the whole apparatus."""),
("code", """ORTU = {(9,6),(9,9),(10,0),(10,10),(10,7),(11,10)}
hit = topk_sets["C"] & ORTU
print(f"recovered {len(hit)}/6: " + " ".join(f"L{l}H{h}" for l, h in sorted(hit)))"""),
("md", """## Causal confirmation by mean-ablation

Mean-ablation over the control distribution, not zero-ablation: a zeroed head is not a state the
model ever occupies, so zero-ablation conflates removing a head's function with feeding the
residual stream something corrupt."""),
("code", """model = load_model()
admitted = D.load_prompts().query("passes_precondition")
abl = PL.run_ablation(model, admitted, topk_sets)
abl.to_csv(PL.RESULTS / "phase3" / "ablation_results.csv", index=False)

json.dump({c: sorted(map(list, s)) for c, s in topk_sets.items()},
          open(PL.RESULTS / "phase3" / "head_sets_topk.json", "w"), indent=2)
abl.groupby("category").mean_delta.agg(["min", "max"]).round(3)"""),
],

"04_phase4_cross_category": [
("md", """# 04 — Phase 4: Cross-Category Overlap (the headline result)

Jaccard overlap between the per-category head sets, reported against **two** nulls, because the
project treats both high and low overlap as informative and a single null can only bound one side.

| Null | Question it answers |
|---|---|
| I — hypergeometric chance floor | Is the overlap above what random k-subsets of 144 heads would give? |
| II — item-label permutation | Is it *below* what one shared mechanism would give? |

Null II is what makes a **low** overlap positive evidence for task-specific arbitration rather
than a bare null result."""),
("code", HEADER),
("code", """from circuit_conflict import dataset as D, metrics as M, pipeline as PL
import numpy as np, pandas as pd, json

effects = {c: np.load(PL.RESULTS / "phase1" / f"effects_{c}.npy") for c in ["A", "B", "C"]}
stats_by_cat = {c: pd.read_csv(PL.RESULTS / "phase3" / f"head_stats_{c}.csv") for c in effects}
topk_sets = {c: M.top_k_head_set(s, k=10) for c, s in stats_by_cat.items()}"""),
("md", "## The chance floor is not zero"),
("code", """null = M.jaccard_null_uniform(10, 10, n_heads=144)
print(f"  expected overlap by chance : {null['expected_jaccard']:.4f}")
print(f"  need >= {null['min_intersection_p05']:.0f} shared heads "
      f"(J >= {null['min_jaccard_p05']:.3f}) to clear p < 0.05")"""),
("md", "## Observed overlap with both nulls, plus a bootstrap CI"),
("code", """selector = lambda e: M.top_k_head_set(M.paired_head_stats(e), k=10)
table = M.compile_cross_category_table(topk_sets, effects, selector,
                                       n_perm=200, n_boot=200, seed=0)
table.to_csv(PL.RESULTS / "phase4" / "cross_category_topk.csv", index=False)
table.round(4)"""),
("md", """## Robustness

A conclusion that only holds at one choice of *k* is not a conclusion, and Spearman rho on the
full 144-head vectors avoids thresholding entirely."""),
("code", """ovk = M.overlap_vs_k(stats_by_cat)
ovk.to_csv(PL.RESULTS / "phase4" / "overlap_vs_k.csv", index=False)
print(ovk.pivot_table(index="k", columns=["cat_a", "cat_b"], values="jaccard").round(3))

scores = {c: s.pivot(index="layer", columns="head_idx", values="median_effect").values
          for c, s in stats_by_cat.items()}
rho = M.rank_correlation_across_categories(scores)
rho.to_csv(PL.RESULTS / "phase4" / "rank_correlation.csv", index=False)
rho.round(4)"""),
],

"05_phase5_figures": [
("md", """# 05 — Phase 5: Figures

Writes PNG (300 dpi) and PDF for every figure to `figures/`."""),
("code", HEADER),
("code", """from circuit_conflict import figures as F
F.main()"""),
("md", """## Headline figure

Both models side by side. The "below a shared mechanism" result holds in both; the "above chance"
result does not survive the move to Pythia-410M."""),
("code", """from IPython.display import Image, display
display(Image(filename=str(F.FIGDIR / "fig0_model_comparison.png"), width=980))"""),
("md", "## Per-model detail"),
("code", """for model in F.MODELS:
    print(f"===== {model} =====")
    for n in ["fig1_head_maps", "fig2_cross_category_overlap",
              "fig3_overlap_vs_k", "fig4_logit_lens", "fig5_ablation"]:
        display(Image(filename=str(F.FIGDIR / f"{n}_{model}.png"), width=880))"""),
],
}

if __name__ == "__main__":
    for old in NB.glob("*.ipynb"):
        old.unlink()
    for name, cells in NOTEBOOKS.items():
        nbf.write(nb(cells), NB / f"{name}.ipynb")
        print(f"  notebooks/{name}.ipynb")
