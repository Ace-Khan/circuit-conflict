# Circuit Conflict — Mechanistic Interpretability Research

**Ace Khan** | GPT-2 Small | TransformerLens

> *When a model resolves a conflict, does it reuse the same machinery regardless of what kind of conflict it is?*

---

## Overview

Conflict-resolution heads are not a new discovery. Prior work has repeatedly found
small sets of attention heads that decide which of two competing answers wins —
but always **within a single conflict type**, and recent replications suggest those
heads may not transfer outside the domain they were found in.

This project asks the transfer question directly. We study **three structurally
unrelated conflict families** in one model, with one methodology, and measure how
much the causally-implicated head set overlaps between them.

| Category | Conflict type | Competing signals | Source |
|---|---|---|---|
| A | Pronoun / reference | Two candidate antecedents | Winograd Schema Challenge (WCS273) |
| B | Instruction | Stated instruction vs. local token statistics | Manually authored |
| C | Factual vs. contextual override | Parametric memory vs. in-context claim | Manually authored |

**Central hypothesis (H1 — domain-general arbitration).** A common set of heads
carries arbitration across all three families; conflict resolution is a reusable
mechanism, largely independent of what is in conflict.

**Alternative (H0 — task-specific arbitration).** Each family recruits its own
heads; "arbitration" is a description of a behaviour, not of a shared circuit.

**Outcome:** neither, cleanly. Overlap is significantly above chance *and* significantly
below a shared-mechanism ceiling, in both models tested — a small shared core plus
substantial per-family machinery. See **Results**.

The primary outcome measure is **cross-category Jaccard overlap of causally
validated head sets**, with per-category ablation effect sizes as support.

---

## Relation to prior work

This project is positioned as a **transfer test**, not a discovery claim. What is
already established, and what is not:

**Already established — treated here as replication, not contribution:**

- **Two-candidate selection in GPT-2 Small.** The IOI circuit (Wang et al., 2022)
  identifies 26 heads across 7 classes — notably S-Inhibition and Name Mover heads —
  that select between two candidate names. Category A is structurally an IOI-like
  task on harder, genuinely ambiguous items.
- **Factual vs. contextual competition.** Ortu et al. (2024) trace this to specific
  GPT-2 heads (L9H6, L9H9, L10H0, L10H10 promote the counterfactual; L10H7, L11H10
  support the factual answer by *suppressing* the competitor). Yu, Merullo & Pavlick
  (2023) show the winner can be flipped by scaling head values. Category C is this task.
- **Ablating conflict heads as an intervention.** Jin et al. (2024, PH3) prune
  conflict-controlling heads to steer the outcome; Taming Knowledge Conflicts
  (ICML 2025) generalises this across 11 datasets and 6 architectures.
- **Method.** Activation patching, zero-ablation sweeps, logit-lens layer traces, and
  Jaccard circuit overlap are all standard tooling.

**Open — where this project sits:**

- **Cross-family transfer is contested.** A 2026 replication of Ortu et al. reports
  that the proposed head ablation "is ineffective for domains that are
  underrepresented in their dataset, and effectiveness varies based on model
  architecture, prompt structure, domain and task." A separate critical
  investigation questions whether the effect reflects factual competition at all
  rather than in-context copying.
- **Partial evidence for sharing exists, but narrowly.** Work on demonstration
  conflict in in-context learning finds late-layer "susceptible heads" that overlap
  across tasks — but only across ICL rule-inference variants, not across
  structurally different conflict families.
- **No study we are aware of** runs coreference, instruction, and factual-override
  conflict through a single model with a single protocol and reports the head-set
  overlap between them. That measurement is this project's contribution.

---

## Phases

| Notebook | Phase | Description |
|---|---|---|
| `00_dataset_construction` | Dataset | Generate token-aligned minimal pairs; run precondition gates |
| `01_phase1_patching` | Phase 1 | Position-resolved activation patching, control → conflict |
| `02_phase2_logit_lens` | Phase 2 | Layer-wise logit-lens trajectories and phase transitions |
| `03_phase3_arbitration_heads` | Phase 3 | Paired stats, FDR, head selection, mean-ablation |
| `04_phase4_cross_category` | Phase 4 | Cross-category overlap against both null models |
| `05_phase5_figures` | Phase 5 | Publication figures |

Or run everything at once:

```bash
uv run python -m circuit_conflict.pipeline && uv run python -m circuit_conflict.figures
```

Phase 3 carries the result: per-category candidate heads are identified by
conflict-vs-unambiguous activation delta, validated causally by ablation, and then
compared **across** categories. Phases 1–2 exist to establish the per-category
baselines that the overlap metric is computed on.

---

## Interpreting the outcomes

Stated before the analysis was run, and retained here so the reasoning can be checked against
what actually happened (see **Results**):

- **High overlap** would mean arbitration is a reusable, domain-general mechanism — a real
  strengthening of a literature that has only ever demonstrated conflict heads one domain at a time.
- **Low overlap** would mean "conflict heads" are task-specific and that findings from any single
  conflict domain should not be generalised — a result rather than a null, because it would be a
  controlled confirmation of the transfer failures reported piecemeal in replication work.

The observed outcome is neither: overlap is above chance *and* below a shared mechanism, in both
models. Partial sharing was not one of the two hypotheses, which is worth stating plainly rather
than retrofitting it to whichever was closer.

---

## Known limitations

These are design constraints to address before writing up, not afterthoughts.

1. **Category B validity.** GPT-2 Small does not follow instructions. Category B
   prompts may measure surface token statistics rather than instruction arbitration.
   Before these prompts count as an arbitration test, they need a behavioural
   precondition check — the model must demonstrably track the instruction in the
   unambiguous case. Prompts failing that check should be excluded or the category
   reported separately.
2. **Single model.** A low-overlap result on GPT-2 Small alone is confounded with
   model capacity. Replicating on a second model (GPT-2 Medium or Pythia-410M)
   closes this and is cheap.
3. **Single seed.** Recent work on attention-head stability finds mid-layer heads
   are unstable across training runs of the same architecture, and that circuits are
   rarely tested for cross-instance robustness. Head-level claims from one
   checkpoint should be stated as such.
4. **Overlap needs a null model.** Jaccard overlap between two sets of *k* heads
   drawn from 144 has a nonzero chance baseline. Report overlap against a permutation
   null, not against zero.
5. **Category A selection bias.** Only 62.5% of generated Category A items passed the
   admission gate, versus 96% and 100% for B and C. Items were kept where the model
   shows the gender-agreement effect, which is a selection on model behaviour. The
   Category A head set is therefore conditioned on items the model handles, and may
   not describe how it treats coreference it gets wrong.
6. **Category C's landmark template is thin.** C2 contributes only 7 items across four
   distinct sequence lengths, so most of Category C rests on the country/capital frame.
7. **Single seed per model.** Two model families are covered, but one checkpoint each.
   Recent work finds mid-layer heads are unstable across training runs of the same
   architecture, so head-level claims describe these checkpoints.
8. **Two models is not many.** Overlap is consistently lower in Pythia-410M, but with n=2
   that cannot be attributed to scale, family, or tokenisation separately.
9. **Category B remains the weakest arm.** It passes its gate on both models, but on models
   this small the B1 template is substantially an induction/copy task; B2, which pits the
   directive against a world-knowledge prior, is the stronger variant.

---

## Setup

### 1 — Install uv (if not already)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2 — Create environment and install dependencies

```bash
cd circuit_conflict
uv sync
```

### 3 — Register the kernel with Jupyter

```bash
uv run python -m ipykernel install --user --name circuit-conflict --display-name "Circuit Conflict"
```

### 4 — Open notebooks

```bash
uv run jupyter lab notebooks/
```

### 5 — Select the `Circuit Conflict` kernel in each notebook before running

---

## Directory Structure

```
circuit-conflict/
├── pyproject.toml / uv.lock    # pinned environment
├── src/circuit_conflict/
│   ├── utils.py       # model loading, logit lens, token/position helpers
│   ├── dataset.py     # minimal-pair generators + precondition gates
│   ├── patching.py    # patching, direct logit attribution, mean-ablation
│   ├── metrics.py     # paired stats, FDR, Jaccard null models
│   ├── pipeline.py    # end-to-end runner
│   └── figures.py     # publication figures
├── notebooks/         # phases 0-5, one per notebook
├── scripts/
│   └── make_notebooks.py       # regenerates the notebooks from source
├── data/
│   ├── prompts/prompts.csv     # generated dataset (tracked)
│   └── results/                # metrics CSV/JSON tracked; .npy arrays not
└── figures/                    # PNG + PDF
```

---

## Results

Two models: **GPT-2 Small** (12L × 12H = 144 heads) and **Pythia-410M** (24L × 16H = 384 heads;
different family, tokeniser and training data). The dataset is rebuilt per model — token ids,
positions, the single-token filter and the gates are all tokeniser- and model-specific.

```bash
uv run python -m circuit_conflict.pipeline --model gpt2
uv run python -m circuit_conflict.pipeline --model pythia-410m --rebuild
uv run python -m circuit_conflict.figures
```

### Pipeline validation

On GPT-2 Small, Category C recovers **5 of Ortu et al. (2024)'s 6 published heads** in its top 10
— L9H9, L10H0, L10H7, L10H10, L11H10 — with signs matching their account (L10H0/L10H10/L9H9
promote the in-context answer; L10H7/L11H10 support the memorised one). The apparatus reproduces a
published result it was not tuned to.

### Headline

| Pair | GPT-2 J | vs chance | Pythia J | vs chance | vs one-mechanism |
|---|---|---|---|---|---|
| A–B | 0.250 (4/10) | p = 0.002 | 0.111 (2/10) | p = 0.025 | p ≤ 0.0001 |
| A–C | 0.333 (5/10) | p = 0.0001 | 0.176 (3/10) | p = 0.001 | p ≤ 0.0001 |
| B–C | 0.250 (4/10) | p = 0.002 | 0.250 (4/10) | p = 0.00005 | p ≤ 0.036 |

Nulls: 10,000 permutations, 10,000 bootstrap resamples. Chance J = 0.036 (GPT-2) / 0.013 (Pythia).

**Every pair, in both models, is significantly above the chance floor and significantly below the
one-shared-mechanism ceiling.** Conflict arbitration is neither a single domain-general mechanism
nor fully task-specific: there is a small shared core plus substantial per-family machinery.

Overlap is consistently lower in Pythia-410M, and its shared core is smaller. Whether that reflects
scale, family, or tokenisation cannot be separated with two models.

### The shared core

Heads appearing in all three categories' top-10 sets:

- **GPT-2 Small** — L10H0, L10H7, L11H10. All three are Ortu et al. heads, found here
  independently on coreference and instruction conflict, so they are not specific to factual
  override.
- **Pythia-410M** — L16H0 only.

Threshold-free corroboration: Spearman ρ on the full head vectors is 0.38–0.57 (GPT-2) and
0.23–0.48 (Pythia), all p < 1e-4. Overlap is stable across k = 5…30, so the result is not an
artefact of k = 10.

### An asymmetry worth more attention than the headline

Mean-ablation effects on GPT-2 Small are an order of magnitude larger in Category C (±1.8 logits)
than in A (±0.27) or B (±0.14). Conflict resolution is far more localised for factual override
than for coreference or instruction conflict. Nearly all prior work sits in factual override —
this is a direct caution against generalising from it, and arguably the most useful finding here.

### Negative result: occupation-based Winograd items are unusable at this scale

The first Category A design followed the Winograd template (*"The nurse is a man. The driver is a
woman…"*). Swapping the stated gender moved GPT-2 Small's pronoun preference by **~0.1 logits** —
0 of 32 items passed the gate. The model runs on a fixed lexical prior over occupation pairs and
effectively ignores the gender statement. Swapping a **name's** gender moves it ~2.5 logits, so
Category A was rebuilt on names.

This is why WSC273 is not used for patching: at this scale such items measure lexical priors, not
coreference.

### Corrections made during analysis

Recorded because the intermediate readings were wrong in ways worth knowing about:

1. **The Category A gate was mis-specified.** One of its two probes had both names male — an
   ambiguous item with no correct answer — so the gate required the model to prefer a particular
   name under ambiguity. That selected on exactly the lexical prior the gate was meant to control
   for, and rejected ~40% of Category A on GPT-2 and ~68% on Pythia. With both probes carrying
   exactly one male name, admission is 100% in both models, and the earlier "Pythia Category A is
   underpowered" caveat no longer applies.
2. **Before that fix**, Pythia's A–B overlap was not above chance, and the reading was that the
   above-chance result failed to replicate. It does replicate; the null was an artefact of the gate.
3. **`logit_lens_diff` omitted the unembedding bias**, shifting every curve by a constant and
   moving the zero crossing that the phase-transition measure keys on.
4. **Category A was not counterbalanced** — `answer_A` was always the first-mentioned name, so
   answer identity was confounded with position. It is now balanced across both sides.
5. **Category B's template and counterbalance were perfectly correlated**, since both derived from
   the same counter. They are now crossed.

---

## Methods & tooling

This project is authored and directed by **Ace Khan**, who is responsible for the
research question, the experimental design decisions, the interpretation of results,
and any errors.

Development was AI-assisted, and that assistance is declared here rather than
buried. Concretely:

| Component | Nature of assistance |
|---|---|
| Initial scaffold (library structure, notebook skeletons, plotting boilerplate) | AI-generated, subsequently audited and substantially rewritten |
| Prompt generators and fact tables | AI-drafted from specifications set by the author; verified item by item |
| Literature positioning | AI-assisted search; every citation read and checked against the claim it supports |
| Experimental design (minimal-pair construction, choice of estimand, null models) | Author's design |
| Bug identification and repair | AI-assisted audit; each finding independently reproduced before fixing |
| Test suite | AI-drafted against invariants specified by the author |
| Results, interpretation, and all claims in the write-up | Author's |

**What this does not mean.** No result in this repository is reported on the
authority of a language model. Every number is produced by the committed code
running on a pinned environment, and the pipeline is validated against an
independent published result (see Verification below) precisely so that its output
does not depend on trusting the process that generated the code.

### Verification anchor

Category C deliberately reproduces the design of Ortu et al. (2024), who report
specific GPT-2 Small heads for factual/counterfactual competition (L9H6, L9H9,
L10H0, L10H10 promoting the counterfactual; L10H7 and L11H10 supporting the factual
answer by suppression). If this pipeline is correct, the Category C head set should
substantially recover those heads. If it does not, the pipeline is wrong — this is
the primary end-to-end check on the whole apparatus.

---

## References

- Wang et al. (2022), *Interpretability in the Wild: a Circuit for Indirect Object
  Identification in GPT-2 small*. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Yu, Merullo & Pavlick (2023), *Characterizing Mechanisms for Factual Recall in
  Language Models*. [arXiv:2310.15910](https://arxiv.org/abs/2310.15910)
- Ortu et al. (2024), *Competition of Mechanisms: Tracing How Language Models Handle
  Facts and Counterfactuals*. ACL. [arXiv:2402.11655](https://arxiv.org/abs/2402.11655)
- Jin et al. (2024), *Cutting Off the Head Ends the Conflict: A Mechanism for
  Interpreting and Mitigating Knowledge Conflicts*. ACL Findings.
  [arXiv:2402.18154](https://arxiv.org/abs/2402.18154)
- *Taming Knowledge Conflicts in Language Models*. ICML 2025.
  [arXiv:2503.10996](https://arxiv.org/abs/2503.10996)
- *On the Generalizability of "Competition of Mechanisms"*.
  [arXiv:2506.22977](https://arxiv.org/abs/2506.22977)
- *Tracing Facts or just Copies? A critical investigation of the Competitions of
  Mechanisms in Large Language Models*. [arXiv:2507.11809](https://arxiv.org/abs/2507.11809)
- *Understanding the Dynamics of Demonstration Conflict in In-Context Learning*.
  [arXiv:2603.04464](https://arxiv.org/abs/2603.04464)
- *Quantifying LLM Attention-Head Stability: Implications for Circuit Universality*.
  [arXiv:2602.16740](https://arxiv.org/abs/2602.16740)

---

## Citation

If you build on this work please cite:

```bibtex
@misc{khan2026circuitconflict,
  title  = {Circuit Conflict: Does Arbitration Transfer Across Conflict Types in GPT-2 Small?},
  author = {Khan, Ace},
  year   = {2026}
}
```
