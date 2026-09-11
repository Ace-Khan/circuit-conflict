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
| `00_dataset_construction` | Dataset | Build & filter conflict/unambiguous prompt pairs |
| `01_phase1_baseline_circuit_mapping` | Phase 1 | Map Circuit A & B in isolation via activation patching |
| `02_phase2_conflict_experiment` | Phase 2 | Logit-lens curves + phase transition detection |
| `03_phase3_arbitration_heads` | Phase 3 | Identify, ablate, and patch arbitration head candidates |
| `04_phase4_metrics_statistics` | Phase 4 | Compute all paper metrics with standard errors |
| `05_phase5_figures` | Phase 5 | Publication-quality figures |

Phase 3 carries the result: per-category candidate heads are identified by
conflict-vs-unambiguous activation delta, validated causally by ablation, and then
compared **across** categories. Phases 1–2 exist to establish the per-category
baselines that the overlap metric is computed on.

---

## Interpreting the outcomes

Both directions are informative, but they are not equally strong, and neither is
a new circuit type.

**High overlap (supports H1).** Evidence that arbitration is a reusable,
domain-general mechanism — a meaningful strengthening of the current literature,
which has only ever demonstrated conflict heads one domain at a time. This is the
stronger publishable outcome.

**Low overlap (supports H0).** Evidence that "conflict heads" are task-specific and
that findings from any single conflict domain should not be generalised. This is a
real result rather than a null, because it would be a controlled, single-model
confirmation of the transfer failures already reported piecemeal in replication
work — but it must be defended carefully against the alternative explanation that
GPT-2 Small is simply too small to support a shared mechanism.

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
5. **Winograd difficulty.** GPT-2 Small performs near chance on WCS273. The margin
   filter in Phase 0 selects for genuine ambiguity, but low absolute accuracy limits
   how much signal the logit-difference metric carries.

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
circuit_conflict/
├── pyproject.toml          # dependencies (uv)
├── src/
│   └── circuit_conflict/
│       ├── utils.py        # model loading, logit lens, token helpers
│       ├── dataset.py      # prompt loading, CSV management
│       ├── patching.py     # activation patching & ablation hooks
│       └── metrics.py      # statistics, phase transition, Jaccard
├── notebooks/
│   ├── 00_dataset_construction.ipynb
│   ├── 01_phase1_baseline_circuit_mapping.ipynb
│   ├── 02_phase2_conflict_experiment.ipynb
│   ├── 03_phase3_arbitration_heads.ipynb
│   ├── 04_phase4_metrics_statistics.ipynb
│   └── 05_phase5_figures.ipynb
├── data/
│   ├── prompts/            # prompts.csv (tracked by git)
│   └── results/            # cached tensors & metric CSVs (not tracked)
│       ├── phase1/
│       ├── phase2/
│       ├── phase3/
│       └── phase4/
└── figures/                # output figures (PNG + PDF)
```

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
| Experimental design (minimal-pair construction, choice of estimand, null models) | Author's decisions, developed in dialogue with AI |
| Bug identification and repair | AI-assisted audit; each finding independently reproduced before fixing |
| Results, interpretation, and all claims in the write-up | Author's |

Commits containing AI-assisted work carry a `Co-Authored-By` trailer, so the
provenance is visible in the history rather than asserted only here.

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
