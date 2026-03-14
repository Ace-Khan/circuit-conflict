# Circuit Conflict — Mechanistic Interpretability Research

**Aether Labs** | GPT-2 Small | TransformerLens

> *Do attention heads in GPT-2 Small act as arbitrators when the model faces two competing valid answers?*

---

## Overview

This project investigates whether GPT-2 Small contains a hidden **arbitration circuit** — a set of attention heads that activate specifically during prompt-level conflict and causally determine which of two competing answers wins.

Three categories of conflict are studied:

| Category | Type | Source |
|---|---|---|
| A | Pronoun / Reference Conflict | Winograd Schema Challenge (WCS273) |
| B | Instruction Conflict | Manually authored |
| C | Factual vs. Contextual Override | Manually authored |

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

## Expected Findings

### Positive result
1–3 attention heads with significantly higher activation during conflict prompts, whose ablation measurably delays or prevents conflict resolution, generalising across ≥2 categories. → Introduces **arbitration circuits** as a new circuit type.

### Negative result
No consistent arbitration heads — conflict resolution appears distributed with no single bottleneck. → Challenges the modular circuit hypothesis and suggests LLM conflict resolution is fundamentally different from single-task circuit behaviour.

**Both are publishable.**

---

## Citation

If you build on this work please cite:

```bibtex
@misc{aetherlabs2025circuitconflict,
  title  = {Circuit Conflict: Arbitration Mechanisms in GPT-2 Small},
  author = {Aether Labs},
  year   = {2026},
  url    = {https://github.com/aetherlabs/circuit-conflict}
}
```
