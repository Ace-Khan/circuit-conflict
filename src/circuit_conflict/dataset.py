"""
dataset.py — prompt I/O, CSV schema, and Category B/C manual prompts.

CSV schema (data/prompts/prompts.csv)
--------------------------------------
prompt_id        : unique string, e.g. "A_001"
category         : "A" | "B" | "C"
is_conflict      : bool  (True = ambiguous conflict version)
prompt_text      : full prompt string
answer_A         : single word (answer A — must be a single GPT-2 token)
answer_B         : single word (answer B — must be a single GPT-2 token)
ground_truth     : "A" | "B"  (which answer the model actually outputs)
notes            : optional free-text
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_CSV = REPO_ROOT / "data" / "prompts" / "prompts.csv"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

SCHEMA = [
    "prompt_id", "category", "is_conflict",
    "prompt_text", "answer_A", "answer_B",
    "ground_truth", "notes",
]


def load_prompts(path: Path = PROMPTS_CSV) -> pd.DataFrame:
    """Load the prompts CSV, enforcing the expected schema."""
    df = pd.read_csv(path)
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prompts CSV: {missing}")
    df["is_conflict"] = df["is_conflict"].astype(bool)
    return df


def save_prompts(df: pd.DataFrame, path: Path = PROMPTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[SCHEMA].to_csv(path, index=False)
    print(f"Saved {len(df)} prompts → {path}")


def empty_prompts_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


# ---------------------------------------------------------------------------
# Category B — Instruction Conflict (manually authored)
# ---------------------------------------------------------------------------
# Each dict has keys matching SCHEMA minus prompt_id (assigned at build time).

CATEGORY_B_CONFLICT = [
    {
        "prompt_text": (
            "Rule 1: Always respond in formal, structured prose with complete sentences. "
            "Rule 2: Always respond casually and directly with just the bare fact, no fluff. "
            "Question: What is the capital of France? Answer:"
        ),
        "answer_A": "Paris",     # bare casual answer
        "answer_B": "The",       # start of formal sentence "The capital of France is Paris."
        "notes": "formal vs casual instruction conflict — capital fact query",
    },
    {
        "prompt_text": (
            "Instruction A: Write in an academic, third-person style at all times. "
            "Instruction B: Write informally, use first-person and contractions. "
            "Now answer: Who invented the telephone? Answer:"
        ),
        "answer_A": "Bell",      # informal bare answer
        "answer_B": "Alexander", # start of formal sentence
        "notes": "academic vs informal style conflict — inventor query",
    },
    {
        "prompt_text": (
            "Protocol 1: Respond only with a single number, nothing else. "
            "Protocol 2: Respond only with a complete English sentence. "
            "How many days are in a week? Answer:"
        ),
        "answer_A": "7",         # number-only
        "answer_B": "There",     # start of "There are seven days..."
        "notes": "number vs sentence format conflict — days in week",
    },
    {
        "prompt_text": (
            "Style directive: Be maximally concise — one word only. "
            "Tone directive: Provide a warm, detailed explanation. "
            "What do plants need to grow? Answer:"
        ),
        "answer_A": "Sunlight",  # one-word concise
        "answer_B": "Plants",    # start of detailed explanation
        "notes": "concise vs detailed conflict — plant biology",
    },
    {
        "prompt_text": (
            "Format rule A: Answer only in bullet points starting with dashes. "
            "Format rule B: Answer only in flowing prose paragraphs. "
            "Describe the water cycle. Answer:"
        ),
        "answer_A": "-",         # bullet point start
        "answer_B": "Water",     # prose start
        "notes": "bullet vs prose format conflict — water cycle",
    },
    {
        "prompt_text": (
            "Directive 1: Use technical jargon and precise scientific terminology. "
            "Directive 2: Explain as if speaking to a five-year-old child. "
            "What is photosynthesis? Answer:"
        ),
        "answer_A": "Photo",     # start of technical term "Photosynthesis is..."
        "answer_B": "Plants",    # kid-friendly "Plants use sunlight..."
        "notes": "expert vs simplified register conflict — photosynthesis",
    },
    {
        "prompt_text": (
            "Rule: Always begin your answer with 'Yes' or 'No'. "
            "Rule: Never start your answer with 'Yes' or 'No'. "
            "Is the Earth round? Answer:"
        ),
        "answer_A": "Yes",       # comply with rule 1
        "answer_B": "The",       # comply with rule 2 — "The Earth is..."
        "notes": "yes/no start vs prohibition conflict",
    },
    {
        "prompt_text": (
            "Instruction: Respond in exactly three words. "
            "Instruction: Respond in exactly one word. "
            "What color is the sky? Answer:"
        ),
        "answer_A": "Blue",      # one-word response
        "answer_B": "The",       # start of three-word phrase "The sky is..."
        "notes": "one-word vs three-word length conflict — sky color",
    },
    {
        "prompt_text": (
            "Guide A: Always state the most common answer that the general public would give. "
            "Guide B: Always state the most technically accurate scientific answer. "
            "What is the boiling point of water? Answer:"
        ),
        "answer_A": "100",       # common answer (100°C)
        "answer_B": "It",        # "It depends on altitude/pressure..." — scientific hedge
        "notes": "common vs scientific precision conflict — boiling point",
    },
    {
        "prompt_text": (
            "Order A: Respond pessimistically, focusing on risks and downsides. "
            "Order B: Respond optimistically, focusing on benefits and opportunities. "
            "Should someone learn to code? Answer:"
        ),
        "answer_A": "Learning",  # positive/optimistic start
        "answer_B": "It",        # "It can be difficult..." pessimistic
        "notes": "pessimistic vs optimistic framing conflict — coding",
    },
    # Unambiguous versions (is_conflict=False) are paired below
]

CATEGORY_B_UNAMBIGUOUS = [
    {
        "prompt_text": (
            "Rule 1: Always respond casually and directly with just the bare fact, no fluff. "
            "Question: What is the capital of France? Answer:"
        ),
        "answer_A": "Paris",
        "answer_B": "The",
        "ground_truth": "A",
        "notes": "unambiguous casual — capital fact query",
    },
    {
        "prompt_text": (
            "Instruction: Write informally, use first-person and contractions. "
            "Now answer: Who invented the telephone? Answer:"
        ),
        "answer_A": "Bell",
        "answer_B": "Alexander",
        "ground_truth": "A",
        "notes": "unambiguous informal — inventor query",
    },
    {
        "prompt_text": (
            "Protocol: Respond only with a single number, nothing else. "
            "How many days are in a week? Answer:"
        ),
        "answer_A": "7",
        "answer_B": "There",
        "ground_truth": "A",
        "notes": "unambiguous number-only — days in week",
    },
    {
        "prompt_text": (
            "Style directive: Be maximally concise — one word only. "
            "What do plants need to grow? Answer:"
        ),
        "answer_A": "Sunlight",
        "answer_B": "Plants",
        "ground_truth": "A",
        "notes": "unambiguous concise — plant biology",
    },
    {
        "prompt_text": (
            "Format rule: Answer only in flowing prose paragraphs. "
            "Describe the water cycle. Answer:"
        ),
        "answer_A": "-",
        "answer_B": "Water",
        "ground_truth": "B",
        "notes": "unambiguous prose — water cycle",
    },
    {
        "prompt_text": (
            "Directive: Explain as if speaking to a five-year-old child. "
            "What is photosynthesis? Answer:"
        ),
        "answer_A": "Photo",
        "answer_B": "Plants",
        "ground_truth": "B",
        "notes": "unambiguous simple — photosynthesis",
    },
    {
        "prompt_text": (
            "Rule: Always begin your answer with 'Yes' or 'No'. "
            "Is the Earth round? Answer:"
        ),
        "answer_A": "Yes",
        "answer_B": "The",
        "ground_truth": "A",
        "notes": "unambiguous yes/no — earth round",
    },
    {
        "prompt_text": (
            "Instruction: Respond in exactly one word. "
            "What color is the sky? Answer:"
        ),
        "answer_A": "Blue",
        "answer_B": "The",
        "ground_truth": "A",
        "notes": "unambiguous one-word — sky color",
    },
    {
        "prompt_text": (
            "Guide: Always state the most common answer that the general public would give. "
            "What is the boiling point of water? Answer:"
        ),
        "answer_A": "100",
        "answer_B": "It",
        "ground_truth": "A",
        "notes": "unambiguous common answer — boiling point",
    },
    {
        "prompt_text": (
            "Order: Respond optimistically, focusing on benefits and opportunities. "
            "Should someone learn to code? Answer:"
        ),
        "answer_A": "Learning",
        "answer_B": "It",
        "ground_truth": "A",
        "notes": "unambiguous optimistic — coding",
    },
]


# ---------------------------------------------------------------------------
# Category C — Factual vs. Contextual Override (manually authored)
# ---------------------------------------------------------------------------

CATEGORY_C_CONFLICT = [
    {
        "prompt_text": (
            "The capital of Australia is Sydney. "
            "Given this, what is the capital of Australia? Answer:"
        ),
        "answer_A": "Sydney",    # context answer (wrong fact)
        "answer_B": "Canberra",  # trained knowledge (correct)
        "notes": "false context: Australia capital — Sydney vs Canberra",
    },
    {
        "prompt_text": (
            "As established in our records, World War II ended in 1944. "
            "Based on this, when did World War II end? Answer:"
        ),
        "answer_A": "1944",   # context answer (wrong)
        "answer_B": "1945",   # trained knowledge (correct)
        "notes": "false context: WW2 end date — 1944 vs 1945",
    },
    {
        "prompt_text": (
            "For reference: water freezes at 0 degrees Fahrenheit. "
            "At what temperature does water freeze? Answer:"
        ),
        "answer_A": "0",   # context gives 0°F (wrong)
        "answer_B": "32",  # trained: 32°F / 0°C
        "notes": "false context: water freezing point — 0°F vs 32°F",
    },
    {
        "prompt_text": (
            "Note: The Great Wall of China is located in Japan. "
            "Based on the above, where is the Great Wall of China located? Answer:"
        ),
        "answer_A": "Japan",   # context answer
        "answer_B": "China",   # trained knowledge
        "notes": "false context: Great Wall location",
    },
    {
        "prompt_text": (
            "According to our data, the speed of light is 100,000 miles per second. "
            "How fast does light travel? The speed of light is"
        ),
        "answer_A": "100",     # context: 100,000 mph
        "answer_B": "186",     # trained: ~186,000 miles/sec
        "notes": "false context: speed of light",
    },
    {
        "prompt_text": (
            "Per the updated records, Mount Everest is 5,000 meters tall. "
            "How tall is Mount Everest? It is"
        ),
        "answer_A": "5",     # context: 5,000 m
        "answer_B": "8",     # trained: ~8,849 m
        "notes": "false context: Everest height",
    },
    {
        "prompt_text": (
            "Context established: Napoleon Bonaparte was born in France. "
            "Where was Napoleon Bonaparte born? He was born in"
        ),
        "answer_A": "France",   # context (wrong — born in Corsica)
        "answer_B": "Corsica",  # trained knowledge
        "notes": "false context: Napoleon birthplace",
    },
    {
        "prompt_text": (
            "Our records state that the Eiffel Tower is located in Berlin. "
            "In which city is the Eiffel Tower? It is in"
        ),
        "answer_A": "Berlin",  # false context
        "answer_B": "Paris",   # trained knowledge
        "notes": "false context: Eiffel Tower city",
    },
    {
        "prompt_text": (
            "As per today's data, the human body has 100 bones. "
            "How many bones does the adult human body have? It has"
        ),
        "answer_A": "100",  # false context
        "answer_B": "206",  # trained (206 bones)
        "notes": "false context: human bone count",
    },
    {
        "prompt_text": (
            "Note: Shakespeare was born in London, England. "
            "Where was William Shakespeare born? He was born in"
        ),
        "answer_A": "London",      # false context
        "answer_B": "Stratford",   # trained: Stratford-upon-Avon
        "notes": "false context: Shakespeare birthplace",
    },
    {
        "prompt_text": (
            "Reference: The Amazon River is the longest river in the world. "
            "Which is the longest river in the world? The longest river is the"
        ),
        "answer_A": "Amazon",  # false context (Nile is longest)
        "answer_B": "Nile",    # trained knowledge
        "notes": "false context: longest river (Amazon vs Nile)",
    },
]

CATEGORY_C_UNAMBIGUOUS = [
    {
        "prompt_text": "What is the capital of Australia? Answer:",
        "answer_A": "Sydney",
        "answer_B": "Canberra",
        "ground_truth": "B",
        "notes": "unambiguous: Australia capital",
    },
    {
        "prompt_text": "When did World War II end? Answer:",
        "answer_A": "1944",
        "answer_B": "1945",
        "ground_truth": "B",
        "notes": "unambiguous: WW2 end",
    },
    {
        "prompt_text": "At what temperature in Fahrenheit does water freeze? Answer:",
        "answer_A": "0",
        "answer_B": "32",
        "ground_truth": "B",
        "notes": "unambiguous: water freezing point",
    },
    {
        "prompt_text": "Where is the Great Wall of China located? Answer:",
        "answer_A": "Japan",
        "answer_B": "China",
        "ground_truth": "B",
        "notes": "unambiguous: Great Wall location",
    },
    {
        "prompt_text": "How fast does light travel in miles per second? The speed of light is",
        "answer_A": "100",
        "answer_B": "186",
        "ground_truth": "B",
        "notes": "unambiguous: speed of light",
    },
    {
        "prompt_text": "How tall is Mount Everest roughly in thousands of meters? It is",
        "answer_A": "5",
        "answer_B": "8",
        "ground_truth": "B",
        "notes": "unambiguous: Everest height",
    },
    {
        "prompt_text": "Where was Napoleon Bonaparte born? He was born in",
        "answer_A": "France",
        "answer_B": "Corsica",
        "ground_truth": "B",
        "notes": "unambiguous: Napoleon birthplace",
    },
    {
        "prompt_text": "In which city is the Eiffel Tower? It is in",
        "answer_A": "Berlin",
        "answer_B": "Paris",
        "ground_truth": "B",
        "notes": "unambiguous: Eiffel Tower city",
    },
    {
        "prompt_text": "How many bones does the adult human body have? It has",
        "answer_A": "100",
        "answer_B": "206",
        "ground_truth": "B",
        "notes": "unambiguous: human bone count",
    },
    {
        "prompt_text": "Where was William Shakespeare born? He was born in",
        "answer_A": "London",
        "answer_B": "Stratford",
        "ground_truth": "B",
        "notes": "unambiguous: Shakespeare birthplace",
    },
    {
        "prompt_text": "Which is the longest river in the world? The longest river is the",
        "answer_A": "Amazon",
        "answer_B": "Nile",
        "ground_truth": "B",
        "notes": "unambiguous: longest river",
    },
]


# ---------------------------------------------------------------------------
# Winograd Schema helpers (Category A)
# ---------------------------------------------------------------------------

def build_winograd_rows(
    wsc_dataset,
    model,
    confidence_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Filter the WCS273 dataset to extract genuinely conflicted prompts.

    A prompt is "conflicted" when |logit_diff(pronoun_A, pronoun_B)| <
    confidence_threshold * (|logit_A| + |logit_B|).

    Parameters
    ----------
    wsc_dataset : HuggingFace dataset split (wsc273 train split)
    model       : loaded HookedTransformer
    confidence_threshold : relative margin below which we consider the model
                          "conflicted" (default 0.15 = 15 % margin)

    Returns
    -------
    pd.DataFrame with one row per (conflict, unambiguous) pair that passes
    the filter.
    """
    import torch
    from circuit_conflict.utils import get_answer_token_id, logit_diff as ld

    rows = []
    pair_idx = 0

    for ex in wsc_dataset:
        text = ex["text"]
        span1 = ex["span1_text"]   # candidate A (the referent of the pronoun)
        span2 = ex["span2_text"]   # candidate B
        label = ex["label"]        # True → span1 is correct

        # Try to get single-token ids for both candidates
        try:
            tok_A = get_answer_token_id(model, span1.split()[0])
            tok_B = get_answer_token_id(model, span2.split()[0])
        except ValueError:
            continue

        tokens = model.to_tokens(text)
        with torch.no_grad():
            logits = model(tokens)
        diff = ld(logits, tok_A, tok_B)

        total = abs(logits[0, -1, tok_A].item()) + abs(logits[0, -1, tok_B].item())
        margin = abs(diff) / (total + 1e-8)

        ground_truth = "A" if label else "B"

        # Build unambiguous version by replacing the pronoun with the answer
        # (simple heuristic: insert "Specifically, <answer>." before the coreference)
        pronoun = ex.get("span2_text", "it")
        unambiguous_text = text.replace(pronoun, span1 if label else span2, 1)

        conflict_row = dict(
            category="A",
            is_conflict=True,
            prompt_text=text,
            answer_A=span1.split()[0],
            answer_B=span2.split()[0],
            ground_truth=ground_truth,
            notes=f"margin={margin:.3f}",
        )
        unamb_row = dict(
            category="A",
            is_conflict=False,
            prompt_text=unambiguous_text,
            answer_A=span1.split()[0],
            answer_B=span2.split()[0],
            ground_truth=ground_truth,
            notes=f"unambiguous pair for A_{pair_idx:03d}",
        )

        if margin < confidence_threshold:
            rows.append(conflict_row)
            rows.append(unamb_row)
            pair_idx += 1

        if pair_idx >= 30:
            break

    df = pd.DataFrame(rows)
    # Assign prompt_ids
    pair = 0
    ids = []
    for i, row in df.iterrows():
        if row["is_conflict"]:
            ids.append(f"A_{pair:03d}_conflict")
        else:
            ids.append(f"A_{pair:03d}_unamb")
            pair += 1
    df.insert(0, "prompt_id", ids)
    return df


# ---------------------------------------------------------------------------
# Build Category B & C DataFrames from manual lists
# ---------------------------------------------------------------------------

def build_category_b_df() -> pd.DataFrame:
    rows = []
    for i, (conf, unamb) in enumerate(zip(CATEGORY_B_CONFLICT, CATEGORY_B_UNAMBIGUOUS)):
        # Conflict row (ground truth determined at model-run time in notebook 00)
        rows.append(dict(
            prompt_id=f"B_{i:03d}_conflict",
            category="B",
            is_conflict=True,
            prompt_text=conf["prompt_text"],
            answer_A=conf["answer_A"],
            answer_B=conf["answer_B"],
            ground_truth="TBD",  # filled in notebook 00
            notes=conf["notes"],
        ))
        # Unambiguous row
        rows.append(dict(
            prompt_id=f"B_{i:03d}_unamb",
            category="B",
            is_conflict=False,
            prompt_text=unamb["prompt_text"],
            answer_A=unamb["answer_A"],
            answer_B=unamb["answer_B"],
            ground_truth=unamb["ground_truth"],
            notes=unamb["notes"],
        ))
    return pd.DataFrame(rows, columns=SCHEMA)


def build_category_c_df() -> pd.DataFrame:
    rows = []
    for i, (conf, unamb) in enumerate(zip(CATEGORY_C_CONFLICT, CATEGORY_C_UNAMBIGUOUS)):
        rows.append(dict(
            prompt_id=f"C_{i:03d}_conflict",
            category="C",
            is_conflict=True,
            prompt_text=conf["prompt_text"],
            answer_A=conf["answer_A"],
            answer_B=conf["answer_B"],
            ground_truth="TBD",
            notes=conf["notes"],
        ))
        rows.append(dict(
            prompt_id=f"C_{i:03d}_unamb",
            category="C",
            is_conflict=False,
            prompt_text=unamb["prompt_text"],
            answer_A=unamb["answer_A"],
            answer_B=unamb["answer_B"],
            ground_truth=unamb["ground_truth"],
            notes=unamb["notes"],
        ))
    return pd.DataFrame(rows, columns=SCHEMA)
