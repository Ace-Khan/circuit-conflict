"""
dataset.py — minimal-pair prompt construction for the three conflict families.

Design rule (this is the whole point of the module)
---------------------------------------------------
    The CONTROL arm keeps every clause of the CONFLICT arm and neutralises the
    conflict by swapping exactly ONE token.

Consequences, all of which the previous clause-deletion design lacked:
  * conflict and control tokenise to the SAME length  -> activation patching is
    well-defined (you cannot patch between different-length runs);
  * the pair is identical everywhere except one index -> a conflict-minus-control
    difference cannot be a length or content confound;
  * both arms share the same two answer tokens        -> the logit difference is
    the same estimand in both arms.

Sign convention (uniform across A, B and C)
-------------------------------------------
    answer_A = the answer supported by the SWAPPED SLOT
               (the in-context / recency-licensed / stated-rule channel)
    answer_B = the answer supported by the OTHER channel
               (parametric memory / grammatical constraint / the first rule)

so `gold_control == "B"` for every item in every category, and per-head effect
signs are directly comparable across categories.  Without this the cross-category
Jaccard would be comparing sign-flipped sets.

Ground truth
------------
There is deliberately NO `ground_truth` column.  The previous schema defined it as
"which answer the model actually outputs", which is the dependent variable — using
it as a label made accuracy 100% by construction.  Instead:

    gold_control  the answer licensed BY DESIGN in the control arm, written by the
                  generator from the rule text / a committed fact table /
                  grammatical gender agreement.  The model is never consulted.

The CONFLICT arm has no gold answer.  It has a measured outcome.  That is the point.

The model is used only for PRECONDITIONS, measured on a separate probe prompt, and
gates item eligibility rather than the outcome label.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# dataset.py lives at <root>/src/circuit_conflict/dataset.py, so parents[2] is
# the repo root.  (This was parents[3], which resolved OUTSIDE the repo and
# silently wrote prompts.csv into the parent directory.)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
PROMPTS_CSV = PROMPTS_DIR / "prompts.csv"   # default (gpt2)


def model_slug(model_name: str) -> str:
    """Filesystem-safe model identifier."""
    return model_name.replace("/", "_")


def prompts_path(model_name: str = "gpt2") -> Path:
    """
    Per-model prompt file.

    The dataset is NOT model-portable: token ids, position indices, the
    single-token vocabulary filter and the precondition gates are all specific to
    one tokeniser and one model's behaviour.  A second model needs its own build.
    """
    return PROMPTS_DIR / f"prompts_{model_slug(model_name)}.csv"

SCHEMA = [
    "item_id",              # pairing key, e.g. "B_007"
    "category",             # "A" | "B" | "C"
    "arm",                  # "conflict" | "control"
    "template_id",          # surface variant, e.g. "B1"
    "counterbalance",       # which side the swapped slot sits on
    "prompt_text",
    "answer_A", "answer_B",         # strings
    "token_A", "token_B",           # single token ids
    "gold_control",                 # always "B" under the sign convention
    "p_end", "p_slot", "p_A", "p_B",  # token positions (BOS-prepended); -1 = absent
    "n_tokens",
    "passes_precondition",
    "precondition_margin",
]


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def empty_prompts_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


def save_prompts(df: pd.DataFrame, path: Path = PROMPTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows -> {path}")


def load_prompts(path: Path = PROMPTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"prompts.csv is missing columns: {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# Vocabularies (filtered to single tokens at build time)
# ---------------------------------------------------------------------------

# Category A uses NAMES, not occupations.  Measured on GPT-2 Small, swapping an
# occupation's stated gender moves the pronoun preference by ~0.1 logits (i.e.
# not at all -- the model runs on a fixed lexical prior over occupation pairs),
# whereas swapping a name's gender moves it by ~2.5 logits.  Occupation-based
# Winograd-style items are therefore not a usable coreference probe at this scale.
MALE_NAMES = [
    "John", "James", "Robert", "Michael", "David", "Paul", "Peter", "Tom",
    "Mark", "Steven", "Brian", "Kevin", "George", "Edward", "Richard",
    "Charles", "Daniel", "Andrew", "Joseph", "Thomas",
]
FEMALE_NAMES = [
    "Mary", "Sarah", "Anna", "Emily", "Kate", "Laura", "Susan", "Linda",
    "Karen", "Jane", "Lisa", "Julia", "Helen", "Alice", "Grace", "Rachel",
    "Diana", "Nancy", "Carol", "Amy",
]

RULE_WORDS = [
    "cat", "dog", "red", "blue", "fish", "bird", "tree", "rock", "sun",
    "moon", "car", "book", "door", "hand", "gold", "green", "north", "south",
    "salt", "rain",
]

# Committed reference table. `gold_control` for Category C is read from here,
# never from the model.
COUNTRY_CAPITAL = [
    ("France", "Paris"), ("Italy", "Rome"), ("England", "London"),
    ("Germany", "Berlin"), ("Japan", "Tokyo"), ("Russia", "Moscow"),
    ("Spain", "Madrid"), ("Greece", "Athens"), ("Ireland", "Dublin"),
    ("Austria", "Vienna"), ("Norway", "Oslo"), ("Portugal", "Lisbon"),
    ("Egypt", "Cairo"), ("Cuba", "Havana"), ("Peru", "Lima"),
    ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"),
]

LANDMARK_CITY = [
    ("Eiffel Tower", "Paris"), ("Colosseum", "Rome"), ("Big Ben", "London"),
    ("Brandenburg Gate", "Berlin"), ("Kremlin", "Moscow"),
    ("Acropolis", "Athens"), ("Sphinx", "Cairo"),
]


def filter_single_token(model, words: list[str]) -> tuple[list[str], list[str]]:
    """Split `words` into (kept, rejected) by single-token-ness."""
    from circuit_conflict.utils import is_single_token
    kept, rejected = [], []
    for w in words:
        (kept if is_single_token(model, w) else rejected).append(w)
    return kept, rejected


# ---------------------------------------------------------------------------
# The minimal-pair builder — the structural guarantee lives here
# ---------------------------------------------------------------------------

class PairRejected(Exception):
    pass


def build_minimal_pair(
    model,
    item_id: str,
    category: str,
    template_id: str,
    counterbalance: str,
    conflict_text: str,
    control_text: str,
    answer_A: str,
    answer_B: str,
    n_expected_diff: int = 1,
) -> list[dict]:
    """
    Build one conflict/control row pair, or raise PairRejected.

    The assertions here are the mechanism that prevents the length/alignment
    flaws from ever recurring: a pair that is not token-aligned cannot enter the
    dataset, so no downstream phase has to defend against one.
    """
    from circuit_conflict.utils import (
        find_token_position,
        require_single_token,
        token_diff_positions,
    )

    try:
        tok_A = require_single_token(model, answer_A)
        tok_B = require_single_token(model, answer_B)
    except ValueError as e:
        raise PairRejected(f"answer not single-token: {e}") from e

    if tok_A == tok_B:
        raise PairRejected(f"answer_A and answer_B are the same token ({answer_A!r})")

    t_conf = model.to_tokens(conflict_text)
    t_ctrl = model.to_tokens(control_text)

    if t_conf.shape != t_ctrl.shape:
        raise PairRejected(
            f"length mismatch: conflict {tuple(t_conf.shape)} vs control {tuple(t_ctrl.shape)}"
        )

    diffs = token_diff_positions(t_conf, t_ctrl)
    if len(diffs) != n_expected_diff:
        raise PairRejected(
            f"expected {n_expected_diff} differing token(s), found {len(diffs)} at {diffs}"
        )

    p_slot = diffs[0]
    n_tokens = int(t_conf.shape[1])
    p_end = n_tokens - 1

    rows = []
    for arm, text, toks in (("conflict", conflict_text, t_conf),
                            ("control", control_text, t_ctrl)):
        pa = find_token_position(toks, tok_A, last=True)
        pb = find_token_position(toks, tok_B, last=True)
        rows.append({
            "item_id": item_id,
            "category": category,
            "arm": arm,
            "template_id": template_id,
            "counterbalance": counterbalance,
            "prompt_text": text,
            "answer_A": answer_A, "answer_B": answer_B,
            "token_A": tok_A, "token_B": tok_B,
            "gold_control": "B",
            "p_end": p_end,
            "p_slot": p_slot,
            "p_A": -1 if pa is None else pa,
            "p_B": -1 if pb is None else pb,
            "n_tokens": n_tokens,
            "passes_precondition": False,
            "precondition_margin": float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# Category A — coreference (gender agreement fixes the control gold answer)
# ---------------------------------------------------------------------------
#
#   "The {OCC1} is a {G1}. The {OCC2} is a {G2}. The {OCC1} spoke to the {OCC2}
#    before he left the office. The one who left the office was the"
#
# conflict: both are men  -> the pronoun "he" is genuinely ambiguous
# control : one is a woman -> grammatical gender agreement licenses exactly one
#
# The control's gold answer is therefore model-independent, which is what makes
# it a real control rather than a second reading of the model's own preference.

TEMPLATE_A = "When {n1} and {n2} went to the store, he bought a drink. The buyer was"

# The probe uses a different scenario and verb from both experimental arms, so it
# shares no surface structure with them.
PROBE_A = "{n1} and {n2} were talking quietly. He said hello. The speaker was"


def build_category_a_df(model, n_items: int = 40, seed: int = 0) -> tuple[pd.DataFrame, list[dict]]:
    """
    Category A -- referential conflict between two candidate antecedents.

    conflict: both candidates are male  -> "he" is genuinely ambiguous
    control : the first name is female  -> gender agreement licenses exactly one

    The control's gold answer is fixed by grammatical agreement with the pronoun,
    so it is model-independent.  The swapped slot is the first name.
    """
    import random
    rng = random.Random(seed)

    males, rej_m = filter_single_token(model, MALE_NAMES)
    females, rej_f = filter_single_token(model, FEMALE_NAMES)
    rejects = ([{"item": w, "reason": "male name not single-token"} for w in rej_m] +
               [{"item": w, "reason": "female name not single-token"} for w in rej_f])

    combos = [(m1, f1, m2) for m1, m2 in itertools.permutations(males, 2)
              for f1 in females]
    rng.shuffle(combos)

    rows: list[dict] = []
    idx = 0
    seen = set()
    for m1, f1, m2 in combos:
        if len(rows) // 2 >= n_items:
            break
        if (m1, m2) in seen:
            continue
        seen.add((m1, m2))

        # slot = first name.  male -> m1 is an available referent (supports m1);
        #                     female -> m1 is excluded (supports m2)
        conflict = TEMPLATE_A.format(n1=m1, n2=m2)
        control = TEMPLATE_A.format(n1=f1, n2=m2)

        item_id = f"A_{idx:03d}"
        try:
            rows.extend(build_minimal_pair(
                model, item_id, "A", "A1", "slot_on_first",
                conflict, control, m1, m2,
            ))
            # stash the female counterpart for the precondition probe
            for r in rows[-2:]:
                r["counterbalance"] = f"slot_on_first|{f1}"
            idx += 1
        except PairRejected as e:
            rejects.append({"item": item_id, "reason": str(e), "n1": m1, "n2": m2})

    return pd.DataFrame(rows, columns=SCHEMA), rejects


# ---------------------------------------------------------------------------
# Category B — instruction / directive conflict
# ---------------------------------------------------------------------------
#
#   B1: "Rule one: say {w1}. Rule two: say {w2}. Obeying the rules, I say"
#   B2: pits a stated directive against a world-knowledge prior.
#
# Be honest in the write-up: for GPT-2 Small, B1 is substantially an induction /
# copy task rather than instruction following.  That is exactly why the
# precondition gate is load-bearing and why B2 exists as a stronger variant.

TEMPLATE_B1 = "Rule one: say {w1}. Rule two: say {w2}. Obeying the rules, I say"
TEMPLATE_B2 = ("Question: name a color. Hint one: the answer is {w1}. "
               "Hint two: the answer is {w2}. Answer: the answer is")


def build_category_b_df(model, n_items: int = 24, seed: int = 1) -> tuple[pd.DataFrame, list[dict]]:
    import random
    rng = random.Random(seed)

    words, rejected_w = filter_single_token(model, RULE_WORDS)
    rejects = [{"item": w, "reason": "rule word not single-token"} for w in rejected_w]

    combos = [(a, b) for a, b in itertools.permutations(words, 2)]
    rng.shuffle(combos)

    rows: list[dict] = []
    idx = 0
    for w_first, w_other in combos:
        if len(rows) // 2 >= n_items:
            break
        template_id = "B1" if idx % 2 == 0 else "B2"
        template = TEMPLATE_B1 if template_id == "B1" else TEMPLATE_B2
        cb = "slot_on_second" if idx % 2 == 0 else "slot_on_first"

        if cb == "slot_on_second":
            # slot = w2.  conflict: rules disagree; control: rules agree on w_first
            conflict = template.format(w1=w_first, w2=w_other)
            control = template.format(w1=w_first, w2=w_first)
            a_ans, b_ans = w_other, w_first
        else:
            # slot = w1.  conflict: rules disagree; control: both say w_other
            conflict = template.format(w1=w_first, w2=w_other)
            control = template.format(w1=w_other, w2=w_other)
            a_ans, b_ans = w_first, w_other

        item_id = f"B_{idx:03d}"
        try:
            rows.extend(build_minimal_pair(
                model, item_id, "B", template_id, cb, conflict, control, a_ans, b_ans,
            ))
            idx += 1
        except PairRejected as e:
            rejects.append({"item": item_id, "reason": str(e), "w1": w_first, "w2": w_other})

    return pd.DataFrame(rows, columns=SCHEMA), rejects


# ---------------------------------------------------------------------------
# Category C — factual vs contextual override
# ---------------------------------------------------------------------------
#
#   C1: "Fact: the capital of {country} is {city}. Question: what is the capital
#        of {country}? Answer: the capital of {country} is"
#   C2: the landmark framing.
#
# This deliberately reproduces Ortu et al. (2024)'s design, which anchors
# Category C to a published GPT-2 Small head set and gives the whole pipeline a
# free end-to-end validation check.

TEMPLATE_C1 = ("Fact: the capital of {entity} is {city}. "
               "Question: what is the capital of {entity}? "
               "Answer: the capital of {entity} is")
TEMPLATE_C2 = ("Fact: the {entity} is in {city}. "
               "Question: where is the {entity}? "
               "Answer: the {entity} is in")


def build_category_c_df(model, n_items: int = 30, seed: int = 2) -> tuple[pd.DataFrame, list[dict]]:
    import random

    from circuit_conflict.utils import is_single_token
    rng = random.Random(seed)

    rejects: list[dict] = []
    cc = [(e, c) for e, c in COUNTRY_CAPITAL if is_single_token(model, c)]
    lc = [(e, c) for e, c in LANDMARK_CITY if is_single_token(model, c)]
    for e, c in COUNTRY_CAPITAL + LANDMARK_CITY:
        if not is_single_token(model, c):
            rejects.append({"item": e, "reason": f"city {c!r} not single-token"})

    rows: list[dict] = []
    idx = 0
    specs = [("C1", TEMPLATE_C1, cc), ("C2", TEMPLATE_C2, lc)]

    for template_id, template, table in specs:
        cities = [c for _, c in table]
        order = list(table)
        rng.shuffle(order)
        for entity, true_city in order:
            if len(rows) // 2 >= n_items:
                break
            # false filler drawn from the SAME distribution, so filler frequency
            # is controlled rather than confounded
            alternatives = [c for c in cities if c != true_city]
            if not alternatives:
                rejects.append({"item": entity, "reason": "no alternative city"})
                continue
            false_city = rng.choice(alternatives)

            conflict = template.format(entity=entity, city=false_city)
            control = template.format(entity=entity, city=true_city)

            item_id = f"C_{idx:03d}"
            try:
                rows.extend(build_minimal_pair(
                    model, item_id, "C", template_id, "slot_on_context",
                    conflict, control, false_city, true_city,
                ))
                idx += 1
            except PairRejected as e:
                rejects.append({"item": item_id, "reason": str(e), "entity": entity})

    return pd.DataFrame(rows, columns=SCHEMA), rejects


# ---------------------------------------------------------------------------
# Preconditions — the ONE legitimate use of the model at dataset time
# ---------------------------------------------------------------------------
#
# Non-circular because (a) it is measured on a prompt DIFFERENT from both
# experimental arms, (b) it gates ELIGIBILITY not the outcome label, and (c) the
# dependent variable -- which channel wins under conflict -- remains free to vary
# in either direction.
#
# Uniform admission rule across all three categories:
#   an item is admitted only if the model's preference REVERSES when the single
#   disambiguating token is swapped, measured on a probe prompt distinct from
#   both experimental arms.
# The reversal requirement is what rules out "answer_A is simply the more
# frequent word", which is the failure mode Category B is most exposed to.

PROBE_B1 = "Rule: say {w}. Obeying the rule, I say"
PROBE_C1 = "Question: what is the capital of {entity}? Answer: the capital of {entity} is"
PROBE_C2 = "Question: where is the {entity}? Answer: the {entity} is in"

MIN_MARGIN = 0.6931471805599453  # ln 2 — the licensed answer at least 2x as likely


@torch.no_grad()
def _ld_on(model, text: str, tok_A: int, tok_B: int) -> float:
    from circuit_conflict.utils import logit_diff
    return logit_diff(model(model.to_tokens(text)), tok_A, tok_B)


@torch.no_grad()
def run_preconditions(model, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Fill `passes_precondition` / `precondition_margin` per item.

    Criterion (all categories): the preference must REVERSE between the two
    probe prompts, and the mean |margin| must exceed ln 2.
    """
    df = df.copy()
    results: dict[str, tuple[bool, float]] = {}

    for item_id, grp in df.groupby("item_id"):
        conf = grp[grp.arm == "conflict"].iloc[0]
        cat = conf["category"]
        tok_A, tok_B = int(conf["token_A"]), int(conf["token_B"])
        a, b = conf["answer_A"], conf["answer_B"]

        try:
            if cat == "B":
                # probe each rule in isolation; preference must follow the rule
                d_a = _ld_on(model, PROBE_B1.format(w=a), tok_A, tok_B)
                d_b = _ld_on(model, PROBE_B1.format(w=b), tok_A, tok_B)
            elif cat == "C":
                # context-free knowledge probe: does the model hold the belief at all?
                # (only one direction is meaningful here, so the "reversal" is
                #  against the context-free prior itself)
                probe = PROBE_C1 if conf["template_id"] == "C1" else PROBE_C2
                entity = _entity_from_prompt(conf["prompt_text"], conf["template_id"])
                d_b = _ld_on(model, probe.format(entity=entity), tok_B, tok_A)
                d_a = -d_b
            else:  # A — gender-agreement reversal on a probe distinct from both arms
                m1, m2 = a, b
                f1 = str(conf["counterbalance"]).split("|")[-1]
                # both male -> "He" is ambiguous but should not disfavour m1
                d_a = _ld_on(model, PROBE_A.format(n1=m1, n2=m2), tok_A, tok_B)
                # first name female -> "He" must resolve to m2
                d_b = _ld_on(model, PROBE_A.format(n1=f1, n2=m2), tok_A, tok_B)
        except Exception as e:  # noqa: BLE001 — record, never silently skip
            results[item_id] = (False, float("nan"))
            if verbose:
                print(f"  {item_id}: precondition error: {e}")
            continue

        if cat in ("A", "B"):
            # true reversal: preference must flip when the disambiguator flips
            reversed_ok = (d_a > 0) and (d_b < 0)
        else:
            # C is a knowledge check, not a reversal: the conflict only EXISTS if
            # the model holds the competing parametric belief in the first place.
            reversed_ok = (d_b > 0)
        margin = (abs(d_a) + abs(d_b)) / 2.0
        results[item_id] = (bool(reversed_ok and margin >= MIN_MARGIN), float(margin))

    df["passes_precondition"] = df.item_id.map(lambda i: results.get(i, (False, float("nan")))[0])
    df["precondition_margin"] = df.item_id.map(lambda i: results.get(i, (False, float("nan")))[1])
    return df


def _entity_from_prompt(text: str, template_id: str) -> str:
    """Recover the entity string from a built Category C prompt."""
    if template_id == "C1":
        return text.split("the capital of ")[1].split(" is ")[0]
    return text.split("Fact: the ")[1].split(" is in ")[0]


def precondition_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category pass rate — the pre-registered Category B gate reads this."""
    items = df[df.arm == "conflict"]
    return (items.groupby("category")
                 .agg(n_items=("item_id", "count"),
                      n_pass=("passes_precondition", "sum"),
                      pass_rate=("passes_precondition", "mean"),
                      median_margin=("precondition_margin", "median"))
                 .reset_index())
