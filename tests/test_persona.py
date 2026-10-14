"""Offline checks for the persona LoRA-DPO data helpers and eval set."""

from __future__ import annotations

import re
from collections import Counter

from src.alignment.eval_sets import load_eval_set
from src.alignment.persona import load_persona_rows, split_rows, style_lexicon, style_stats, to_conversational


def test_split_is_deterministic_disjoint_and_complete():
    rows = load_persona_rows()
    train, val = split_rows(rows, val_fraction=0.1, seed=42)
    again_train, again_val = split_rows(rows, val_fraction=0.1, seed=42)
    assert [r["index"] for r in val] == [r["index"] for r in again_val]
    assert {r["index"] for r in train}.isdisjoint(r["index"] for r in val)
    assert len(train) + len(val) == len(rows) and len(val) == round(len(rows) * 0.1)


def test_conversational_format_matches_trl():
    row = to_conversational({"prompt": "p", "chosen": "c", "rejected": "r"})
    assert row == {"prompt": [{"role": "user", "content": "p"}],
                   "chosen": [{"role": "assistant", "content": "c"}],
                   "rejected": [{"role": "assistant", "content": "r"}]}


def test_style_metric_separates_held_out_chosen_from_rejected():
    train, val = split_rows(load_persona_rows())
    lexicon = style_lexicon(train)
    chosen = style_stats([r["chosen"] for r in val], lexicon)
    rejected = style_stats([r["rejected"] for r in val], lexicon)
    assert chosen["responses_with_marker"] > 0.8 > 0.2 > rejected["responses_with_marker"]


def test_persona_eval_set_is_balanced_and_unseen_in_training():
    rows = load_eval_set("persona")
    assert Counter(r["category"] for r in rows) == {
        "cooking": 10, "coding": 10, "general_help": 10, "safety_sensitive": 10}
    train_prompts = {re.sub(r"\W+", " ", r["prompt"].lower()).strip() for r in load_persona_rows()}
    assert not any(re.sub(r"\W+", " ", r["instruction"].lower()).strip() in train_prompts for r in rows)
