"""Offline checks for eval sets, generation post-processing and blinded pair files."""

from __future__ import annotations

import json

import pytest

from src.alignment.eval_sets import load_eval_set
from src.alignment.generate import clean_output
from src.alignment.pairs import build_pairs


def gen_rows(generator, n=40, prefix="p"):
    return [{"id": f"{prefix}{i}", "instruction": f"q{i}", "output": f"{generator}-{i}",
             "generator": generator, "dataset": "toy", "category": "c"} for i in range(n)]


@pytest.mark.parametrize("name", ["alpaca", "hh"])
def test_eval_sets_are_fixed_and_unique(name):
    rows = load_eval_set(name)
    assert len(rows) == 300
    assert len({row["id"] for row in rows}) == 300
    assert all(row["instruction"].strip() for row in rows)


def test_alpaca_subset_matches_seed_42_recipe():
    import random
    rows = load_eval_set("alpaca")
    assert [row["alpaca_eval_index"] for row in rows] == random.Random(42).sample(range(805), 300)


def test_clean_output_cuts_run_on_template_turns():
    assert clean_output("Answer.\n\n### Instruction:\nnext q") == ("Answer.", True)
    assert clean_output("  plain answer \n") == ("plain answer", False)


def test_pairs_hide_identity_and_map_back_correctly():
    pairs = build_pairs(gen_rows("dpo"), gen_rows("sft"))
    for pair in pairs:
        visible = json.dumps({k: v for k, v in pair.items() if k != "hidden"})
        assert "dpo" not in visible.replace("dpo-", "") and "sft" not in visible.replace("sft-", "")
        hidden = pair["hidden"]
        assert {hidden["model_a"], hidden["model_b"]} == {"dpo", "sft"}
        assert pair["response_a"].startswith(hidden["model_a"])
        assert pair["response_b"].startswith(hidden["model_b"])


def test_pair_order_is_randomized_deterministic_and_order_independent():
    pairs = build_pairs(gen_rows("dpo"), gen_rows("sft"))
    in_a = [p["hidden"]["model_a"] == "dpo" for p in pairs]
    assert 0 < sum(in_a) < len(in_a)
    again = build_pairs(gen_rows("dpo")[::-1], gen_rows("sft"))
    assert {p["pair_id"]: p["hidden"]["model_a"] for p in again} == {p["pair_id"]: p["hidden"]["model_a"] for p in pairs}


def test_pairs_reject_mismatched_files():
    with pytest.raises(ValueError):
        build_pairs(gen_rows("dpo", n=5), gen_rows("sft", n=4))
    other = gen_rows("sft", n=5)
    other[2]["instruction"] = "different"
    with pytest.raises(ValueError):
        build_pairs(gen_rows("dpo", n=5), other)
