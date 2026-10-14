"""Offline checks for the pairwise judge: prompt blinding, parsing, A/B mapping,
retries, resume, failure logging, position-swap combination and summary math."""

from __future__ import annotations

import json

import pytest

from src.alignment import judging
from src.alignment.judging import (MockJudge, ParseError, build_prompt, combine_orders, load_jsonl, outcome,
                                   parse_response, wilson)
from src.alignment.pairs import build_pairs


def write_pairs(tmp_path, n=12, dataset="toy"):
    """Candidate "dpo" always writes the longer answer, so MockJudge always prefers it."""
    gen = lambda name, extra: [{"id": f"p{i}", "instruction": f"question {i}", "output": f"answer {i}" + extra,
                                "generator": name, "dataset": dataset, "category": "c1" if i % 2 else "c2"}
                               for i in range(n)]
    pairs = build_pairs(gen("dpo", " with more detail"), gen("sft", ""))
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    return path, pairs


def args_for(tmp_path, pairs_path, *extra):
    return ["--pairs_path", str(pairs_path), "--output_path", str(tmp_path / "judged.jsonl"),
            "--backend", "mock", "--sleep_seconds", "0", "--batch_size", "5", *extra]


def test_prompt_never_reveals_model_identity(tmp_path):
    _, pairs = write_pairs(tmp_path)
    prompt = build_prompt(pairs[:5], "general")
    assert "dpo" not in prompt and "sft" not in prompt and "hidden" not in prompt
    assert prompt.count("### Item") == 5 and "exactly 5 objects" in prompt


def test_parse_handles_code_fences_aliases_and_bad_items():
    reply = '```json\n[{"item": 1, "reason": "x", "winner": "Response A"}, {"item": 2, "winner": "b."},' \
            ' {"item": 3, "winner": "maybe"}, {"item": 9, "winner": "A"}]\n```'
    parsed = parse_response(reply, 3, ["winner"])
    assert parsed[1]["winner"] == "A" and parsed[2]["winner"] == "B" and parsed[3]["winner"] is None
    assert 9 not in parsed
    with pytest.raises(ParseError):
        parse_response("I think A is better.", 1, ["winner"])


def test_outcome_maps_positions_back_to_models():
    pair = {"hidden": {"model_a": "sft", "model_b": "dpo", "candidate": "dpo"}}
    assert (outcome("A", pair), outcome("B", pair), outcome("tie", pair), outcome(None, pair)) == \
           ("loss", "win", "tie", "unknown")


def test_full_run_maps_back_correctly_and_writes_all_files(tmp_path):
    pairs_path, pairs = write_pairs(tmp_path)
    summary = judging.main(args_for(tmp_path, pairs_path))
    rows = load_jsonl(tmp_path / "judged.jsonl")
    assert len(rows) == len(pairs)
    # MockJudge prefers the longer answer, which is always the candidate's, whichever position it is in
    assert all(r["outcomes"]["winner"] == "win" for r in rows)
    assert {r["verdicts"]["winner"] for r in rows} == {"A", "B"}
    o = summary["fields"]["winner"]["overall"]
    assert (o["wins"], o["losses"], o["failures"], o["win_rate_no_ties"]) == (12, 0, 0, 1.0)
    assert set(summary["fields"]["winner"]["by_category"]) == {"c1", "c2"}
    assert (tmp_path / "judged.summary.json").exists() and (tmp_path / "judged.raw.jsonl").exists()


def test_resume_skips_done_pairs_and_refuses_to_overwrite(tmp_path):
    pairs_path, _ = write_pairs(tmp_path)
    judging.main(args_for(tmp_path, pairs_path))
    judge = MockJudge()
    judging.main(args_for(tmp_path, pairs_path, "--reuse_judgments"), judge=judge)
    assert judge.calls == 0
    with pytest.raises(SystemExit):
        judging.main(args_for(tmp_path, pairs_path))


def test_transient_errors_are_retried(tmp_path):
    pairs_path, pairs = write_pairs(tmp_path, n=5)
    judge = MockJudge(fail_first=2)
    judging.main(args_for(tmp_path, pairs_path, "--max_retries", "3"), judge=judge)
    assert judge.calls == 3 and len(load_jsonl(tmp_path / "judged.jsonl")) == 5


def test_persistent_failures_are_logged_and_retried_on_next_run(tmp_path):
    pairs_path, _ = write_pairs(tmp_path, n=5)
    summary = judging.main(args_for(tmp_path, pairs_path, "--max_retries", "2"), judge=MockJudge(fail_first=99))
    assert summary["fields"]["winner"]["overall"]["failures"] == 5
    assert len(load_jsonl(tmp_path / "judged.failures.jsonl")) == 1
    summary = judging.main(args_for(tmp_path, pairs_path, "--reuse_judgments"), judge=MockJudge())
    assert summary["fields"]["winner"]["overall"]["failures"] == 0


def test_partial_reply_marks_missing_items_unknown(tmp_path):
    pairs_path, _ = write_pairs(tmp_path, n=3)
    reply = json.dumps([{"item": 1, "reason": "r", "winner": "tie"}])
    summary = judging.main(args_for(tmp_path, pairs_path, "--max_retries", "1"), judge=lambda prompt: reply)
    o = summary["fields"]["winner"]["overall"]
    assert (o["ties"], o["unknown"], o["failures"]) == (1, 2, 0)


def test_both_orders_detects_position_bias(tmp_path):
    pairs_path, pairs = write_pairs(tmp_path, n=10)
    always_a = lambda prompt: json.dumps([{"item": i, "reason": "", "winner": "A"} for i in range(1, 6)])
    summary = judging.main(args_for(tmp_path, pairs_path, "--both_orders"), judge=always_a)
    r = summary["fields"]["winner"]
    # A judge that always says "A" has no real preference: every pair flips, so all become ties
    assert r["position_consistency"] == 0.0
    assert r["overall"]["ties"] == 10 and r["verdict_position_counts"] == {"A": 20}


def test_combine_orders_and_wilson():
    assert combine_orders("win", "win") == "win"
    assert combine_orders("win", "loss") == "tie"
    assert combine_orders("unknown", "loss") == "loss"
    lo, hi = wilson(60, 100)
    assert 0.49 < lo < 0.6 < hi < 0.7 and wilson(0, 0) is None


def test_rubric_is_chosen_from_dataset(tmp_path):
    pairs_path, _ = write_pairs(tmp_path, n=5, dataset="persona_style_eval")
    summary = judging.main(args_for(tmp_path, pairs_path))
    assert summary["rubric"] == "persona" and set(summary["fields"]) == {"style", "helpfulness", "safety"}


def test_kaggle_backend_requires_polite_sleep(tmp_path):
    pairs_path, _ = write_pairs(tmp_path, n=2)
    with pytest.raises(SystemExit):
        judging.parse_args(["--pairs_path", str(pairs_path), "--output_path", str(tmp_path / "x.jsonl"),
                            "--sleep_seconds", "1"])
