"""Build blinded pairwise comparison files from two generation files.

Each JSONL row shows the judge only "instruction", "response_a" and "response_b"
(plus the eval set's category / expected_behavior rubric, which says nothing about
which model wrote what). Which model sits in position A is decided per prompt by a
seeded coin flip and kept under "hidden", to be mapped back only after judging.

    python -m src.alignment.pairs --model_path outputs/generations/alpaca/dpo.json \
        --baseline_path outputs/generations/alpaca/sft.json --output_path outputs/eval/x_pairs.jsonl
"""

import argparse
import json
import os
import random

JUDGE_VISIBLE_EXTRAS = ("category", "expected_behavior")


def build_pairs(model_rows, baseline_rows, seed=42):
    """model_rows / baseline_rows: generation rows for the same prompts. Returns pair rows."""
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    if len(model_rows) != len(baseline_rows) or set(baseline_by_id) != {row["id"] for row in model_rows}:
        raise ValueError("the two generation files must cover exactly the same prompt ids")

    pairs = []
    for model_row in model_rows:
        base_row = baseline_by_id[model_row["id"]]
        if model_row["instruction"] != base_row["instruction"]:
            raise ValueError(f"instruction mismatch for id {model_row['id']}")
        # Seeded by id, so a prompt's A/B order doesn't depend on file order.
        model_is_a = random.Random(f"{seed}-{model_row['id']}").random() < 0.5
        a, b = (model_row, base_row) if model_is_a else (base_row, model_row)
        pair = {
            "pair_id": model_row["id"],
            "instruction": model_row["instruction"],
            "response_a": a["output"],
            "response_b": b["output"],
            **{k: model_row[k] for k in JUDGE_VISIBLE_EXTRAS if k in model_row},
            "hidden": {
                "model_a": a["generator"],
                "model_b": b["generator"],
                "candidate": model_row["generator"],
                "baseline": base_row["generator"],
                "dataset": model_row.get("dataset"),
            },
        }
        pairs.append(pair)
    return pairs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True, help="generations of the model being tested (e.g. DPO)")
    p.add_argument("--baseline_path", required=True, help="generations of the baseline (e.g. SFT)")
    p.add_argument("--output_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    with open(args.model_path, encoding="utf-8") as f:
        model_rows = json.load(f)
    with open(args.baseline_path, encoding="utf-8") as f:
        baseline_rows = json.load(f)
    pairs = build_pairs(model_rows, baseline_rows, args.seed)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    n_a = sum(pair["hidden"]["model_a"] == pair["hidden"]["candidate"] for pair in pairs)
    print(f"wrote {len(pairs)} pairs to {args.output_path} ({pairs[0]['hidden']['candidate']} vs "
          f"{pairs[0]['hidden']['baseline']}; candidate in position A {n_a}/{len(pairs)})")


if __name__ == "__main__":
    main()
