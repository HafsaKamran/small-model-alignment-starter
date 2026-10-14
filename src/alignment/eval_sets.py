"""Fixed evaluation prompt sets. Every model is compared on exactly these prompts.

- "alpaca": 300 prompts sampled from AlpacaEval's 805 with random.Random(42), saved once
  to eval_sets/alpaca_eval_300_seed42.json (built by `python -m src.alignment.eval_sets`).
- "hh": the synthetic helpful/harmless set provided with the assignment.
"""

import json
import os
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_SETS = {
    "alpaca": os.path.join(REPO_ROOT, "eval_sets", "alpaca_eval_300_seed42.json"),
    "hh": os.path.join(REPO_ROOT, "eval_sets", "hh_rlhf_eval_300_seed42.json"),
    # 40 hand-written prompts (10 each: cooking, coding, general_help, safety_sensitive) for the
    # persona LoRA track, checked to share no topic with data/persona_lora/ training prompts.
    "persona": os.path.join(REPO_ROOT, "eval_sets", "persona_style_eval_40.json"),
}


def build_alpaca_eval_subset(n=300, seed=42):
    """Sample n AlpacaEval prompts. Rows keep their sampled order and original index."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        full = json.load(f)
    indices = random.Random(seed).sample(range(len(full)), n)
    return [{
        "id": f"alpaca-eval-{seed}-{i:04d}",
        "instruction": full[index]["instruction"],
        "dataset": "alpaca_eval",
        "source": full[index]["dataset"],   # selfinstruct, oasst, koala, helpful_base or vicuna
        "alpaca_eval_index": index,
    } for i, index in enumerate(indices)]


def load_eval_set(name):
    with open(EVAL_SETS[name], encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    rows = build_alpaca_eval_subset()
    with open(EVAL_SETS["alpaca"], "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} prompts to {EVAL_SETS['alpaca']}")
