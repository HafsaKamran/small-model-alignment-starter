"""Preference data loading: raw datasets -> formatted (prompt, chosen, rejected) rows.

Both tracks are reduced to single-turn pairs and formatted with the same template
as SFT, so the DPO policy sees prompts exactly as it did during SFT.

Train pairs come from each dataset's training split and validation pairs from its
separate test split, so the two never overlap.
"""

from .formatting import format_prompt

HUMAN = "\n\nHuman:"
ASSISTANT = "\n\nAssistant:"

TRACKS = {
    "hh": {"dataset": "Anthropic/hh-rlhf", "train_split": "train", "val_split": "test"},
    "ultrafeedback": {"dataset": "HuggingFaceH4/ultrafeedback_binarized",
                      "train_split": "train_prefs", "val_split": "test_prefs"},
}


def split_hh_transcript(text):
    """'\\n\\nHuman: X\\n\\nAssistant: Y' -> (X, Y). Returns None for multi-turn or malformed text."""
    text = "\n\n" + text.lstrip()
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    human_part, response = text.split(ASSISTANT)
    if not human_part.startswith(HUMAN):
        return None
    instruction = human_part[len(HUMAN):].strip()
    return instruction, response.strip()


def parse_hh(example):
    """HH-RLHF stores two full transcripts that share everything except the last reply."""
    chosen = split_hh_transcript(example["chosen"])
    rejected = split_hh_transcript(example["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    return chosen[0], chosen[1], rejected[1]


def parse_ultrafeedback(example):
    """UltraFeedback-binarized stores chosen/rejected as [user, assistant] message lists."""
    if example["score_chosen"] == example["score_rejected"]:
        return None  # a tie carries no preference signal
    chosen, rejected = example["chosen"], example["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if chosen[0]["role"] != "user" or chosen[1]["role"] != "assistant" or rejected[1]["role"] != "assistant":
        return None
    return example["prompt"].strip(), chosen[1]["content"].strip(), rejected[1]["content"].strip()


PARSERS = {"hh": parse_hh, "ultrafeedback": parse_ultrafeedback}


def collect_pairs(dataset, parser, tokenizer, n, max_prompt_tokens, id_prefix):
    rows = []
    for index, example in enumerate(dataset):
        parsed = parser(example)
        if parsed is None:
            continue
        instruction, chosen, rejected = parsed
        if not instruction or not chosen or not rejected or chosen == rejected:
            continue
        prompt = format_prompt(instruction)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_tokens:
            continue
        rows.append({
            "id": example.get("prompt_id") or f"{id_prefix}-{index}",
            "instruction": instruction,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise ValueError(f"Only found {len(rows)} usable pairs in {id_prefix}, needed {n}")
    return rows


def load_preference_splits(track, tokenizer, n_train=15_000, n_val=1_000, seed=42, max_prompt_tokens=256):
    """Return (train_rows, val_rows) of single-turn, template-formatted preference pairs."""
    from datasets import load_dataset

    spec = TRACKS[track]
    parser = PARSERS[track]
    train_ds = load_dataset(spec["dataset"], split=spec["train_split"]).shuffle(seed=seed)
    val_ds = load_dataset(spec["dataset"], split=spec["val_split"]).shuffle(seed=seed)
    train_rows = collect_pairs(train_ds, parser, tokenizer, n_train, max_prompt_tokens, f"{track}-train")
    val_rows = collect_pairs(val_ds, parser, tokenizer, n_val, max_prompt_tokens, f"{track}-val")
    return train_rows, val_rows
