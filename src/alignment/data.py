"""Dataset loading: raw UltraChat conversations -> formatted (prompt, response) rows."""

import json
import os

from .formatting import format_prompt


def first_turn(messages):
    """Return (instruction, response) from the first user->assistant exchange, or None."""
    if len(messages) < 2:
        return None
    user, assistant = messages[0], messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    instruction = (user.get("content") or "").strip()
    response = (assistant.get("content") or "").strip()
    if not instruction or not response:
        return None
    return instruction, response


def load_sft_splits(
    tokenizer,
    n_train=30_000,
    n_val=1_000,
    seed=42,
    max_prompt_tokens=384,
    dataset_name="HuggingFaceH4/ultrachat_200k",
    split="train_sft",
):
    """Shuffle UltraChat with a fixed seed and return (train_rows, val_rows).

    Each row is {"id", "instruction", "prompt", "response"}, where "prompt" is
    already formatted with the template. Multi-turn conversations are reduced to
    their first exchange. Prompts longer than max_prompt_tokens are skipped,
    since they would leave little or no room for response tokens at max_length.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split).shuffle(seed=seed)
    needed = n_train + n_val
    rows = []
    for index, example in enumerate(dataset):
        pair = first_turn(example["messages"])
        if pair is None:
            continue
        instruction, response = pair
        prompt = format_prompt(instruction)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_tokens:
            continue
        rows.append({
            "id": example.get("prompt_id", str(index)),
            "instruction": instruction,
            "prompt": prompt,
            "response": response,
        })
        if len(rows) >= needed:
            break

    if len(rows) < needed:
        raise ValueError(f"Only found {len(rows)} usable examples, needed {needed}")

    val_rows, train_rows = rows[:n_val], rows[n_val:]
    return train_rows, val_rows


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
