"""Persona (Ramsay-style) LoRA-DPO data helpers and a simple style metric.

Unlike the from-scratch tracks, this task uses the model's own chat template:
SmolLM2-360M-Instruct is already chat-tuned, and TRL's DPOTrainer applies the
template itself when rows are role/content message lists.
"""

import json
import math
import os
import random
import re
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERSONA_DATA = os.path.join(REPO_ROOT, "data", "persona_lora", "ramsay_style_preferences.json")


def load_persona_rows(path=PERSONA_DATA):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_rows(rows, val_fraction=0.1, seed=42):
    """Shuffle with a fixed seed and hold out val_fraction of the pairs. Returns (train, val)."""
    rows = [dict(row, index=i) for i, row in enumerate(rows)]
    random.Random(seed).shuffle(rows)
    n_val = max(1, round(len(rows) * val_fraction))
    return rows[n_val:], rows[:n_val]


def to_conversational(row):
    """{prompt, chosen, rejected} strings -> TRL's conversational preference format."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def words(text):
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


# Function words that differ between the styles only because persona replies say
# "you" a lot; they would count any second-person answer as persona-styled.
COMMON_WORDS = {"you", "your", "you're", "you've", "yourself", "what", "that", "this", "with", "have", "it's",
                "don't", "because", "anyone", "entire", "made", "needs", "turns", "turned", "second", "straight",
                "into", "just", "like", "they", "their", "there", "from", "will", "would", "need", "make"}


def style_lexicon(rows, k=15, min_count=3):
    """Words most characteristic of chosen (persona) vs rejected (neutral) responses.

    Ranked by log-odds with add-one smoothing, so the marker list comes from the
    training data instead of being hand-picked. Short and common words are dropped.
    """
    chosen = Counter(w for row in rows for w in words(row["chosen"]))
    rejected = Counter(w for row in rows for w in words(row["rejected"]))
    n_c, n_r, vocab = sum(chosen.values()), sum(rejected.values()), len(set(chosen) | set(rejected))
    score = {w: math.log((chosen[w] + 1) / (n_c + vocab)) - math.log((rejected[w] + 1) / (n_r + vocab))
             for w in chosen if chosen[w] >= min_count and len(w) >= 4 and w not in COMMON_WORDS}
    return sorted(score, key=score.get, reverse=True)[:k]


def style_stats(texts, lexicon):
    """Share of responses using any persona marker, markers per 100 words, and '!' per response."""
    lexicon = set(lexicon)
    n = max(len(texts), 1)
    total_words = sum(len(words(t)) for t in texts) or 1
    hits = [sum(w in lexicon for w in words(t)) for t in texts]
    return {
        "responses_with_marker": sum(h > 0 for h in hits) / n,
        "markers_per_100_words": 100 * sum(hits) / total_words,
        "exclamations_per_response": sum(t.count("!") for t in texts) / n,
        "mean_words": total_words / n,
    }
