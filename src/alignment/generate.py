"""Generate responses from one checkpoint on one fixed eval set.

Writes a JSON list of {"id", "instruction", "output", "generator", "dataset", ...}
(manual section 8), rewritten after every batch so an interrupted run resumes
where it stopped. A sidecar <output>.config.json records the decoding settings.

Every model must be generated with the same decoding settings so the judge
compares models, not decoding policies.
"""

import argparse
import json
import os
import time

import torch

from .eval_sets import load_eval_set
from .formatting import PROMPT_TEMPLATE, format_prompt

# Where the next turn of the template would start. A model that keeps going past
# its answer (mostly the base model) gets cut here; trained models stop at EOS first.
STOP_MARKER = "### Instruction:"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True, help="checkpoint dir or hub id, e.g. Qwen/Qwen2.5-0.5B")
    p.add_argument("--generator", required=True, help="name stored in every row, e.g. qwen2.5-0.5b-sft")
    p.add_argument("--eval_set", required=True, choices=["alpaca", "hh"])
    p.add_argument("--output_path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity checks)")
    return p.parse_args(argv)


def clean_output(text):
    """Cut a run-on continuation at the next template turn. Returns (text, was_cut)."""
    index = text.find(STOP_MARKER)
    if index == -1:
        return text.strip(), False
    return text[:index].strip(), True


def load_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)  # atomic, so a disconnect never leaves a half-written file


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_eval_set(args.eval_set)[: args.limit]
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    existing = {row["id"]: row for row in load_existing(args.output_path)}
    todo = [row for row in prompts if row["id"] not in existing]
    print(f"[{args.generator}/{args.eval_set}] {len(prompts)} prompts, {len(existing)} already done, {len(todo)} to go",
          flush=True)

    config = {k: v for k, v in vars(args).items() if k not in ("output_path", "device", "batch_size", "limit")}
    config.update(do_sample=False, prompt_template=PROMPT_TEMPLATE, stop_marker=STOP_MARKER)
    config_path = args.output_path.removesuffix(".json") + ".config.json"
    if existing and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            previous = json.load(f)
        changed = {k for k in config if previous.get(k) != config[k]}
        if changed:  # resuming would mix two decoding policies in one file
            raise SystemExit(f"{args.output_path} was generated with different settings ({sorted(changed)}); "
                             f"delete it or use another --output_path")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)

    if todo:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = "left"  # generation appends on the right, so pad on the left
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # torch_dtype (not dtype) works across the whole transformers>=4.43 range in requirements.txt
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=getattr(torch, args.dtype))
        model.to(args.device).eval()

        # Longest prompts first: similar lengths share a batch (less padding), and an
        # out-of-memory error shows up on the first batch rather than the last.
        todo.sort(key=lambda row: -len(tokenizer.encode(format_prompt(row["instruction"]))))
        start = time.time()
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b : b + args.batch_size]
            enc = tokenizer([format_prompt(row["instruction"]) for row in batch], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(args.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            for row, tokens in zip(batch, new_tokens):
                hit_eos = bool((tokens == tokenizer.eos_token_id).any())
                n_tokens = int((tokens == tokenizer.eos_token_id).int().argmax()) if hit_eos else len(tokens)
                text, cut = clean_output(tokenizer.decode(tokens[:n_tokens], skip_special_tokens=True))
                existing[row["id"]] = {
                    **{k: v for k, v in row.items()},
                    "output": text,
                    "generator": args.generator,
                    "n_new_tokens": n_tokens,
                    "stopped_at_eos": hit_eos,
                    "cut_at_stop_marker": cut,
                }
            # Keep eval-set order in the file, whatever order we generated in.
            write_json([existing[row["id"]] for row in prompts if row["id"] in existing], args.output_path)
            done = min(b + args.batch_size, len(todo))
            print(f"[{args.generator}/{args.eval_set}] {done}/{len(todo)}  {(time.time() - start) / 60:.1f} min",
                  flush=True)

    rows = [existing[row["id"]] for row in prompts]
    n = len(rows)
    print(f"[{args.generator}/{args.eval_set}] done: stopped at EOS {sum(r['stopped_at_eos'] for r in rows)}/{n}, "
          f"cut at stop marker {sum(r['cut_at_stop_marker'] for r in rows)}/{n}, "
          f"empty {sum(not r['output'] for r in rows)}/{n}, "
          f"mean new tokens {sum(r['n_new_tokens'] for r in rows) / max(n, 1):.0f}", flush=True)


if __name__ == "__main__":
    main()
