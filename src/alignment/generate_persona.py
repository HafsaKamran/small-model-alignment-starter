"""Generate persona-eval responses from SmolLM2-Instruct, with or without the LoRA adapter.

Same output format and resume behavior as generate.py, but prompts go through the
model's own chat template (the format TRL trained the adapter on) instead of the
"### Instruction:" template used for the Qwen tracks.
"""

import argparse
import json
import os

import torch

from .eval_sets import load_eval_set
from .generate import load_existing, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--adapter_path", default=None, help="LoRA adapter dir; omit for the plain instruct model")
    p.add_argument("--generator", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity checks)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_eval_set("persona")[: args.limit]
    existing = {row["id"]: row for row in load_existing(args.output_path)}
    todo = [row for row in prompts if row["id"] not in existing]
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("output_path", "device", "batch_size", "limit")}
    with open(args.output_path.removesuffix(".json") + ".config.json", "w", encoding="utf-8") as f:
        json.dump(dict(config, do_sample=False, prompt_format="tokenizer chat template"), f, indent=1)

    if todo:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
        if args.adapter_path:
            from peft import PeftModel
            # Merge the adapter into the weights: same outputs, faster generation.
            model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
        model.to(device=args.device, dtype=getattr(torch, args.dtype)).eval()

        for b in range(0, len(todo), args.batch_size):
            batch = todo[b : b + args.batch_size]
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": row["instruction"]}],
                                                   tokenize=False, add_generation_prompt=True) for row in batch]
            # The chat template already includes any special tokens, so don't add them twice.
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                     repetition_penalty=args.repetition_penalty,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            for row, tokens in zip(batch, out[:, enc["input_ids"].shape[1]:]):
                hit_eos = bool((tokens == tokenizer.eos_token_id).any())
                n_tokens = int((tokens == tokenizer.eos_token_id).int().argmax()) if hit_eos else len(tokens)
                existing[row["id"]] = {**row, "dataset": "persona_style_eval",
                                       "output": tokenizer.decode(tokens[:n_tokens], skip_special_tokens=True).strip(),
                                       "generator": args.generator, "n_new_tokens": n_tokens,
                                       "stopped_at_eos": hit_eos}
            write_json([existing[row["id"]] for row in prompts if row["id"] in existing], args.output_path)
            print(f"[{args.generator}] {min(b + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    rows = [existing[row["id"]] for row in prompts]
    print(f"[{args.generator}] done: {len(rows)} responses, stopped at EOS {sum(r['stopped_at_eos'] for r in rows)}",
          flush=True)


if __name__ == "__main__":
    main()
