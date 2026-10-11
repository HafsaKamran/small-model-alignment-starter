"""Supervised fine-tuning loop for Qwen2.5-0.5B.

Run from the repo root (relative imports need the -m form):

    python -m src.alignment.train_sft --output_dir outputs/sft

Pilot run (a few minutes) before the real one:

    python -m src.alignment.train_sft --output_dir outputs/sft_pilot \
        --n_train 640 --n_val 32 --eval_every 10
"""

import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from .batching import build_sft_batch
from .data import load_sft_splits, save_jsonl
from .losses import compute_response_logprobs, compute_sft_loss


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SFT on single-turn UltraChat")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dataset_name", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--dataset_split", default="train_sft")
    p.add_argument("--output_dir", default="outputs/sft")
    p.add_argument("--n_train", type=int, default=30_000)
    p.add_argument("--n_val", type=int, default=1_000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_tokens", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = no limit)")
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validation runs")
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gradient_checkpointing", action="store_true", help="use if you hit CUDA OOM")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_text(rows):
    """The DataLoader only yields strings; tokenizing happens inside compute_sft_loss."""
    return [r["prompt"] for r in rows], [r["response"] for r in rows]


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def make_optimizer(model, lr, weight_decay):
    """AdamW, with no weight decay on biases and norm weights (1-D params)."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


@torch.no_grad()
def evaluate(model, tokenizer, rows, max_length, batch_size, autocast):
    """Token-mean response NLL over the whole validation set (exact, not a mean of batch means)."""
    model.eval()
    total_logp, total_tokens = 0.0, 0
    for start in range(0, len(rows), batch_size):
        prompts, responses = collate_text(rows[start:start + batch_size])
        with autocast():
            logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
        labels = build_sft_batch(tokenizer, prompts, responses, max_length)["labels"]
        total_logp += logps.sum().item()
        total_tokens += labels[:, 1:].ne(-100).sum().item()
    model.train()
    return -total_logp / max(1, total_tokens)


def log_metrics(path, record):
    """Append one line per call, so a disconnect never loses earlier metrics."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def save_checkpoint(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def train(model, tokenizer, train_rows, val_rows, args):
    """The training loop itself. Separate from main() so it can be tested on a tiny model."""
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    model.to(device)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False  # the KV cache is only useful for generation

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_text, generator=generator)

    micro_per_epoch = len(loader)
    total_micro = micro_per_epoch * args.epochs
    total_steps = math.ceil(total_micro / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
        total_micro = min(total_micro, total_steps * args.grad_accum)
    warmup_steps = max(1, round(args.warmup_ratio * total_steps))

    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)

    print(f"train={len(train_rows)} val={len(val_rows)} micro_batches={total_micro} "
          f"optimizer_steps={total_steps} warmup={warmup_steps} amp={use_amp}", flush=True)

    best_val = float("inf")
    step = 0
    micro = 0
    window_loss, window_micro = 0.0, 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    # Validation loss of the untrained model, as a reference point for the report.
    # "best" below means best among *trained* checkpoints; step 0 is only a baseline.
    start_val = evaluate(model, tokenizer, val_rows, args.max_length, args.eval_batch_size, autocast)
    log_metrics(metrics_path, {"type": "eval", "step": 0, "val_loss": start_val,
                               "val_ppl": math.exp(min(start_val, 50)), "baseline": True})

    done = False
    for epoch in range(args.epochs):
        for prompts, responses in loader:
            # The last accumulation group can be shorter than grad_accum; divide by its real size.
            group_start = (micro // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, total_micro - group_start)

            with autocast():
                loss = compute_sft_loss(model, tokenizer, prompts, responses, args.max_length)
            scaler.scale(loss / group_size).backward()

            window_loss += loss.item()
            window_micro += 1
            micro += 1

            if micro - group_start < group_size:
                continue  # keep accumulating gradients

            # One optimizer step: unscale -> clip -> step -> update scale -> schedule.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                log_metrics(metrics_path, {
                    "type": "train", "step": step, "epoch": epoch,
                    "train_loss": window_loss / window_micro,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "elapsed_min": round((time.time() - start_time) / 60, 2),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if use_amp else 0.0,
                })
                window_loss, window_micro = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate(model, tokenizer, val_rows, args.max_length, args.eval_batch_size, autocast)
                improved = val_loss < best_val
                log_metrics(metrics_path, {"type": "eval", "step": step, "val_loss": val_loss,
                                           "val_ppl": math.exp(min(val_loss, 50)),
                                           "change_vs_start": val_loss - start_val,
                                           "best_so_far": improved})
                if improved:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best"))

            if step >= total_steps:
                done = True
                break
        if done:
            break

    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final"))
    summary = {"start_val_loss": start_val, "best_val_loss": best_val, "optimizer_steps": step,
               "minutes": round((time.time() - start_time) / 60, 2)}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done:", summary, flush=True)
    return summary


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # fp32 weights: required for GradScaler; autocast does the fp16 compute.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_rows, val_rows = load_sft_splits(
        tokenizer, n_train=args.n_train, n_val=args.n_val, seed=args.seed,
        max_prompt_tokens=args.max_prompt_tokens,
        dataset_name=args.dataset_name, split=args.dataset_split,
    )
    # Save exactly what was used: full validation rows, train ids only (keeps outputs small).
    save_jsonl(val_rows, os.path.join(args.output_dir, "val_split.jsonl"))
    save_jsonl([{"id": r["id"]} for r in train_rows], os.path.join(args.output_dir, "train_ids.jsonl"))

    train(model, tokenizer, train_rows, val_rows, args)


if __name__ == "__main__":
    main()