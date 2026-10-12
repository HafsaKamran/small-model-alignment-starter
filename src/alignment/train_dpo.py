"""From-scratch DPO training, starting from the SFT checkpoint.

Run from the repo root (relative imports need the -m form):

    python -m src.alignment.train_dpo --track hh --sft_checkpoint <path>/sft/best --output_dir outputs/dpo_hh

The same loop is used for both tracks; only --track (the preference data) changes.

Device layout:
  default:  policy on cuda:0, frozen reference on cuda:1 (the manual's suggestion)
  parallel: --policy_device cuda:1 --ref_device cuda:1 puts one whole run on one GPU,
            so two runs (hh and ultrafeedback) can train at the same time.
"""

import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from .data import save_jsonl
from .losses import compute_dpo_loss_from_logps, compute_response_logprobs
from .preference_data import TRACKS, load_preference_splits


def parse_args(argv=None):
    has_two_gpus = torch.cuda.device_count() >= 2
    default_policy = "cuda:0" if torch.cuda.is_available() else "cpu"
    default_ref = "cuda:1" if has_two_gpus else default_policy

    p = argparse.ArgumentParser(description="From-scratch DPO")
    p.add_argument("--track", choices=sorted(TRACKS), required=True)
    p.add_argument("--sft_checkpoint", required=True, help="path to the SFT best/ directory")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_train", type=int, default=15_000)
    p.add_argument("--n_val", type=int, default=1_000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_tokens", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=0, help="0 = fixed learning rate (manual default)")
    p.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = no limit)")
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validation runs")
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy_device", default=default_policy)
    p.add_argument("--ref_device", default=default_ref)
    p.add_argument("--ref_dtype", choices=["float32", "float16"], default="float32",
                   help="float32 makes policy and reference numerically identical at step 0")
    p.add_argument("--gradient_checkpointing", action="store_true", help="use if you hit CUDA OOM")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_pairs(rows):
    return ([r["prompt"] for r in rows], [r["chosen"] for r in rows], [r["rejected"] for r in rows])


def sequence_logps(policy, reference, tokenizer, prompts, chosen, rejected, max_length, autocast, policy_grad=True):
    """The four (B,) log-prob vectors DPO needs, all returned on the policy's device.

    The reference runs first: its GPU work is queued on its own device, so it can
    overlap with the policy's forward/backward instead of waiting for it.
    """
    with torch.no_grad(), autocast():
        ref_chosen = compute_response_logprobs(reference, tokenizer, prompts, chosen, max_length)
        ref_rejected = compute_response_logprobs(reference, tokenizer, prompts, rejected, max_length)
    with torch.set_grad_enabled(policy_grad), autocast():
        pol_chosen = compute_response_logprobs(policy, tokenizer, prompts, chosen, max_length)
        pol_rejected = compute_response_logprobs(policy, tokenizer, prompts, rejected, max_length)
    device = pol_chosen.device
    return pol_chosen, pol_rejected, ref_chosen.to(device), ref_rejected.to(device)


def reward_stats(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta):
    """Implicit rewards (manual eq. 7-8). Note r_w - r_l is exactly the DPO logit."""
    r_w = beta * (pol_chosen - ref_chosen)
    r_l = beta * (pol_rejected - ref_rejected)
    return r_w.detach(), r_l.detach()


@torch.no_grad()
def evaluate(policy, reference, tokenizer, rows, args, autocast):
    policy.eval()
    loss_sum, correct, margin_sum, rw_sum, rl_sum, n = 0.0, 0, 0.0, 0.0, 0.0, 0
    for start in range(0, len(rows), args.eval_batch_size):
        prompts, chosen, rejected = collate_pairs(rows[start:start + args.eval_batch_size])
        logps = sequence_logps(policy, reference, tokenizer, prompts, chosen, rejected,
                               args.max_length, autocast, policy_grad=False)
        batch = len(prompts)
        loss_sum += compute_dpo_loss_from_logps(*logps, args.beta).item() * batch
        r_w, r_l = reward_stats(*logps, args.beta)
        correct += (r_w > r_l).sum().item()
        margin_sum += (r_w - r_l).sum().item()
        rw_sum += r_w.sum().item()
        rl_sum += r_l.sum().item()
        n += batch
    policy.train()
    return {"val_loss": loss_sum / n, "val_reward_acc": correct / n, "val_reward_margin": margin_sum / n,
            "val_reward_chosen": rw_sum / n, "val_reward_rejected": rl_sum / n}


def log_metrics(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def save_checkpoint(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def make_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def train(policy, reference, tokenizer, train_rows, val_rows, args):
    """The DPO loop. Separate from main() so it can be tested on tiny models."""
    policy_device = torch.device(args.policy_device)
    use_amp = policy_device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    policy.to(policy_device).train()
    reference.to(torch.device(args.ref_device)).eval()
    reference.requires_grad_(False)
    for model in (policy, reference):
        if hasattr(model, "config"):
            model.config.use_cache = False

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_pairs, generator=generator)
    total_micro = len(loader) * args.epochs
    total_steps = math.ceil(total_micro / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
        total_micro = min(total_micro, total_steps * args.grad_accum)

    optimizer = make_optimizer(policy, args)
    warmup = args.warmup_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp)

    print(f"track={args.track} train={len(train_rows)} val={len(val_rows)} micro_batches={total_micro} "
          f"optimizer_steps={total_steps} policy={args.policy_device} ref={args.ref_device} amp={use_amp}",
          flush=True)

    # Step 0: policy == reference, so loss should be log(2) = 0.6931 and reward accuracy 0.
    start_metrics = evaluate(policy, reference, tokenizer, val_rows, args, autocast)
    log_metrics(metrics_path, {"type": "eval", "step": 0, **start_metrics, "baseline": True})

    best_acc = -1.0
    step, micro, skipped_steps = 0, 0, 0
    window = {"loss": 0.0, "correct": 0, "margin": 0.0, "pairs": 0}
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    done = False
    for epoch in range(args.epochs):
        for prompts, chosen, rejected in loader:
            group_start = (micro // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, total_micro - group_start)

            logps = sequence_logps(policy, reference, tokenizer, prompts, chosen, rejected, args.max_length, autocast)
            loss = compute_dpo_loss_from_logps(*logps, args.beta)
            scaler.scale(loss / group_size).backward()

            r_w, r_l = reward_stats(*logps, args.beta)
            window["loss"] += loss.item() * len(prompts)
            window["correct"] += (r_w > r_l).sum().item()
            window["margin"] += (r_w - r_l).sum().item()
            window["pairs"] += len(prompts)
            micro += 1

            if micro - group_start < group_size:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_steps += 1  # fp16 overflow: the scaler skipped this update
            else:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                elapsed = time.time() - start_time
                log_metrics(metrics_path, {
                    "type": "train", "step": step, "epoch": epoch,
                    "train_loss": window["loss"] / window["pairs"],
                    "train_reward_acc": window["correct"] / window["pairs"],
                    "train_reward_margin": window["margin"] / window["pairs"],
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "skipped_steps": skipped_steps,
                    "elapsed_min": round(elapsed / 60, 2),
                    "eta_min": round(elapsed / step * (total_steps - step) / 60, 1),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated(policy_device) / 1e9, 2) if use_amp else 0.0,
                })
                window = {"loss": 0.0, "correct": 0, "margin": 0.0, "pairs": 0}

            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate(policy, reference, tokenizer, val_rows, args, autocast)
                improved = metrics["val_reward_acc"] > best_acc
                log_metrics(metrics_path, {"type": "eval", "step": step, **metrics, "best_so_far": improved})
                if improved:
                    best_acc = metrics["val_reward_acc"]
                    save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "best"))

            if step >= total_steps:
                done = True
                break
        if done:
            break

    save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "final"))
    summary = {"track": args.track, "best_val_reward_acc": best_acc, "optimizer_steps": step,
               "skipped_steps": skipped_steps, "minutes": round((time.time() - start_time) / 60, 2)}
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

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    # Policy: fp32 weights for GradScaler; autocast does the fp16 compute.
    policy = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.float32)
    reference = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=getattr(torch, args.ref_dtype))
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_rows, val_rows = load_preference_splits(
        args.track, tokenizer, n_train=args.n_train, n_val=args.n_val,
        seed=args.seed, max_prompt_tokens=args.max_prompt_tokens)
    save_jsonl(val_rows, os.path.join(args.output_dir, "val_split.jsonl"))
    save_jsonl([{"id": r["id"]} for r in train_rows], os.path.join(args.output_dir, "train_ids.jsonl"))

    train(policy, reference, tokenizer, train_rows, val_rows, args)


if __name__ == "__main__":
    main()
