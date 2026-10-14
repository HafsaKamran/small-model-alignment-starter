"""Task 5: persona LoRA-DPO on SmolLM2-360M-Instruct with TRL's DPOTrainer + PEFT.

No separate reference model is loaded: with a PEFT model, DPOTrainer computes the
reference log-probs by running the same network with the adapter disabled, i.e.
the frozen instruct model.

Outputs in --output_dir: adapter/ (LoRA weights + tokenizer), metrics.jsonl (every
train/eval log), split.json (held-out pair indices), config.json, summary.json.
"""

import argparse
import json
import os
import time

import torch

from .persona import load_persona_rows, split_rows, to_conversational

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--n_train", type=int, default=None, help="use only the first N training pairs (smoke tests)")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=ALL_LINEAR)
    # Optimization
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max_steps", type=int, default=-1, help="overrides epochs when > 0")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--eval_steps", type=int, default=5)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                   help="fp16 autocast; defaults to on when CUDA is available")
    return p.parse_args(argv)


def build_trainer(args):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    train_rows, val_rows = split_rows(load_persona_rows(), args.val_fraction, args.seed)
    if args.n_train:
        train_rows = train_rows[: args.n_train]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump({"train_indices": [r["index"] for r in train_rows], "val_indices": [r["index"] for r in val_rows]}, f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 weights; with fp16=True the Trainer autocasts the forward pass and scales gradients,
    # so the trainable LoRA weights stay in fp32 (same rule as the from-scratch runs).
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM")
    fp16 = torch.cuda.is_available() if args.fp16 is None else args.fp16
    config = DPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        beta=args.beta,
        max_length=args.max_length,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=fp16,
        bf16=False,                     # TRL turns bf16 on whenever fp16 is off; T4 GPUs don't support it
        # TRL zeroes every nn.Dropout by default, which would silently disable lora_dropout too.
        # SmolLM2 itself has no dropout, so keeping dropout on only affects the LoRA branch and the
        # reference pass (adapter disabled) stays deterministic.
        disable_dropout=False,
        gradient_checkpointing=False,   # 360M params fits easily; checkpointing would only slow it down
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",             # the adapter is saved once at the end, below
        report_to="none",
    )

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    class JsonlLogger(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **kwargs):
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **(logs or {})}) + "\n")

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list([to_conversational(r) for r in train_rows]),
        eval_dataset=Dataset.from_list([to_conversational(r) for r in val_rows]),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[JsonlLogger()],
    )
    return trainer, tokenizer, len(train_rows), len(val_rows)


def main(argv=None):
    args = parse_args(argv)
    trainer, tokenizer, n_train, n_val = build_trainer(args)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    lora_dropouts = {m.p for n, m in trainer.model.named_modules() if "lora_dropout" in n and hasattr(m, "p")}
    print(f"train pairs {n_train} | val pairs {n_val} | trainable {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%) | lora dropout {sorted(lora_dropouts)}", flush=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    start = time.time()
    baseline = trainer.evaluate()   # step 0: adapter is zero-initialized, so loss = log 2
    print("step 0 eval:", {k: round(v, 4) for k, v in baseline.items() if isinstance(v, float)}, flush=True)
    trainer.train()
    final = trainer.evaluate()
    trainer.save_model(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "adapter"))

    summary = {
        "train_pairs": n_train, "val_pairs": n_val, "trainable_params": trainable, "total_params": total,
        "optimizer_steps": trainer.state.global_step, "minutes": round((time.time() - start) / 60, 2),
        "step0_eval": baseline, "final_eval": final,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("final eval:", {k: round(v, 4) for k, v in final.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main()
