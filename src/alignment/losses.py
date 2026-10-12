import torch
from .batching import build_sft_batch


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    # build batch and move to model device
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # get output logits and shift: (B, T, V)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shifted_logits = logits[:, :-1]             # (B, T-1, V)
    shifted_labels = batch["labels"][:, 1:]     # (B, T-1, V)
    
    mask = shifted_labels != -100
    targets = shifted_labels.masked_fill(~mask, 0)  # -100 isn't a valid index, so use a dummy

    logps = torch.log_softmax(shifted_logits.float(), dim=-1)           # (B, T-1, V)
    picked = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)        # (B, T-1)
    return (picked * mask).sum(dim=1)                                   # (B,)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
    labels = build_sft_batch(tokenizer, prompts, responses, max_length)["labels"]
    n_tokens = labels[:, 1:].ne(-100).sum().clamp_min(1)    # same shifted count the public test uses
    return -logps.sum() / n_tokens.to(logps.device)


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_gap - reference_gap)
    loss = -torch.nn.functional.logsigmoid(logits)
    return loss.mean()


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    policy_chosen = compute_response_logprobs(policy_model, tokenizer, prompts, chosen_responses, max_length)
    policy_rejected = compute_response_logprobs(policy_model, tokenizer, prompts, rejected_responses, max_length)
    
    with torch.no_grad():
        reference_chosen = compute_response_logprobs(reference_model, tokenizer, prompts, chosen_responses, max_length).to(policy_chosen.device)
        reference_rejected = compute_response_logprobs(reference_model, tokenizer, prompts, rejected_responses, max_length).to(policy_chosen.device)

    return compute_dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)