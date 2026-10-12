import torch


def tokenize_example_pair(tokenizer, prompt, response, eos, max_length):
    """Tokenizes a single pair and returns (ids, labels) truncated to max_length."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    
    # build ids and labels, append eos
    eos = [] if eos is None else [eos]
    ids = prompt_ids + response_ids + eos
    labels = [-100] * len(prompt_ids) + response_ids + eos
    
    # truncate to max_length
    ids, labels = ids[:max_length], labels[:max_length]
    
    return ids, labels


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = pad_id if pad_id is not None else (eos_id if eos_id is not None else 0)
        
    rows = [tokenize_example_pair(tokenizer, p, r, eos_id, max_length) for p, r in zip(prompts, responses)]
    longest = max((len(ids) for ids, _ in rows), default=0)      # for padding up to longest example
    
    all_ids, all_labels, all_mask = [], [], []
    for ids, labels in rows:
        pad_length = longest - len(ids)
        all_ids.append(ids + [pad_id] * pad_length)
        all_labels.append(labels + [-100] * pad_length)
        all_mask.append([1] * len(ids) + [0] * pad_length)
    
    return {
        "input_ids": torch.tensor(all_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_mask, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }
    
    
def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {
        "chosen": chosen,
        "rejected": rejected
    }