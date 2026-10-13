"""Edge-case checks for the submission adapter, beyond the public smoke tests.

Covers exact numerical values, truncation boundaries, unusual tokenizer
configurations, mixed sequence lengths, row ordering, and small optimization
steps (manual section 4.2, "Public Tests Are a Minimum Check").
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class BigramLM(nn.Module):
    """Logits at position t depend on input_ids[t], so off-by-one shift bugs show up."""

    def __init__(self, vocab_size, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, generator=g))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(logits=self.table[input_ids])


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class PadIsEosTokenizer(ToyTokenizer):
    """Like many causal-LM tokenizers (e.g. Qwen base), pad and eos share an id."""
    pad_token_id = 1


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


def expected_logp(model, tokenizer, prompt, response, max_length):
    """Reference implementation: sum of log p(token | previous token) over the response span."""
    p = tokenizer.encode(prompt, add_special_tokens=False)
    r = tokenizer.encode(response, add_special_tokens=False)
    eos = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    ids = (p + r + eos)[:max_length]
    total = 0.0
    for t in range(max(len(p), 1), len(ids)):
        logits = model(torch.tensor([ids[:t]])).logits[0, -1]
        total += torch.log_softmax(logits.float(), -1)[ids[t]].item()
    return total


# ---------------------------------------------------------------- DPO objective


def test_dpo_from_logps_matches_hand_computed_value():
    pc = torch.tensor([-1.0, -3.0])
    pr = torch.tensor([-4.0, -2.0])
    rc = torch.tensor([-2.0, -2.5])
    rr = torch.tensor([-3.0, -2.5])
    beta = 0.5
    logits = beta * ((pc - pr) - (rc - rr))  # [1.0, -0.5]
    expected = -torch.nn.functional.logsigmoid(logits).mean()
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    torch.testing.assert_close(loss, expected)


def test_dpo_from_logps_decreases_when_policy_prefers_chosen():
    ref = torch.tensor([-5.0])
    good = adapter.compute_dpo_loss_from_logps(torch.tensor([-1.0]), torch.tensor([-9.0]), ref, ref, 0.1)
    bad = adapter.compute_dpo_loss_from_logps(torch.tensor([-9.0]), torch.tensor([-1.0]), ref, ref, 0.1)
    assert good < math.log(2) < bad


def test_dpo_from_logps_is_stable_for_extreme_logits():
    big = torch.tensor([1e4])
    zero = torch.tensor([0.0])
    loss = adapter.compute_dpo_loss_from_logps(zero, big, zero, zero, beta=1.0)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(1e4))


def test_dpo_from_logps_gradients_have_the_right_sign():
    pc = torch.tensor([-2.0], requires_grad=True)
    pr = torch.tensor([-2.0], requires_grad=True)
    ref = torch.tensor([-2.0])
    adapter.compute_dpo_loss_from_logps(pc, pr, ref, ref, 0.1).backward()
    assert pc.grad.item() < 0  # descending the loss raises the chosen log-prob
    assert pr.grad.item() > 0  # and lowers the rejected log-prob


# ------------------------------------------------------- response log-probs


def test_response_logprobs_match_reference_with_context_dependent_model():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    prompts, responses = ["Q: hi ", "Tell me: "], ["yes", "a longer answer"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=64)
    want = torch.tensor([expected_logp(model, tok, p, r, 64) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_response_logprobs_do_not_depend_on_other_rows_in_the_batch():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    alone = adapter.compute_response_logprobs(model, tok, ["P: "], ["short"], max_length=64)
    mixed = adapter.compute_response_logprobs(
        model, tok, ["P: ", "A much longer prompt here: "], ["short", "and a much longer response too"], max_length=64
    )
    torch.testing.assert_close(alone[0], mixed[0])


def test_row_order_is_preserved():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    prompts, responses = ["a: ", "bb: ", "ccc: "], ["x", "yy yy", "zzz zzz zzz"]
    fwd = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=64)
    rev = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], max_length=64)
    torch.testing.assert_close(fwd, rev.flip(0))

    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=64)
    for row, (p, r) in enumerate(zip(prompts, responses)):
        n = len(tok.encode(p + r)) + 1
        assert batch["input_ids"][row, :n].tolist() == tok.encode(p) + tok.encode(r) + [tok.eos_token_id]


# ------------------------------------------------------------- truncation


def test_eos_kept_at_exact_fit_and_dropped_one_token_short():
    tok = ToyTokenizer()
    p, r = "Q: ", "abc"
    full = len(tok.encode(p)) + len(tok.encode(r)) + 1

    fit = adapter.build_sft_batch(tok, [p], [r], max_length=full)
    assert fit["input_ids"].shape[1] == full
    assert fit["labels"][0, -1] == tok.eos_token_id

    short = adapter.build_sft_batch(tok, [p], [r], max_length=full - 1)
    assert short["input_ids"].shape[1] == full - 1
    assert tok.eos_token_id not in short["labels"][0].tolist()
    assert short["labels"][0, -1] == tok.encode(r)[-1]


def test_no_row_exceeds_max_length():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["x" * 50, "y: "], ["z" * 50, "ok"], max_length=10)
    assert batch["input_ids"].shape[1] <= 10
    dpo = adapter.build_dpo_batch(tok, ["x" * 50], ["z" * 50], ["w"], max_length=10)
    assert dpo["chosen"]["input_ids"].shape[1] <= 10
    assert dpo["rejected"]["input_ids"].shape[1] <= 10


def test_prompt_longer_than_max_length_gives_zero_logp_and_finite_loss():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    prompts, responses = ["p" * 30], ["response"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=8)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=8)
    assert logps.item() == 0.0
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=8)
    assert torch.isfinite(loss)


def test_truncated_logprobs_match_reference():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    prompts, responses = ["Q: ", "Question: "], ["abcdefghij", "xyz"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=8)
    want = torch.tensor([expected_logp(model, tok, p, r, 8) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


# -------------------------------------------------- tokenizer configurations


def test_pad_equal_to_eos_keeps_eos_label_and_masks_padding():
    tok = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "longer"], max_length=32)
    short_len = len(tok.encode("Q: a")) + 1
    assert batch["labels"][0, short_len - 1] == tok.eos_token_id
    assert batch["attention_mask"][0, short_len - 1] == 1
    assert batch["labels"][0, short_len:].eq(-100).all()
    assert batch["attention_mask"][0, short_len:].eq(0).all()


def test_tokenizer_without_pad_token_still_pads():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "longer"], max_length=32)
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()


def test_tokenizer_without_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["abc"], max_length=32)
    assert batch["input_ids"].shape[1] == len(tok.encode("Q: abc"))
    assert batch["labels"][0].ne(-100).sum() == 3


# ------------------------------------------------------------- DPO batches


def test_dpo_batch_masks_the_same_prompt_on_both_sides():
    tok = ToyTokenizer()
    prompts = ["Prompt one: ", "P2: "]
    batch = adapter.build_dpo_batch(tok, prompts, ["good answer", "y"], ["n", "bad answer here"], max_length=64)
    for row, p in enumerate(prompts):
        n = len(tok.encode(p))
        for side in ("chosen", "rejected"):
            assert batch[side]["input_ids"][row, :n].tolist() == tok.encode(p)
            assert batch[side]["labels"][row, :n].eq(-100).all()
            assert batch[side]["labels"][row, n] != -100


# ------------------------------------------------------------ full DPO loss


def test_dpo_loss_is_log_two_when_policy_equals_reference():
    tok = ToyTokenizer()
    policy = BigramLM(tok.vocab_size, seed=3)
    reference = BigramLM(tok.vocab_size, seed=3)
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["Q: ", "R: "], ["yes", "ok"], ["no", "nope"], 0.1, 32)
    torch.testing.assert_close(loss, torch.tensor(math.log(2)))


def test_dpo_loss_matches_logps_composition():
    tok = ToyTokenizer()
    policy, reference = BigramLM(tok.vocab_size, seed=1), BigramLM(tok.vocab_size, seed=2)
    args = (tok, ["Q: ", "Question two: "], ["good", "fine answer"], ["bad", "x"])
    prompts, chosen, rejected = args[1], args[2], args[3]
    loss = adapter.compute_dpo_loss(policy, reference, *args, beta=0.3, max_length=32)
    lp = lambda m, r: adapter.compute_response_logprobs(m, tok, prompts, r, 32)
    want = adapter.compute_dpo_loss_from_logps(
        lp(policy, chosen), lp(policy, rejected), lp(reference, chosen), lp(reference, rejected), 0.3
    )
    torch.testing.assert_close(loss, want.detach())


def test_reference_is_not_part_of_the_graph():
    tok = ToyTokenizer()
    policy, reference = BigramLM(tok.vocab_size, seed=1), BigramLM(tok.vocab_size, seed=2)
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["Q: "], ["yes"], ["no"], 0.1, 32)
    loss.backward()
    assert reference.table.grad is None
    assert policy.table.grad is not None and policy.table.grad.abs().sum() > 0


def test_a_few_dpo_steps_increase_reward_margin():
    tok = ToyTokenizer()
    policy, reference = BigramLM(tok.vocab_size, seed=5), BigramLM(tok.vocab_size, seed=5)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    prompts, chosen, rejected = ["Q: ", "Ask: "], ["yes please", "sure"], ["no way", "never"]
    lp = lambda m, r: adapter.compute_response_logprobs(m, tok, prompts, r, 32)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, 0.1, 32)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    with torch.no_grad():
        margin = (lp(policy, chosen) - lp(reference, chosen)) - (lp(policy, rejected) - lp(reference, rejected))
    assert (margin > 0).all()


# ----------------------------------------------------------------- SFT loss


def test_sft_loss_matches_token_mean_reference_with_mixed_lengths():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    prompts, responses = ["Q: ", "Longer question: "], ["a", "a much longer response"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=64)
    total = sum(expected_logp(model, tok, p, r, 64) for p, r in zip(prompts, responses))
    n_tokens = sum(len(tok.encode(r)) + 1 for r in responses)
    per_example = -total / len(prompts)
    per_token = -total / n_tokens
    assert any(math.isclose(loss.item(), v, rel_tol=1e-5) for v in (per_example, per_token))


def test_sft_steps_reduce_loss():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    args = (tok, ["Q: "], ["hello"], 32)
    first = adapter.compute_sft_loss(model, *args)
    for _ in range(5):
        opt.zero_grad()
        adapter.compute_sft_loss(model, *args).backward()
        opt.step()
    assert adapter.compute_sft_loss(model, *args) < first


@pytest.mark.parametrize("bias_scale", [0.0, 5.0])
def test_toy_model_logps_are_finite(bias_scale):
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=bias_scale)
    logps = adapter.compute_response_logprobs(model, tok, ["Q: ", "W: "], ["x", "yyyy"], max_length=16)
    assert torch.isfinite(logps).all() and (logps < 0).all()
