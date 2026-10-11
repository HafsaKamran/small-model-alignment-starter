"""Prompt template. The single source of truth for SFT, DPO and generation."""

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Wrap a raw instruction so the model knows where its answer starts.

    The template ends exactly where the response begins (after "### Response:\\n").
    The response itself is appended later by build_sft_batch, never here.
    """
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())
