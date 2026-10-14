"""Judge a blinded pairwise JSONL file with a Kaggle model-proxy LLM (manual §13).

    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/hh_dpo_hh_vs_sft_pairs.jsonl \
        --output_path outputs/eval/hh_dpo_hh_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

All logic lives in src/alignment/judging.py; see its docstring for how a run works.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alignment.judging import main  # noqa: E402

if __name__ == "__main__":
    main()
