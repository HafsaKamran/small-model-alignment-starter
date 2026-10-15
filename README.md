# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

PA2 submission: supervised fine-tuning of `Qwen/Qwen2.5-0.5B`, two from-scratch DPO runs
(HH-RLHF and UltraFeedback), a TRL `DPOTrainer` + LoRA persona run on
`HuggingFaceTB/SmolLM2-360M-Instruct`, and pairwise evaluation with a Kaggle judge model.

- **Final report:** [`docs/final_report.pdf`](docs/final_report.pdf)
- **Assignment spec:** [`docs/assignment_manual.pdf`](docs/assignment_manual.pdf)

## Repository layout

```text
src/alignment/
  formatting.py          prompt template shared by SFT, DPO and generation
  data.py                UltraChat -> single-turn SFT rows
  preference_data.py     HH-RLHF / UltraFeedback -> single-turn preference pairs
  batching.py            SFT/DPO batches with response-only labels (-100 on prompt and padding)
  losses.py              response log-probs, SFT loss, DPO loss
  train_sft.py           SFT training loop (no HF Trainer)
  train_dpo.py           from-scratch DPO training loop (both tracks)
  eval_sets.py           fixed eval sets; builds the AlpacaEval 300/seed-42 subset
  generate.py            batched greedy generation for the Qwen models
  pairs.py               blinded A/B pair files for the judge
  persona.py, train_lora_dpo.py, generate_persona.py   Task 5 (TRL DPOTrainer + PEFT LoRA)
  judging.py             pairwise judging: batching, retries, resume, A/B mapping, summaries
scripts/judge_pairwise_kaggle.py   CLI entry point for judging.py
submission_adapter.py    grading interface (thin wrappers around src/alignment)
tests/                   public smoke tests + edge-case, generation, persona and judging tests
notebooks/               Kaggle notebooks 01-05, saved with their outputs
eval_sets/               alpaca_eval_300_seed42.json, hh_rlhf_eval_300_seed42.json, persona_style_eval_40.json
data/persona_lora/       provided persona preference data
docs/                    final_report.pdf, assignment_manual.pdf, report_notes_judging.md
results/                 small artifacts from the Kaggle runs (see below)
```

`compute_sft_loss` uses the mean over all unmasked response tokens in the batch.

## Running the notebooks (Kaggle)

Run them in order. Each one clones this repo; a private repo needs a Kaggle secret named
`GITHUB_TOKEN`. Every notebook has a quick pilot or sanity mode; set `MODE = "full"` and use
**Save & Run All** for the real run, then attach that saved version as an input where listed below.

| # | Notebook | Accelerator | Kaggle inputs | Produces |
|---|---|---|---|---|
| 01 | `01-sft-train.ipynb` | GPU T4 x2 | none | `outputs/sft/{best,final}` + metrics |
| 02 | `02-dpo-train.ipynb` | GPU T4 x2 | saved **01** | `outputs/dpo_hh`, `outputs/dpo_ultrafeedback` |
| 03 | `03-generate.ipynb` | GPU T4 x2 | saved **01** and **02** | generations + DPO-vs-SFT pair files |
| 04 | `04-lora-dpo.ipynb` | GPU T4 (one GPU used) | none | LoRA adapter, persona generations + pair file |
| 05 | `05-judge.ipynb` | none | saved **03** and **04** | judged files + summaries |

Notebook 05 must be created from <https://www.kaggle.com/benchmarks/tasks/new> (a benchmark task
notebook), which provides `kaggle_benchmarks` and judge-model access. Judge: `google/gemini-3.6-flash`.

The notebooks are saved with the outputs of their full Kaggle runs. One exception: the
`!pip install -q pytest` line in notebook 05's setup cell was added after its run (the benchmark
environment lacked pytest), so that cell's saved output still shows "No module named pytest".
The judge tests were run locally instead (see Tests).

Setup inside each notebook keeps Kaggle's preinstalled PyTorch:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" "trl>=0.29,<1" "peft>=0.11.0" \
    huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

## Results

Model checkpoints are not in the repo. Everything needed to reproduce the tables and plots is in `results/`:

| Path | Contents |
|---|---|
| `results/{sft,dpo_hh,dpo_ultrafeedback,lora_persona}/` | `config.json` (exact run arguments), `metrics.jsonl`, `summary.json`, train/val split ids, training-curve PNGs; `lora_persona/style_metrics.json` |
| `results/generations/{alpaca,hh,persona}/` | one JSON per model, plus `*.config.json` with the decoding settings |
| `results/eval/` | blinded pair files (`*_pairs.jsonl`); model identities are under `hidden` |
| `results/judged/` | per-judgment rows (`*_judged.jsonl`), raw judge replies (`*.raw.jsonl`), summaries (`*.summary.json`), `results_table.csv`, `win_rates.png` |

`docs/report_notes_judging.md` documents how the judged results were checked (both-order aggregation,
blinding, position and length bias).

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py   # public adapter checks
PYTHONPATH=. python -m pytest -q tests/                               # full suite
```

All tests run offline on CPU with toy models; no GPU, model download or dataset download needed.
