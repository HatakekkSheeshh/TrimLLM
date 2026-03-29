# Project Structure & File Inventory

## Complete File List

### Essential — Core Training Scripts (`scripts/`)

These implement TrimLLM's novel contribution:

| File | Purpose |
|---|---|
| `scripts/condensation_trainer.py` | **THE CORE NOVEL FILE.** Extends HuggingFace `Trainer` with layer-dropping logic. Contains: `LMEvalAdaptor` (bridges trainer to lm_eval harness), layer-freezing via `torch.nn.Identity`, `condense_epoch`, `max_budget`, `sparsity_ratio`, `tie_breaker_strategy` params, activation-norm-based layer importance scoring, static/dynamic sparse modes. Output: `figs/activation/{mean,std,kurtosis,l1_norm,l2_norm,fro_norm}/` |
| `scripts/modeling_llama.py` | Modified LLaMA architecture. Key: `LlamaDecoderLayer.forward()` checks `isinstance(self.self_attn, Identity)` / `isinstance(self.mlp, Identity)` to skip removed layers — true structural removal at inference |
| `scripts/data.py` | Dataset loading: `get_raw_datasets()`, `get_tokenized_datasets()`, `get_lm_datasets()` (causal LM), `process_text2text_datasets()` (QA) |
| `scripts/tasks.py` | Task prompt formatters: `PIQA`, `HellaSwag`, `OpenBookQA`, `ARC`, `RACE`, `SciQ`, `WebQs`, `MedMCQA`, `LEX_GLUE_casehold`, `FiQA`. Exports `task_dict`, `map_dataset_name_and_config()`, `LM_EVAL_TASK_NAME_MAPPING` |
| `scripts/run_clm_llama_lwcd.py` | **Primary training entry point.** Orchestrates: zero-shot eval → `CondensationTrainer` init → training loop → per-epoch eval → thorough `evaluator.simple_evaluate()` |
| `scripts/run_clm_llama_lwcd_static_sparse.py` | Condensation with **static sparsity** (fixed `sparsity_ratio`, no dynamic re-scoring). `sparse_update='static'` |
| `scripts/run_clm_llama_lwcd_static_sparse_static_mask.py` | Condensation with **predefined mask** (`--static_mask="[1,5,10]"`). `sparse_update='static'` + `static_mask` |
| `scripts/run_clm_llama_lwcd_dynamic_sparse.py` | Condensation with **dynamic sparsity** (layer importance re-evaluated each condensation epoch). `sparse_update='dynamic'` |
| `scripts/run_clm_llama_lwcd_exhausive.py` | **Exhaustive search** — tries all layer removal candidates within budget per step |
| `scripts/run_clm_llama_lwcd_cross_val.py` | Cross-validation variant for layer selection |

### Baseline Fine-Tuning Scripts

| File | Purpose |
|---|---|
| `scripts/run_clm_llama.sh` | Baseline full fine-tuning (no compression) using standard HF `run_clm.py` |
| `scripts/run_clm_llama_lora.sh` | LoRA fine-tuning baseline using `run_clm_lora.py` (PEFT) |
| `scripts/run_clm_llama.py` | Standard HF CLM fine-tuning entry point |
| `scripts/run_clm.py` | Standard HuggingFace CLM script (referenced by `run_clm_llama.sh`) |
| `scripts/run_clm_lora.py` | LoRA fine-tuning using PEFT library |

### Evaluation Harness — EleutherAI lm-evaluation-harness (`lm_eval/`)

| File | Purpose |
|---|---|
| `lm_eval/evaluator.py` | Core eval engine: `simple_evaluate()`, `evaluate()` (loglikelihood, greedy_until, loglikelihood_rolling), `make_table()` |
| `lm_eval/base.py` | Abstract LM interface: `BaseLM`, `LM`, `Task`, `MultipleChoiceTask`, `PerplexityTask`, `loglikelihood()`, `loglikelihood_rolling()`, `greedy_until()`, `Request`, `CachingLM` |
| `lm_eval/models/huggingface.py` | `AutoCausalLM`, `AutoSeq2SeqLM` — HuggingFace model wrappers |
| `lm_eval/models/gpt2.py` | `HFLM`, `GPT2LM` — GPT-2/NeoX wrappers |
| `lm_eval/models/anthropic_llms.py` | Anthropic API wrapper |
| `lm_eval/models/__init__.py` | `MODEL_REGISTRY` |
| `lm_eval/main.py` | CLI entry point: `--model`, `--tasks`, `--num_fewshot`, `--batch_size`, `--device` |
| `lm_eval/tasks/__init__.py` | Task registry, exports `ALL_TASKS` |
| `lm_eval/tasks/*.py` (60 files) | Individual task definitions: `hellaswag.py`, `piqa.py`, `arc.py`, `sciq.py`, `gsm8k.py`, `lambada.py`, `winogrande.py`, `xcopa.py`, `xnli.py`, `mmlu.py`, `truthfulqa.py`, `superglue.py`, `glue.py`, `bigbench.py`, `wikitext.py`, `pile.py`, `coqa.py`, `drop.py`, `quac.py`, `triviaqa.py`, `webqs.py`, `naturalqs.py`, `squad.py`, `race.py`, `logiqa.py`, `hendrycks_math.py`, `hendrycks_ethics.py`, `hendrycks_test.py`, `lambada_multilingual.py`, `xstorycloze.py`, `translation.py`, `toxigen.py`, etc. |
| `lm_eval/decontamination/decontaminate.py` | N-gram overlap detection for train/test contamination |
| `lm_eval/decontamination/archiver.py`, `janitor.py` | Decontamination utilities |
| `lm_eval/datasets/` | Custom dataset loading for: `coqa`, `drop`, `hendrycks_math`, `hendrycks_ethics`, `logiqa`, `mutual`, `pile`, `quac`, `triviaqa`, `unscramble`, `asdiv`, `headqa`, `sat_analogies` |
| `lm_eval/setup.py` | Package setup (`pip install -e .`) |
| `lm_eval/requirements.txt` | lm_eval dependencies |
| `lm_eval/pile_statistics.json` | Dataset statistics for Pile subset |

### Run Scripts — Shell Launchers (`scripts/`)

| File | Purpose |
|---|---|
| `scripts/run_clm_llama_lr_*.sh` | LR sweep variants (0.05, 0.25, 0.5, 2, 3, 4) for ablation |
| `scripts/run_clm_llama_lr_0p05_ep*.sh` | Epoch variants (ep5, ep17) at LR=0.05 |
| `scripts/run_clm_llama_lwcd_cross_val.sh` | Cross-validation run |
| `scripts/run_clm_llama_lwcd_dynamic_sparse_exhausive.sh` | Dynamic sparse + exhaustive |
| `scripts/run_clm_llama_lwcd_static_sparse_exhausive.sh` | Static sparse + exhaustive |
| `scripts/run_clm_llama_wikitext_lr_0p05.sh` | WikiText-2 language modeling eval |
| `scripts/run_clm_opt.sh` | OPT model baseline fine-tuning |

### HPC / SLURM Scripts (`slurm/`)

| File | Purpose |
|---|---|
| `slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_*.sh` | 14 HPC cluster variants (task-specific: hellaswag, piqa, sciq, medmcqa, etc.) |
| `slurm/scripts/*_resume.sh` | Resume interrupted runs |
| `slurm/scripts/*_eval_only.sh` | Evaluation-only runs |

---

## Redundant / Miscellaneous Files (Safe to Ignore)

| File/Directory | Why |
|---|---|
| `scripts/modeling_llama_legacy.py` | Legacy version of `modeling_llama.py` — superseded |
| `scripts/run_clm_llama_legacy.py` | Old training script variant |
| `lm_eval/results/` (entire tree) | Generated eval output JSONs — not source code |
| `lm_eval/lm_eval.egg-info/` | Python packaging artifacts from `pip install -e .` |
| `lm_eval/lm_eval/datasets/` subdirs | Custom loaders; most also available from HuggingFace hub |
| `lm_eval/lm_eval/datasets/bigbench_resources/` | BigBench test data — auto-downloaded at runtime |
| `lm_eval/.git/`, `lm_eval/docs/`, `lm_eval/CODEOWNERS` | Dev tooling, docs, git metadata |
| `lm_eval/.gitignore`, `.flake8`, `.pre-commit-config.yaml`, `.coveragerc` | Pre-commit hooks, linting, coverage config |
| `lm_eval/scripts/write_out.py` | Debug utility — writes prompts/logits to JSON |
| `lm_eval/ignore.txt` | Word list for ROUGE metric filtering |
| `scripts/clean_training_data/` | Pile decontamination script — not part of main workflow |
| `slurm/` | HPC scripts — only needed if you use SLURM cluster |
| `.DS_Store` | macOS system file — ignore |

---

## Minimal File Set to Run

If you want the bare minimum to run TrimLLM:

```
scripts/
├── condensation_trainer.py
├── modeling_llama.py
├── data.py
├── tasks.py
├── run_clm_llama_lwcd.py
├── run_clm_llama_lwcd_static_sparse.py
└── requirements.txt

lm_eval/
├── evaluator.py
├── base.py
├── main.py
├── models/
│   └── huggingface.py
└── tasks/
    ├── __init__.py
    └── *.py
```

All other files are: ablation variants, output artifacts, or dev tooling.
