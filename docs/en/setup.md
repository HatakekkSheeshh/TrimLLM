# Setup Guide

## Environment

```bash
conda create -n trimllm python=3.9
conda activate trimllm
pip install -r requirements.txt
```

Verify key packages:
```bash
python -c "import transformers; print(transformers.__version__)"  # Expected: ~4.31
python -c "import torch; print(torch.__version__)"                  # Expected: ~1.13
```

> **Note:** If you encounter version errors, check `scripts/memo.txt` for known version fixes and workarounds (e.g., HF version checks, DeepSpeed fixes, layer dropping in LLaMA/BERT).

## Two-Step Workflow

### STEP 1 — Train with TrimLLM

```bash
cd scripts
```

Choose the appropriate script for your use case:

| Script | Mode | When to Use |
|---|---|---|
| `run_clm_llama_lwcd_static_sparse_exhausive.sh` | Static sparse + exhaustive | **Recommended for most cases** |
| `run_clm_llama_lwcd_static_sparse.py` | Static sparse | Fixed sparsity ratio, no exhaustive search |
| `run_clm_llama_lwcd_dynamic_sparse.py` | Dynamic sparse | Layer importance re-evaluated every epoch |
| `run_clm_llama_lwcd_static_sparse_static_mask.py` | Predefined mask | You know exactly which layers to remove |
| `run_clm_llama_lwcd_exhausive.py` | Exhaustive | Search all layer removal candidates |
| `run_clm_llama_lwcd_cross_val.py` | Cross-validation | Layer selection via CV |

Usage:
```bash
bash run_clm_llama_lwcd_static_sparse_exhausive.sh {task} {model_path} {batch_size} {trial}
```

Example:
```bash
bash run_clm_llama_lwcd_static_sparse_exhausive.sh hellaswag decapoda-research/llama-7b-hf 8 1
```

Key arguments:
```bash
--sparsity_ratio 0.75          # 75% frozen / 25% trainable
--max_budget 16                # Remove at most 16 layers
--condense_epoch 1             # 1 epoch per condensation step
--sparse_update static         # or: dynamic
--tie_breaker_strategy naive   # or: activation
--static_mask "[1,5,10]"       # Optional: predefined layer removal list
```

### STEP 2 — Evaluate with lm_eval Harness

```bash
cd lm_eval/lm-evaluation-harness
pip install -e .
```

Evaluate your trained model:
```bash
python main.py \
    --model hf-causal \
    --model_args pretrained={OUTPUT_DIR} \
    --tasks hellaswag,piqa,arc,sciq \
    --device cuda:0 \
    --batch_size 8
```

Evaluate the **original pretrained model** for baseline comparison:
```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag,piqa,arc,sciq \
    --device cuda:0 \
    --batch_size 8
```

---

## Supported Tasks

The project supports the following downstream tasks (via `scripts/tasks.py` and lm_eval):

| Task | lm_eval name | Type |
|---|---|---|
| PIQA | `piqa` | Commonsense reasoning |
| HellaSwag | `hellaswag` | Commonsense reasoning |
| ARC (Challenge) | `arc_challenge` | Science QA |
| OpenBookQA | `openbookqa` | Science QA |
| RACE | `race` | Reading comprehension |
| SciQ | `sciq` | Science QA |
| WebQs | `webqs` | Fact Q&A |
| MedMCQA | `medmcqa` | Medical QA |
| MMLU | `mmlu` | Multi-task (57 subjects) |
| GSM8K | `gsm8k` | Math reasoning |
| WinoGrande | `winogrande` | Commonsense reasoning |
| TruthfulQA | `truthfulqa` | Truthfulness |

---

## SLURM HPC Setup

If running on a SLURM cluster, use the scripts under `slurm/scripts/`:

```bash
sbatch slurm/scripts/run_clm_llama_lwcd_static_sparse_exhaustic_hellaswag.sh
```

Available SLURM variants cover: `hellaswag`, `piqa`, `sciq`, `medmcqa`, `race`, `webqs`, `mmlu`, `lex_glue_casehold`, `fiqa`, and more.

Variants:
- `*_resume.sh` — resume an interrupted run
- `*_eval_only.sh` — run evaluation only (no training)

---

## Output Artifacts

After training:
```
output_dir/
├── figs/
│   └── activation/
│       ├── mean/
│       ├── std/
│       ├── kurtosis/
│       ├── l1_norm/
│       ├── l2_norm/
│       └── fro_norm/
│           └── layer_scores_per_epoch.png
├── checkpoint/
│   └── pytorch_model.bin   # or adapter_model.bin for LoRA
└── eval_results.json       # Per-epoch metrics
```

After evaluation:
```
lm_eval/results/
└── {model_name}/
    └── {task_name}_results.json
```
