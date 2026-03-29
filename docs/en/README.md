# TrimLLM — Layerwise Weight Condensation for LLM Fine-Tuning

## Overview

TrimLLM is a research framework that implements **layerwise weight condensation** — a structured model compression technique applied *during fine-tuning* of Large Language Models. The core idea is to progressively identify and freeze (replace with zero-parameter `Identity`) redundant transformer layers (MLP and attention modules) as fine-tuning progresses, thereby compressing the model depth-wise while maintaining task performance.

**Key paper:** [TrimLLM: Layerwise Weight Condensation for Efficient LLM Fine-Tuning]

> **Bottom line:** Start with a 32-layer LLaMA, end up with a structurally shorter model trained end-to-end — with fewer effective layers and zero extra inference cost.

---

## Quick Start

### 1. Install Dependencies

```bash
conda create -n trimllm python=3.9
conda activate trimllm
pip install -r requirements.txt
```

### 2. Train with TrimLLM (Layerwise Condensation)

```bash
cd scripts
bash run_clm_llama_lwcd_static_sparse_exhausive.sh {task} {model_path} {batch_size} {trial}
```

Example:
```bash
bash run_clm_llama_lwcd_static_sparse_exhausive.sh hellaswag decapoda-research/llama-7b-hf 8 1
```

### 3. Evaluate with lm_eval Harness

```bash
cd lm_eval/lm-evaluation-harness
pip install -e .
python main.py \
    --model hf-causal \
    --model_args pretrained={OUTPUT_DIR} \
    --tasks hellaswag \
    --device cuda:0
```

---

## Key Arguments

| Argument | Description |
|---|---|
| `--sparsity_ratio` | Fraction of frozen vs. trainable parameters |
| `--max_budget` | Maximum number of layers to remove before stopping |
| `--tie_breaker_strategy` | Strategy when layers have equal importance scores (`naive`, `activation`) |
| `--condense_epoch` | Number of epochs per condensation step |
| `--sparse_update` | `static` (fixed mask) or `dynamic` (re-evaluate importance each step) |
| `--static_mask` | Predefined list of layers to remove, e.g. `"[1,5,10]"` |

---

## Two-Step Workflow

### STEP 1 — Training (TrimLLM Condensation)

Fine-tune the model using `CondensationTrainer`, which progressively removes transformer layers during training.

### STEP 2 — Evaluation (EleutherAI lm-evaluation-harness)

Evaluate the compressed model on 60+ downstream benchmarks using the lm-evaluation-harness fork included in this repo.

---

## Project Structure

```
TrimLLM/
├── scripts/                    # Core training code
│   ├── condensation_trainer.py  # ⭐ THE CORE NOVEL FILE
│   ├── modeling_llama.py        # Modified LLaMA with Identity skip
│   ├── data.py                  # Dataset loading
│   ├── tasks.py                 # Task prompt formatters
│   ├── run_clm_llama_lwcd*.py   # Training entry points
│   └── *.sh                     # Run scripts
├── lm_eval/                    # EleutherAI lm-evaluation-harness fork
│   ├── evaluator.py             # Core eval engine
│   ├── base.py                  # LM interface abstractions
│   ├── models/huggingface.py    # HF model wrapper
│   ├── tasks/__init__.py        # Task registry
│   └── tasks/*.py               # Individual task definitions
├── docs/                       # Documentation
└── requirements.txt
```

See [project_structure.md](./project_structure.md) for a complete file inventory.
