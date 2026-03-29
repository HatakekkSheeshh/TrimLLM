# Evaluation Without Training

> **You don't need to train to evaluate.** TrimLLM ships with the full EleutherAI lm-evaluation-harness — you can evaluate any pretrained HuggingFace model (LLaMA, OPT, GPT-2, etc.) on 60+ benchmarks with a single command. No fine-tuning, no GPU cluster, no CondensationTrainer needed.

---

## Zero-Shot Evaluation (No Training)

This is the fastest way to get benchmark numbers — just load a model from HuggingFace and run the harness.

### 1. Install the harness

```bash
cd lm_eval/lm-evaluation-harness
pip install -e .
```

### 2. Run evaluation

```bash
# Evaluate a single task
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

# Evaluate multiple tasks at once
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag,piqa,arc_challenge,openbookqa,sciq \
    --device cuda:0 \
    --batch_size 8

# Print full result breakdown
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag \
    --device cuda:0 \
    --output_path ./results/llama7b_hellaswag.json
```

That's it. The harness downloads the model, runs inference on the evaluation dataset, and outputs accuracy/F1/etc.

---

## What Models Can You Evaluate?

Any model on HuggingFace that uses a standard causal language modeling architecture:

| Model | Layers | Memory (fp16) | GPU |
|---|---|---|---|
| `decapoda-research/llama-7b-hf` | 32 | ~14 GB | 1× A10G or better |
| `facebook/opt-1.3b` | 24 | ~2.6 GB | 1× consumer GPU ✅ |
| `facebook/opt-6.7b` | 32 | ~13 GB | 1× A10G/A100 |
| `gpt2-medium` | 24 | ~1.5 GB | 1× any GPU ✅ |
| `gpt2-large` | 36 | ~3 GB | 1× any GPU ✅ |
| `mistralai/Mistral-7B-v0.1` | 32 | ~14 GB | 1× A10G/A100 |
| `microsoft/phi-2` | 2 | ~1.4 GB | 1× any GPU ✅ |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 22 | ~2 GB | 1× any GPU ✅ |

---

## GPU Memory by Model Size

If your local GPU can't fit the full model, use quantization:

```bash
# 8-bit quantization — halves VRAM usage
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf,load_in_8bit=True \
    --tasks hellaswag \
    --device cuda:0

# 4-bit quantization (QLoRA-style) — ~25% of fp16 VRAM
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf,load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16 \
    --tasks hellaswag \
    --device cuda:0
```

**Quantization comparison:**

| Precision | VRAM for LLaMA-7B | Quality loss |
|---|---|---|
| fp16 (full) | ~14 GB | None |
| int8 (8-bit) | ~7 GB | Minimal |
| int4 (4-bit, NF4) | ~3.5 GB | Moderate |
| fp32 | ~28 GB | None (overkill) |

---

## Evaluate a Fine-Tuned Checkpoint

After running TrimLLM training, the checkpoint is saved at:

```
outputs/{MODEL}/{TASK}_bs_{bs}/layerwise_condense_sparse_exhausive_sr{sr}_trial_{trial}_max_fro/
```

Evaluate it the same way:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=./outputs/decapoda-research/llama-7b-hf/piqa_bs_2/layerwise_condense_sparse_exhausive_sr0.625_trial_4_max_fro/checkpoint-24500 \
    --tasks piqa,hellaswag,arc_challenge \
    --device cuda:0 \
    --batch_size 8
```

**Note:** Since `modeling_llama.py` (the modified architecture) is not registered in the standard HuggingFace model type, you must evaluate checkpoints within the `TrimLLM/scripts/` directory tree where the custom model class is importable. Or copy `modeling_llama.py` into your evaluation environment and register it.

---

## All lm_eval Arguments

```bash
python main.py \
    --model hf-causal              # Model type: hf-causal, hf-seq2seq, gpt2, etc.
    --model_args pretrained=...     # HuggingFace model path or local checkpoint
                                  # Comma-separated: pretrained=...,load_in_8bit=True
    --tasks hellaswag              # Task name(s), comma-separated
    --num_fewshot 0                # Number of in-context examples (0 = zero-shot)
    --batch_size 8                 # Increase if GPU memory allows
    --device cuda:0                # Device
    --output_path ./results.json   # Save results to JSON
    --limit 100                    # Run only first N samples (for quick testing)
    --verbosity INFO               # DEBUG, INFO, WARNING
```

---

## Available Tasks (Selected)

| Task | lm_eval name | Domain | Few-shot |
|---|---|---|---|
| HellaSwag | `hellaswag` | Commonsense | 0/10 |
| PIQA | `piqa` | Commonsense | 0 |
| ARC Challenge | `arc_challenge` | Science QA | 0/25 |
| ARC Easy | `arc_easy` | Science QA | 0 |
| OpenBookQA | `openbookqa` | Science QA | 0 |
| SciQ | `sciq` | Science QA | 0/3 |
| MMLU | `mmlu` | Multi-domain | 0/5 |
| GSM8K | `gsm8k` | Math reasoning | 0/5 |
| WinoGrande | `winogrande` | Commonsense | 0 |
| TruthfulQA | `truthfulqa` | Truthfulness | 0 |
| RACE | `race` | Reading comp. | 0 |
| WikiText | `wikitext` | Perplexity | 0 |
| Lambada | `lambada` | Perplexity | 0 |

Full list: `python main.py --tasks help`

---

## Quick Test (Single GPU, ~5 minutes)

To verify everything works before running full benchmarks:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=gpt2 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 16 \
    --limit 50 \
    --verbosity INFO
```

This runs 50 samples of HellaSwag on GPT-2 (~2 GB VRAM, any GPU).
