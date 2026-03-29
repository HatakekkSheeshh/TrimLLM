# Hardware Requirements

## Training

Training LLaMA-7B (32 layers) with TrimLLM requires significant GPU resources due to **full-parameter fine-tuning + FSDP sharding + bf16 precision**.

| Component | Requirement |
|---|---|
| **GPUs** | **8 minimum** (local) / **16** (SLURM production scripts) |
| **GPU memory** | ~40GB per GPU (A100-40GB recommended) |
| **GPU compute** | Ampere+ architecture (A100, RTX 3090, RTX 4090, H100) |
| **Precision** | `--bf16 True` (Ampere+) or `--fp16 True` ( Volta+) |
| **CPU** | ~96 CPUs per node (SLURM config) |
| **RAM** | High; node runs in `--exclusive` mode |
| **Interconnect** | NVLink / NVSwitch highly recommended for multi-GPU FSDP |

### GPU Memory Calculation (LLaMA-7B, 32 layers)

```
Full model (bf16):          ≈ 14 GB
Optimizer states (AdamW):   ≈ 28 GB   (3× model params for Adam)
Gradients (bf16):           ≈ 14 GB
Activations (per batch):    ≈  4–8 GB  (sequence length, batch size dependent)
─────────────────────────────────────
Total (no sharding):        ≈ 60–64 GB  →  impossible on any single GPU

With FSDP across 8 GPUs:
  Per GPU:  14 / 8  +  28 / 8  +  14 / 8  +  activations
           ≈ 1.75   +  3.5    +  1.75   +  0.5–1
           ≈ 7.5–8 GB per GPU  →  A100-40GB is comfortable
```

### What the Scripts Actually Use

```bash
# Local training (scripts/run_clm_llama_lwcd_static_sparse_exhausive.sh)
torchrun --nproc_per_node=8 \
    --bf16 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'

# SLURM HPC (slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_hellaswag_slurm.sh)
#SBATCH --nodes=1
#SBATCH --gres=gpu:16
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
torchrun --nproc_per_node=16 \
    --nnodes $SLURM_NNODES --node_rank=$SLURM_PROCID \
    --bf16 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
```

---

## Evaluation

Evaluation is **inference-only** and far less demanding than training.

| Component | Requirement |
|---|---|
| **GPUs** | **1 GPU minimum** |
| **GPU memory** | ~16–24 GB (varies by model size and batch size) |
| **Precision** | bf16 or fp16 (auto-detected) |

```bash
# Single GPU is sufficient
python main.py \
    --model hf-causal \
    --model_args pretrained={OUTPUT_DIR} \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

---

## Can I Run on a Single / Consumer GPU?

**No, not with the current scripts.** The training scripts are hardcoded for 8–16 GPU setups with FSDP. To experiment on limited hardware, you would need to:

### Option A — Gradient Checkpointing + Smaller Batch

Add these flags to reduce activation memory:

```bash
torchrun --nproc_per_node=1 \
    run_clm_llama_lwcd_static_sparse.py \
    ...existing args... \
    --gradient_checkpointing \
    --gradient_accumulation_steps 32 \
    --per_device_train_batch_size 1
```

### Option B — QLoRA-Style Quantization (Recommended for Single GPU)

Integrate bitsandbytes 4-bit quantization:

```bash
pip install bitsandbytes
```

Modify `condensation_trainer.py` to load model in 4-bit:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

Then run with `--fsdp` removed and reduced batch size.

### Option C — Smaller Model First

Start with a model that fits on one GPU:

| Model | Layers | bf16 Memory | Single GPU? |
|---|---|---|---|
| LLaMA-160M (TinyLLaMA) | 16 | ~320 MB | ✅ Yes |
| LLaMA-1B | 22 | ~2 GB | ✅ Yes |
| LLaMA-3B | 26 | ~6 GB | ✅ Yes (consumer) |
| LLaMA-7B | 32 | ~14 GB | ⚠️ 1× A100 only |
| LLaMA-13B | 40 | ~26 GB | ❌ Needs 2+ GPUs |

To use a smaller model, adjust `--total_layer_count` in the training script:
```bash
--total_layer_count 16   # for TinyLLaMA
--total_layer_count 22   # for LLaMA-1B
```

### Option D — Use LoRA Instead of Full Fine-Tuning

```bash
bash run_clm_llama_lora.sh   # LoRA fine-tuning baseline
```

LoRA freezes all model weights and only trains adapter matrices, dramatically reducing memory. You can then apply TrimLLM's layer condensation on top of LoRA.

---

## SLURM HPC Cluster Setup

The project is configured for a specific SLURM cluster (g42cloud, 16× A100 per node):

```bash
# Reserve a node
#SBATCH --reservation=high-profile
#SBATCH --partition=high-profile

# Request GPUs
#SBATCH --gres=gpu:16

# Submit job
sbatch slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_hellaswag_slurm.sh \
    hellaswag decapoda-research/llama-7b-hf 2 0.625 1
```

### Environment Setup (per SLURM script)

```bash
cd /nfs/projects/mbzuai/ext_hao.zhang/lanxiang
source .bashrc
conda activate lanxiang_llm
```

### Resuming Interrupted Runs

```bash
# Edit the resume SLURM script to point to your checkpoint
sbatch slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_resume.sh
```

### Evaluation Only (No Training)

```bash
# After training completes, run only evaluation
sbatch slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_eval_only.sh
```

---

## Summary Table

| Scenario | GPUs | GPU Memory | Viable? |
|---|---|---|---|
| Train LLaMA-7B (original scripts) | 8–16 A100 | 40GB each | ✅ Production |
| Train LLaMA-7B (single GPU, QLoRA) | 1 | 24GB | ⚠️ Possible but slow |
| Train LLaMA-3B (single GPU) | 1 | 16GB | ✅ Yes |
| Train TinyLLaMA (single GPU) | 1 | 8GB | ✅ Yes |
| Evaluate any model | 1 | 16–24GB | ✅ Yes |
| Train with consumer GPU (RTX 3090/4090) | 1 | 24GB | ⚠️ QLoRA only |
