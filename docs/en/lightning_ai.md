# Renting GPU on Lightning AI

> **What you need:** To run TrimLLM training (LLaMA-7B with FSDP, bf16), you need **8× A100 40GB** or equivalent. This guide walks you through renting that hardware on [Lightning AI Studio](https://lightning.ai/studio) — no hardware required on your end.

---

## What is Lightning AI Studio?

Lightning AI Studio provides cloud-based GPU instances for ML training and inference. Key advantages:

- **Pre-installed environments** — PyTorch, CUDA, transformers already configured
- **Per-second billing** — pay only for what you use
- **Persistent storage** — attach volumes to save checkpoints between sessions
- **CLI + Web UI** — manage instances from terminal or browser
- **FSDP-compatible** — works with multi-GPU distributed training out of the box

---

## Step 1 — Create Account

1. Go to [lightning.ai/studio](https://lightning.ai/studio)
2. Sign up with GitHub or email
3. Add credits or link a payment method

> **Tip:** New accounts often receive free credits (check the promotions page).

---

## Step 2 — Create a Cloud Space (GPU Instance)

### Via Web UI

1. Log in at [lightning.ai/studio](https://lightning.ai/studio)
2. Click **New Project** → **Create Cloud Space**
3. Under **Hardware**, select:
   - **Accelerator:** `NVIDIA A100` (or `H100` for premium)
   - **Count:** `8` (minimum for TrimLLM training)
   - **Type:** `40GB` (A100-40GB) or `80GB` (A100-80GB / H100)
4. Set a **name** (e.g., `trimllm-training`)
5. Click **Start**

### Via CLI

```bash
# Install Lightning AI CLI
pip install lightning

# Login
lightning login

# Create a cloud space with 8× A100
lightning run cloud train.py \
    --accelerator gpu \
    --num_nodes 1 \
    --resources "A100" \
    --count 8 \
    --disk_size 200 \
    --name trimllm-training
```

---

## Step 3 — Clone TrimLLM Into the Instance

Once the Cloud Space is running, open the terminal (or connect via SSH):

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/TrimLLM.git
cd TrimLLM

# Install dependencies
pip install -r requirements.txt

# Install lm_eval harness
cd lm_eval/lm-evaluation-harness
pip install -e .
```

---

## Step 4 — Upload Your Model (If Not on HuggingFace)

If using a local model (not on HuggingFace Hub), mount your storage or upload:

```bash
# Via Lightning AI SDK
from lightning.ai.core import LightningModule
# Use lightning.storage to attach a volume with your model weights
```

Or use `scp` / `rsync` from your local machine:

```bash
# Get the instance IP from Lightning AI Studio dashboard
scp -r ./your-model-path/ user@<instance-ip>:/root/TrimLLM/models/
```

---

## Step 5 — Run TrimLLM Training

From the instance terminal:

```bash
cd TrimLLM/scripts

# Static sparse + exhaustive search (recommended)
torchrun --nproc_per_node=8 \
    run_clm_llama_lwcd_static_sparse.py \
    --bf16 True \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --total_layer_count 32 \
    --tie_breaker_strategy "activation" \
    --dataset_name hellaswag \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 41 \
    --sparsity_ratio 0.75 \
    --condense_epoch 1 \
    --evaluation_strategy "epoch" \
    --max_budget 16 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --output_dir ./outputs/hellaswag_layerwise_condense
```

---

## Step 6 — Download Results Back to Local

```bash
# From your local machine
scp -r user@<instance-ip>:/root/TrimLLM/outputs/ ./local_outputs/

# Or download specific checkpoints
scp -r user@<instance-ip>:/root/TrimLLM/outputs/.../checkpoint-24500 ./checkpoints/
```

---

## Stopping / Pausing the Instance

**Always stop the instance when not in use** — billing runs while the machine is on.

### Via Web UI
Click **Stop** on your Cloud Space dashboard.

### Via CLI
```bash
lightning stop cloud_space --name trimllm-training
```

---

## Pricing Overview

> ⚠️ **Prices below are estimates from ~2025. Always verify current rates at [lightning.ai/pricing](https://lightning.ai/pricing) before renting.**

| GPU | Count | Approx. Hourly | 1 Trial (~24h) | Notes |
|---|---|---|---|---|
| A100 40GB | 8 | ~$16–24/hr | ~$384–576 | ✅ Recommended for TrimLLM |
| A100 80GB | 8 | ~$24–32/hr | ~$576–768 | More headroom for long sequences |
| H100 SXM | 8 | ~$40–64/hr | ~$960–1536 | Premium, faster but expensive |
| A10G 24GB | 8 | ~$8–16/hr | ~$192–384 | ⚠️ May OOM for full LLaMA-7B |

**To estimate total cost:**
```
Cost = (GPU hourly rate) × (number of GPUs) × (hours used)
Example: $2.50/hr × 8 GPUs × 24 hours = $480
```

### Reducing Cost

- **`--limit` flag in lm_eval** — test on a subset before running full eval
- **`--save_total_limit 1`** — don't save every checkpoint, just the best
- Use **A100 40GB** (not 80GB) unless you hit OOM
- Stop the instance immediately after training/eval completes

---

## Alternative: Lambda Labs (Comparison)

If Lightning AI doesn't suit your needs, [Lambda Labs](https://lambdalabs.com) is a strong alternative:

| Feature | Lightning AI | Lambda Labs |
|---|---|---|
| Multi-GPU (8× A100) | ✅ Yes | ✅ Yes |
| Pre-installed PyTorch | ✅ Yes | ✅ Yes |
| Per-second billing | ✅ Yes | ✅ Hourly |
| Persistent storage | ✅ Yes | ✅ Yes |
| FSDP support | ✅ Yes | ✅ Yes |
| Interface | Web + CLI | SSH + Jupyter |
| Hourly (A100 40GB × 8) | ~$16–24 | ~$14–20 |

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--per_device_train_batch_size 4    # instead of 8

# Or enable gradient checkpointing
--gradient_checkpointing
```

### Instance Won't Start
- Check if the region has available A100s (some regions sell out)
- Try a different region or GPU type
- Verify your account has sufficient credits

### SSH Connection Refused
- Wait 2–3 minutes after creating the instance for it to fully boot
- Check the instance IP in the Lightning AI dashboard
- Ensure your SSH key is added to the instance

---

## Summary Checklist

```
☐ Create Lightning AI account + add credits
☐ Spin up Cloud Space: 8× A100 40GB
☐ Clone TrimLLM repo into instance
☐ pip install -r requirements.txt
☐ pip install -e lm_eval/lm-evaluation-harness
☐ Run training: torchrun --nproc_per_node=8 ...
☐ Download outputs to local machine
☐ STOP the instance immediately
```
