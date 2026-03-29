# Yêu Cầu Phần Cứng

## Huấn Luyện

Việc huấn luyện LLaMA-7B (32 tầng) với TrimLLM đòi hỏi tài nguyên GPU lớn vì **full-parameter fine-tuning + FSDP sharding + bf16 precision**.

| Thành phần | Yêu cầu |
|---|---|
| **GPU** | **Tối thiểu 8** (local) / **16** (SLURM production scripts) |
| **Bộ nhớ GPU** | ~40GB mỗi GPU (A100-40GB khuyến nghị) |
| **Compute GPU** | Ampere+ architecture (A100, RTX 3090, RTX 4090, H100) |
| **Precision** | `--bf16 True` (Ampere+) hoặc `--fp16 True` (Volta+) |
| **CPU** | ~96 CPUs per node (SLURM config) |
| **RAM** | Cao; node chạy ở chế độ `--exclusive` |
| **Interconnect** | NVLink / NVSwitch khuyến nghị cho multi-GPU FSDP |

### Tính Toán Bộ Nhớ GPU (LLaMA-7B, 32 tầng)

```
Full model (bf16):          ≈ 14 GB
Optimizer states (AdamW):   ≈ 28 GB   (3× model params cho Adam)
Gradients (bf16):           ≈ 14 GB
Activations (per batch):    ≈  4–8 GB  (sequence length, batch size dependent)
─────────────────────────────────────
Total (no sharding):        ≈ 60–64 GB  →  không thể trên bất kỳ GPU đơn nào

Với FSDP across 8 GPUs:
  Per GPU:  14 / 8  +  28 / 8  +  14 / 8  +  activations
           ≈ 1.75   +  3.5    +  1.75   +  0.5–1
           ≈ 7.5–8 GB per GPU  →  A100-40GB thoải mái
```

### Những Gì Scripts Thực Sự Sử Dụng

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

## Đánh Giá

Đánh giá chỉ là **inference** và ít tốn kém tài nguyên hơn nhiều so với huấn luyện.

| Thành phần | Yêu cầu |
|---|---|
| **GPU** | **Tối thiểu 1 GPU** |
| **Bộ nhớ GPU** | ~16–24 GB (tùy model size và batch size) |
| **Precision** | bf16 hoặc fp16 (tự động phát hiện) |

```bash
# Một GPU là đủ
python main.py \
    --model hf-causal \
    --model_args pretrained={OUTPUT_DIR} \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

---

## Tôi Có Thể Chạy Trên Một / GPU Thường Không?

**Không, với các script hiện tại.** Các script huấn luyện được hardcode cho cài đặt 8–16 GPU với FSDP. Để thử nghiệm trên phần cứng hạn chế, bạn cần:

### Phương Án A — Gradient Checkpointing + Batch Nhỏ Hơn

Thêm các flags này để giảm activation memory:

```bash
torchrun --nproc_per_node=1 \
    run_clm_llama_lwcd_static_sparse.py \
    ...existing args... \
    --gradient_checkpointing \
    --gradient_accumulation_steps 32 \
    --per_device_train_batch_size 1
```

### Phương Án B — Quantization QLoRA-Style (Khuyến Nghị Cho Single GPU)

Tích hợp bitsandbytes 4-bit quantization:

```bash
pip install bitsandbytes
```

Sửa `condensation_trainer.py` để load model ở 4-bit:
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

Sau đó chạy với `--fsdp` bị loại bỏ và batch size giảm.

### Phương Án C — Bắt Đầu Với Model Nhỏ Hơn

Bắt đầu với model vừa trong một GPU:

| Model | Tầng | bf16 Memory | Một GPU? |
|---|---|---|---|
| LLaMA-160M (TinyLLaMA) | 16 | ~320 MB | ✅ Có |
| LLaMA-1B | 22 | ~2 GB | ✅ Có |
| LLaMA-3B | 26 | ~6 GB | ✅ Có (consumer) |
| LLaMA-7B | 32 | ~14 GB | ⚠️ Chỉ 1× A100 |
| LLaMA-13B | 40 | ~26 GB | ❌ Cần 2+ GPU |

Để dùng model nhỏ hơn, điều chỉnh `--total_layer_count` trong script huấn luyện:
```bash
--total_layer_count 16   # cho TinyLLaMA
--total_layer_count 22   # cho LLaMA-1B
```

### Phương Án D — Dùng LoRA Thay Vì Full Fine-Tuning

```bash
bash run_clm_llama_lora.sh   # LoRA fine-tuning baseline
```

LoRA đóng băng tất cả trọng số model và chỉ huấn luyện các ma trận adapter, giảm đáng kể bộ nhớ. Bạn có thể áp dụng layer condensation của TrimLLM lên trên LoRA.

---

## Cài Đặt SLURM HPC Cluster

Dự án được cấu hình cho một SLURM cluster cụ thể (g42cloud, 16× A100 mỗi node):

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

### Cài Đặt Môi Trường (per SLURM script)

```bash
cd /nfs/projects/mbzuai/ext_hao.zhang/lanxiang
source .bashrc
conda activate lanxiang_llm
```

### Tiếp Tục Run Bị Gián Đoạn

```bash
# Chỉnh sửa resume SLURM script để trỏ đến checkpoint của bạn
sbatch slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_resume.sh
```

### Chỉ Đánh Giá (Không Huấn Luyện)

```bash
# Sau khi huấn luyện xong, chỉ chạy đánh giá
sbatch slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_eval_only.sh
```

---

## Bảng Tổng Hợp

| Kịch bản | GPU | Bộ nhớ GPU | Khả thi? |
|---|---|---|---|
| Train LLaMA-7B (scripts gốc) | 8–16 A100 | 40GB mỗi cái | ✅ Production |
| Train LLaMA-7B (single GPU, QLoRA) | 1 | 24GB | ⚠️ Có thể nhưng chậm |
| Train LLaMA-3B (single GPU) | 1 | 16GB | ✅ Có |
| Train TinyLLaMA (single GPU) | 1 | 8GB | ✅ Có |
| Đánh giá bất kỳ model nào | 1 | 16–24GB | ✅ Có |
| Train với GPU consumer (RTX 3090/4090) | 1 | 24GB | ⚠️ Chỉ QLoRA |
