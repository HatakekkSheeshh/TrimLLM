# Thuê GPU Trên Lightning AI

> **Bạn cần gì:** Để chạy huấn luyện TrimLLM (LLaMA-7B với FSDP, bf16), bạn cần **8× A100 40GB** hoặc tương đương. Hướng dẫn này sẽ hướng dẫn bạn thuê phần cứng đó trên [Lightning AI Studio](https://lightning.ai/studio) — không cần phần cứng nào ở máy bạn.

---

## Lightning AI Studio Là Gì?

Lightning AI Studio cung cấp các instance GPU trên cloud cho huấn luyện ML và inference. Các lợi thế chính:

- **Môi trường đã cài sẵn** — PyTorch, CUDA, transformers đã được cấu hình
- **Tính theo giây** — chỉ trả tiền cho thời gian sử dụng
- **Lưu trữ liên tục** — gắn volumes để lưu checkpoints giữa các session
- **CLI + Web UI** — quản lý instance từ terminal hoặc trình duyệt
- **Tương thích FSDP** — hoạt động với distributed training đa GPU ngay từ đầu

---

## Bước 1 — Tạo Tài Khoản

1. Truy cập [lightning.ai/studio](https://lightning.ai/studio)
2. Đăng ký bằng GitHub hoặc email
3. Nạp credits hoặc liên kết phương thức thanh toán

> **Mẹo:** Tài khoản mới thường nhận được credits miễn phí (kiểm tra trang khuyến mãi).

---

## Bước 2 — Tạo Cloud Space (GPU Instance)

### Qua Web UI

1. Đăng nhập tại [lightning.ai/studio](https://lightning.ai/studio)
2. Click **New Project** → **Create Cloud Space**
3. Trong phần **Hardware**, chọn:
   - **Accelerator:** `NVIDIA A100` (hoặc `H100` cho premium)
   - **Count:** `8` (tối thiểu cho huấn luyện TrimLLM)
   - **Type:** `40GB` (A100-40GB) hoặc `80GB` (A100-80GB / H100)
4. Đặt **tên** (ví dụ: `trimllm-training`)
5. Click **Start**

### Qua CLI

```bash
# Cài Lightning AI CLI
pip install lightning

# Đăng nhập
lightning login

# Tạo cloud space với 8× A100
lightning run cloud train.py \
    --accelerator gpu \
    --num_nodes 1 \
    --resources "A100" \
    --count 8 \
    --disk_size 200 \
    --name trimllm-training
```

---

## Bước 3 — Clone TrimLLM Vào Instance

Khi Cloud Space đang chạy, mở terminal (hoặc kết nối qua SSH):

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/TrimLLM.git
cd TrimLLM

# Cài dependencies
pip install -r requirements.txt

# Cài lm_eval harness
cd lm_eval/lm-evaluation-harness
pip install -e .
```

---

## Bước 4 — Upload Model Của Bạn (Nếu Không Có Trên HuggingFace)

Nếu dùng model local (không có trên HuggingFace Hub), gắn storage hoặc upload:

```bash
# Qua Lightning AI SDK
from lightning.ai.core import LightningModule
# Dùng lightning.storage để gắn volume chứa model weights

# Hoặc dùng scp / rsync từ máy local:
# Lấy instance IP từ Lightning AI Studio dashboard
scp -r ./your-model-path/ user@<instance-ip>:/root/TrimLLM/models/
```

---

## Bước 5 — Chạy Huấn Luyện TrimLLM

Từ terminal của instance:

```bash
cd TrimLLM/scripts

# Static sparse + exhaustive search (khuyến nghị)
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

## Bước 6 — Tải Kết Quả Về Máy Local

```bash
# Từ máy local của bạn
scp -r user@<instance-ip>:/root/TrimLLM/outputs/ ./local_outputs/

# Hoặc tải các checkpoint cụ thể
scp -r user@<instance-ip>:/root/TrimLLM/outputs/.../checkpoint-24500 ./checkpoints/
```

---

## Dừng / Tạm Dừng Instance

**Luôn dừng instance khi không sử dụng** — billing chạy khi máy đang bật.

### Qua Web UI
Click **Stop** trên Cloud Space dashboard của bạn.

### Qua CLI
```bash
lightning stop cloud_space --name trimllm-training
```

---

## Tổng Quan Giá

> ⚠️ **Giá bên dưới là ước tính từ ~2025. Luôn kiểm tra giá hiện tại tại [lightning.ai/pricing](https://lightning.ai/pricing) trước khi thuê.**

| GPU | Số lượng | Ước tính / giờ | 1 Trial (~24h) | Ghi chú |
|---|---|---|---|---|
| A100 40GB | 8 | ~$16–24/hr | ~$384–576 | ✅ Khuyến nghị cho TrimLLM |
| A100 80GB | 8 | ~$24–32/hr | ~$576–768 | Thêm headroom cho sequence dài |
| H100 SXM | 8 | ~$40–64/hr | ~$960–1536 | Premium, nhanh hơn nhưng đắt hơn |
| A10G 24GB | 8 | ~$8–16/hr | ~$192–384 | ⚠️ Có thể OOM với LLaMA-7B đầy đủ |

**Ước tính tổng chi phí:**
```
Chi phí = (Giá GPU/giờ) × (số GPU) × (giờ sử dụng)
Ví dụ: $2.50/hr × 8 GPUs × 24 giờ = $480
```

### Giảm Chi Phí

- **`--limit` flag trong lm_eval** — test trên tập con trước khi chạy full eval
- **`--save_total_limit 1`** — không lưu mọi checkpoint, chỉ lưu tốt nhất
- Dùng **A100 40GB** (không phải 80GB) trừ khi bạn bị OOM
- Dừng instance ngay sau khi huấn luyện/eval hoàn thành

---

## Thay Thế: Lambda Labs (So Sánh)

Nếu Lightning AI không phù hợp, [Lambda Labs](https://lambdalabs.com) là lựa chọn thay thế mạnh:

| Tính năng | Lightning AI | Lambda Labs |
|---|---|---|
| Multi-GPU (8× A100) | ✅ Có | ✅ Có |
| PyTorch đã cài sẵn | ✅ Có | ✅ Có |
| Tính theo giây | ✅ Có | ❌ Theo giờ |
| Lưu trữ liên tục | ✅ Có | ✅ Có |
| Hỗ trợ FSDP | ✅ Có | ✅ Có |
| Giao diện | Web + CLI | SSH + Jupyter |
| Theo giờ (A100 40GB × 8) | ~$16–24 | ~$14–20 |

---

## Xử Lý Sự Cố

### Out of Memory (OOM)
```bash
# Giảm batch size
--per_device_train_batch_size 4    # thay vì 8

# Hoặc bật gradient checkpointing
--gradient_checkpointing
```

### Instance Không Khởi Động Được
- Kiểm tra xem region có A100 còn trống không (một số region hết hàng)
- Thử region hoặc loại GPU khác
- Xác minh tài khoản có đủ credits

### SSH Connection Refused
- Đợi 2–3 phút sau khi tạo instance để nó khởi động hoàn toàn
- Kiểm tra instance IP trong Lightning AI dashboard
- Đảm bảo SSH key đã được thêm vào instance

---

## Checklist Tóm Tắt

```
☐ Tạo tài khoản Lightning AI + nạp credits
☐ Spin up Cloud Space: 8× A100 40GB
☐ Clone TrimLLM repo vào instance
☐ pip install -r requirements.txt
☐ pip install -e lm_eval/lm-evaluation-harness
☐ Chạy huấn luyện: torchrun --nproc_per_node=8 ...
☐ Tải outputs về máy local
☐ DỪNG instance ngay lập tức
```
