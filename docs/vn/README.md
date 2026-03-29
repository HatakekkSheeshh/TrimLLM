# TrimLLM — Condensation Trọng Lượng Theo Tầng cho Việc Fine-Tuning LLM

## Tổng Quan

TrimLLM là một framework nghiên cứu triển khai **condensation trọng lượng theo tầng** (layerwise weight condensation) — một kỹ thuật nén mô hình có cấu trúc được áp dụng *trong quá trình fine-tuning* các Mô hình Ngôn ngữ Lớn (LLM). Ý tưởng cốt lõi là từ từ nhận diện và đóng băng (thay thế bằng `Identity` không tham số) các tầng transformer dư thừa (MLP và module attention) trong quá trình fine-tuning, qua đó nén mô hình theo chiều sâu mà vẫn giữ được hiệu suất trên tác vụ.

**Bài báo chính:** [TrimLLM: Layerwise Weight Condensation for Efficient LLM Fine-Tuning]

> **Tóm lại:** Bắt đầu với LLaMA 32 tầng, kết thúc với một mô hình có cấu trúc ngắn hơn, được huấn luyện end-to-end — ít tầng hơn và không tốn thêm chi phí inference.

---

## Yêu Cầu Phần Cứng

**⚠️ Huấn luyện yêu cầu 8–16× GPU A100 40GB.** Xem [hardware.md](./hardware.md) để biết chi tiết đầy đủ, tính toán bộ nhớ, và các phương án thay thế cho single GPU (QLoRA, model nhỏ hơn).

---

## Bắt Đầu Nhanh

### 1. Cài Đặt Phụ Thuộc

```bash
conda create -n trimllm python=3.9
conda activate trimllm
pip install -r requirements.txt
```

### 2. Huấn Luyện với TrimLLM (Layerwise Condensation)

```bash
cd scripts
bash run_clm_llama_lwcd_static_sparse_exhausive.sh {task} {model_path} {batch_size} {trial}
```

Ví dụ:
```bash
bash run_clm_llama_lwcd_static_sparse_exhausive.sh hellaswag decapoda-research/llama-7b-hf 8 1
```

### 3. Đánh Giá với lm_eval Harness

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

## Các Tham Số Quan Trọng

| Tham số | Mô tả |
|---|---|
| `--sparsity_ratio` | Tỷ lệ tham số bị đóng băng so với tổng số |
| `--max_budget` | Số tầng tối đa được phép loại bỏ trước khi dừng |
| `--tie_breaker_strategy` | Chiến lược khi các tầng có điểm quan trọng bằng nhau (`naive`, `activation`) |
| `--condense_epoch` | Số epoch cho mỗi bước condensation |
| `--sparse_update` | `static` (mask cố định) hoặc `dynamic` (đánh giá lại mức độ quan trọng mỗi bước) |
| `--static_mask` | Danh sách các tầng được xác định trước để loại bỏ, ví dụ `"[1,5,10]"` |

---

## Quy Trình Hai Bước

### BƯỚC 1 — Huấn Luyện (TrimLLM Condensation)

Fine-tune mô hình bằng `CondensationTrainer`, công cụ này từ từ loại bỏ các tầng transformer trong quá trình huấn luyện.

### BƯỚC 2 — Đánh Giá (EleutherAI lm-evaluation-harness)

Đánh giá mô hình đã nén trên 60+ benchmark downstream bằng bản fork lm-evaluation-harness có sẵn trong repo này.

---

## Cấu Trúc Dự Án

```
TrimLLM/
├── scripts/                    # Mã huấn luyện cốt lõi
│   ├── condensation_trainer.py  # ⭐ TỆP CỐT LÕI, CHỨA ĐÓNG GÓP MỚI
│   ├── modeling_llama.py        # LLaMA đã sửa đổi với skip Identity
│   ├── data.py                  # Tiện ích tải dataset
│   ├── tasks.py                 # Định dạng prompt cho từng task
│   ├── run_clm_llama_lwcd*.py   # Điểm khởi chạy huấn luyện
│   └── *.sh                     # Các script chạy
├── lm_eval/                    # Fork của EleutherAI lm-evaluation-harness
│   ├── evaluator.py             # Engine đánh giá cốt lõi
│   ├── base.py                  # Trừu tượng giao diện LM
│   ├── models/huggingface.py    # Wrapper model HuggingFace
│   ├── tasks/__init__.py        # Registry các task
│   └── tasks/*.py               # Định nghĩa từng task riêng
├── docs/                       # Tài liệu
└── requirements.txt
```

Xem [project_structure.md](./project_structure.md) để biết danh sách đầy đủ các tệp.
