# Hướng Dẫn Cài Đặt

## Môi Trường

```bash
conda create -n trimllm python=3.9
conda activate trimllm
pip install -r requirements.txt
```

Kiểm tra các package chính:
```bash
python -c "import transformers; print(transformers.__version__)"  # Expected: ~4.31
python -c "import torch; print(torch.__version__)"                  # Expected: ~1.13
```

> **Lưu ý:** Nếu gặp lỗi version, xem `scripts/memo.txt` để biết các fix đã biết (ví dụ: kiểm tra phiên bản HF, sửa DeepSpeed, layer dropping trong LLaMA/BERT).

## Quy Trình Hai Bước

### BƯỚC 1 — Huấn Luyện với TrimLLM

```bash
cd scripts
```

Chọn script phù hợp với trường hợp sử dụng của bạn:

| Script | Chế Độ | Khi Nào Dùng |
|---|---|---|
| `run_clm_llama_lwcd_static_sparse_exhausive.sh` | Static sparse + exhaustive | **Khuyến nghị cho hầu hết các trường hợp** |
| `run_clm_llama_lwcd_static_sparse.py` | Static sparse | Tỷ lệ sparsity cố định, không tìm kiếm kiệt quệ |
| `run_clm_llama_lwcd_dynamic_sparse.py` | Dynamic sparse | Điểm quan trọng của tầng được đánh giá lại mỗi epoch |
| `run_clm_llama_lwcd_static_sparse_static_mask.py` | Predefined mask | Bạn biết chính xác tầng nào cần loại bỏ |
| `run_clm_llama_lwcd_exhausive.py` | Exhaustive | Tìm kiếm tất cả ứng viên loại bỏ tầng |
| `run_clm_llama_lwcd_cross_val.py` | Cross-validation | Chọn tầng qua CV |

Cách dùng:
```bash
bash run_clm_llama_lwcd_static_sparse_exhausive.sh {task} {model_path} {batch_size} {trial}
```

Ví dụ:
```bash
bash run_clm_llama_lwcd_static_sparse_exhausive.sh hellaswag decapoda-research/llama-7b-hf 8 1
```

Các tham số chính:
```bash
--sparsity_ratio 0.75          # 75% đóng băng / 25% có thể train
--max_budget 16                # Loại bỏ tối đa 16 tầng
--condense_epoch 1              # 1 epoch cho mỗi bước condensation
--sparse_update static          # hoặc: dynamic
--tie_breaker_strategy naive    # hoặc: activation
--static_mask "[1,5,10]"        # Tùy chọn: danh sách tầng xác định trước
```

### BƯỚC 2 — Đánh Giá với lm_eval Harness

```bash
cd lm_eval/lm-evaluation-harness
pip install -e .
```

Đánh giá model đã huấn luyện:
```bash
python main.py \
    --model hf-causal \
    --model_args pretrained={OUTPUT_DIR} \
    --tasks hellaswag,piqa,arc,sciq \
    --device cuda:0 \
    --batch_size 8
```

Đánh giá **model pretrained gốc** để so sánh baseline:
```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag,piqa,arc,sciq \
    --device cuda:0 \
    --batch_size 8
```

---

## Các Task Được Hỗ Trợ

Dự án hỗ trợ các task downstream sau (qua `scripts/tasks.py` và lm_eval):

| Task | Tên trong lm_eval | Loại |
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

## Cài Đặt SLURM HPC

Nếu chạy trên SLURM cluster, dùng các script trong `slurm/scripts/`:

```bash
sbatch slurm/scripts/run_clm_llama_lwcd_static_sparse_exhaustic_hellaswag.sh
```

Các biến thể SLURM bao gồm: `hellaswag`, `piqa`, `sciq`, `medmcqa`, `race`, `webqs`, `mmlu`, `lex_glue_casehold`, `fiqa`, và nhiều hơn.

Các biến thể:
- `*_resume.sh` — tiếp tục run bị gián đoạn
- `*_eval_only.sh` — chỉ chạy đánh giá (không huấn luyện)

---

## Output Artifacts

Sau khi huấn luyện:
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
│   └── pytorch_model.bin   # hoặc adapter_model.bin cho LoRA
└── eval_results.json       # Metrics mỗi epoch
```

Sau khi đánh giá:
```
lm_eval/results/
└── {model_name}/
    └── {task_name}_results.json
```
