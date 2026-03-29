# Tham Khảo Nhanh — Chỉ Các Tệp Cần Thiết

## Bộ Tệp Tối Thiểu

**Tổng: ~13 tệp cần thiết để chạy dự án.**

### Huấn Luyện (`scripts/`)

| Tệp | Vai trò |
|---|---|
| `condensation_trainer.py` | Cốt lõi — trainer loại bỏ tầng |
| `modeling_llama.py` | LLaMA đã sửa đổi với Identity skip |
| `data.py` | Tải dataset |
| `tasks.py` | Định dạng prompt cho từng task |
| `run_clm_llama_lwcd.py` | Điểm khởi chạy huấn luyện chính |
| `run_clm_llama_lwcd_static_sparse.py` | Biến thể static sparsity |

### Đánh Giá (`lm_eval/`)

| Tệp | Vai trò |
|---|---|
| `evaluator.py` | Engine đánh giá cốt lõi |
| `base.py` | Interface LM |
| `main.py` | CLI entry |
| `models/huggingface.py` | HF model wrapper |

### Cấu Hình

| Tệp | Vai trò |
|---|---|
| `requirements.txt` | Tất cả dependencies |

---

## Tất Cả Các Tệp Khác

| Danh mục | Số lượng | Ý nghĩa |
|---|---|---|
| Ablation run scripts (`run_clm_llama_lr_*.sh`) | ~20 | Các biến thể LR sweep — tái sử dụng một script |
| SLURM HPC scripts (`slurm/`) | ~14 | Launcher cho HPC cluster |
| Định nghĩa task (`lm_eval/tasks/*.py`) | ~60 | Các benchmark đánh giá |
| Legacy files (`*_legacy.py`) | ~2 | Các phiên bản đã thay thế |
| Dev tooling (`.gitignore`, `.flake8`, etc.) | ~6 | Linting, coverage, pre-commit |
| Output artifacts (`lm_eval/results/`) | nhiều | JSON được sinh ra — không phải code |

---

## Các Hyperparameter Quan Trọng

| Flag | Mặc định | Mô tả |
|---|---|---|
| `--sparsity_ratio` | — | Tỷ lệ đóng băng (0–1) |
| `--max_budget` | — | Số tầng tối đa được loại bỏ |
| `--condense_epoch` | 1 | Số epoch mỗi bước condensation |
| `--sparse_update` | `static` | `static` hoặc `dynamic` |
| `--tie_breaker_strategy` | `naive` | `naive` hoặc `activation` |
| `--static_mask` | `None` | Danh sách tầng xác định trước |

---

## Các Activation Norm Metrics

Được hỗ trợ trong `figs/activation/{metric}/`:

```
mean · std · kurtosis · l1_norm · l2_norm · fro_norm
```

Dùng `fro_norm` (Frobenius norm) làm metric chính cho điểm quan trọng của tầng.
