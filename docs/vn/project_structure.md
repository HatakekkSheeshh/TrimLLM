# Cấu Trúc Dự Án & Danh Sách Tệp

## Danh Sách Tệp Đầy Đủ

### Cốt Lõi — Script Huấn Luyện (`scripts/`)

Các script này triển khai đóng góp mới của TrimLLM:

| Tệp | Mục đích |
|---|---|
| `scripts/condensation_trainer.py` | **TỆP CỐT LÕI CHÍNH.** Mở rộng HuggingFace `Trainer` với logic loại bỏ tầng. Chứa: `LMEvalAdaptor` (kết nối trainer với lm_eval harness), đóng băng tầng qua `torch.nn.Identity`, các tham số `condense_epoch`, `max_budget`, `sparsity_ratio`, `tie_breaker_strategy`, tính điểm quan trọng tầng dựa trên activation norm, chế độ sparse tĩnh/động. Output: `figs/activation/{mean,std,kurtosis,l1_norm,l2_norm,fro_norm}/` |
| `scripts/modeling_llama.py` | Kiến trúc LLaMA đã sửa đổi. Quan trọng: `LlamaDecoderLayer.forward()` kiểm tra `isinstance(self.self_attn, Identity)` / `isinstance(self.mlp, Identity)` để bỏ qua các tầng bị loại — loại bỏ cấu trúc thực sự khi inference |
| `scripts/data.py` | Tải dataset: `get_raw_datasets()`, `get_tokenized_datasets()`, `get_lm_datasets()` (causal LM), `process_text2text_datasets()` (QA) |
| `scripts/tasks.py` | Định dạng prompt cho từng task: `PIQA`, `HellaSwag`, `OpenBookQA`, `ARC`, `RACE`, `SciQ`, `WebQs`, `MedMCQA`, `LEX_GLUE_casehold`, `FiQA`. Exports `task_dict`, `map_dataset_name_and_config()`, `LM_EVAL_TASK_NAME_MAPPING` |
| `scripts/run_clm_llama_lwcd.py` | **Điểm khởi chạy huấn luyện chính.** Điều phối: zero-shot eval → khởi tạo `CondensationTrainer` → vòng huấn luyện → eval mỗi epoch → eval kỹ lưỡng qua `evaluator.simple_evaluate()` |
| `scripts/run_clm_llama_lwcd_static_sparse.py` | Condensation với **static sparsity** (tỷ lệ cố định, không đánh giá lại động). `sparse_update='static'` |
| `scripts/run_clm_llama_lwcd_static_sparse_static_mask.py` | Condensation với **mask được xác định trước** (`--static_mask="[1,5,10]"`). `sparse_update='static'` + `static_mask` |
| `scripts/run_clm_llama_lwcd_dynamic_sparse.py` | Condensation với **dynamic sparsity** (điểm quan trọng được đánh giá lại mỗi condensation epoch). `sparse_update='dynamic'` |
| `scripts/run_clm_llama_lwcd_exhausive.py` | **Tìm kiếm kiệt quệ** — thử tất cả ứng viên loại bỏ trong budget mỗi bước |
| `scripts/run_clm_llama_lwcd_cross_val.py` | Biến thể cross-validation cho việc chọn tầng |

### Baseline Fine-Tuning Scripts

| Tệp | Mục đích |
|---|---|
| `scripts/run_clm_llama.sh` | Baseline full fine-tuning (không nén) dùng HF `run_clm.py` chuẩn |
| `scripts/run_clm_llama_lora.sh` | Baseline LoRA fine-tuning dùng `run_clm_lora.py` (PEFT) |
| `scripts/run_clm_llama.py` | Standard HF CLM fine-tuning entry point |
| `scripts/run_clm.py` | Script CLM HuggingFace chuẩn |
| `scripts/run_clm_lora.py` | LoRA fine-tuning dùng thư viện PEFT |

### Evaluation Harness — EleutherAI lm-evaluation-harness (`lm_eval/`)

| Tệp | Mục đích |
|---|---|
| `lm_eval/evaluator.py` | Engine eval cốt lõi: `simple_evaluate()`, `evaluate()` (loglikelihood, greedy_until, loglikelihood_rolling), `make_table()` |
| `lm_eval/base.py` | Interface LM trừu tượng: `BaseLM`, `LM`, `Task`, `MultipleChoiceTask`, `PerplexityTask`, `loglikelihood()`, `loglikelihood_rolling()`, `greedy_until()`, `Request`, `CachingLM` |
| `lm_eval/models/huggingface.py` | `AutoCausalLM`, `AutoSeq2SeqLM` — HuggingFace model wrappers |
| `lm_eval/models/gpt2.py` | `HFLM`, `GPT2LM` — GPT-2/NeoX wrappers |
| `lm_eval/models/anthropic_llms.py` | Anthropic API wrapper |
| `lm_eval/models/__init__.py` | `MODEL_REGISTRY` |
| `lm_eval/main.py` | CLI entry point: `--model`, `--tasks`, `--num_fewshot`, `--batch_size`, `--device` |
| `lm_eval/tasks/__init__.py` | Task registry, exports `ALL_TASKS` |
| `lm_eval/tasks/*.py` (60 tệp) | Định nghĩa từng task: `hellaswag.py`, `piqa.py`, `arc.py`, `sciq.py`, `gsm8k.py`, `lambada.py`, `winogrande.py`, `xcopa.py`, `xnli.py`, `mmlu.py`, `truthfulqa.py`, `superglue.py`, `glue.py`, `bigbench.py`, `wikitext.py`, `pile.py`, `coqa.py`, `drop.py`, `quac.py`, `triviaqa.py`, `webqs.py`, `naturalqs.py`, `squad.py`, `race.py`, `logiqa.py`, `hendrycks_math.py`, `hendrycks_ethics.py`, `hendrycks_test.py`, `lambada_multilingual.py`, `xstorycloze.py`, `translation.py`, `toxigen.py`, v.v. |
| `lm_eval/decontamination/decontaminate.py` | Phát hiện n-gram overlap cho train/test contamination |
| `lm_eval/decontamination/archiver.py`, `janitor.py` | Tiện ích decontamination |
| `lm_eval/datasets/` | Tải dataset tùy chỉnh cho: `coqa`, `drop`, `hendrycks_math`, `hendrycks_ethics`, `logiqa`, `mutual`, `pile`, `quac`, `triviaqa`, `unscramble`, `asdiv`, `headqa`, `sat_analogies` |
| `lm_eval/setup.py` | Package setup (`pip install -e .`) |
| `lm_eval/requirements.txt` | lm_eval dependencies |
| `lm_eval/pile_statistics.json` | Dataset statistics cho Pile subset |

### Run Scripts — Shell Launchers (`scripts/`)

| Tệp | Mục đích |
|---|---|
| `scripts/run_clm_llama_lr_*.sh` | Các biến thể LR sweep (0.05, 0.25, 0.5, 2, 3, 4) cho ablation |
| `scripts/run_clm_llama_lr_0p05_ep*.sh` | Các biến thể epoch (ep5, ep17) tại LR=0.05 |
| `scripts/run_clm_llama_lwcd_cross_val.sh` | Chạy cross-validation |
| `scripts/run_clm_llama_lwcd_dynamic_sparse_exhausive.sh` | Dynamic sparse + exhaustive |
| `scripts/run_clm_llama_lwcd_static_sparse_exhausive.sh` | Static sparse + exhaustive |
| `scripts/run_clm_llama_wikitext_lr_0p05.sh` | WikiText-2 language modeling eval |
| `scripts/run_clm_opt.sh` | Baseline fine-tuning cho OPT model |

### HPC / SLURM Scripts (`slurm/`)

| Tệp | Mục đích |
|---|---|
| `slurm/scripts/run_clm_llama_lwcd_static_sparse_exhausive_*.sh` | 14 biến thể cho HPC cluster (theo task: hellaswag, piqa, sciq, medmcqa, v.v.) |
| `slurm/scripts/*_resume.sh` | Tiếp tục run bị gián đoạn |
| `slurm/scripts/*_eval_only.sh` | Run chỉ đánh giá |

---

## Các Tệp Thừa / Linh Tinh (Có Thể Bỏ Qua)

| Tệp/Thư Mục | Lý Do |
|---|---|
| `scripts/modeling_llama_legacy.py` | Phiên bản cũ của `modeling_llama.py` — đã bị thay thế |
| `scripts/run_clm_llama_legacy.py` | Biến thể script cũ |
| `lm_eval/results/` (toàn bộ cây) | JSON output từ các lần chạy đánh giá — không phải source code |
| `lm_eval/lm_eval.egg-info/` | Artifacts đóng gói Python từ `pip install -e .` |
| `lm_eval/lm_eval/datasets/` subdirs | Custom loaders; hầu hết cũng có trên HuggingFace hub |
| `lm_eval/lm_eval/datasets/bigbench_resources/` | BigBench test data — tự động tải khi chạy |
| `lm_eval/.git/`, `lm_eval/docs/`, `lm_eval/CODEOWNERS` | Dev tooling, docs, git metadata |
| `lm_eval/.gitignore`, `.flake8`, `.pre-commit-config.yaml`, `.coveragerc` | Pre-commit hooks, linting, coverage config |
| `lm_eval/scripts/write_out.py` | Tiện ích debug — ghi prompts/logits ra JSON |
| `lm_eval/ignore.txt` | Word list cho việc lọc ROUGE metric |
| `scripts/clean_training_data/` | Script decontamination Pile — không nằm trong workflow chính |
| `slurm/` | HPC scripts — chỉ cần nếu dùng SLURM cluster |
| `.DS_Store` | File hệ thống macOS — bỏ qua |

---

## Bộ Tệp Tối Thiểu Để Chạy

Nếu bạn muốn tối thiểu để chạy TrimLLM:

```
scripts/
├── condensation_trainer.py
├── modeling_llama.py
├── data.py
├── tasks.py
├── run_clm_llama_lwcd.py
├── run_clm_llama_lwcd_static_sparse.py
└── requirements.txt

lm_eval/
├── evaluator.py
├── base.py
├── main.py
├── models/
│   └── huggingface.py
└── tasks/
    ├── __init__.py
    └── *.py
```

Tất cả các tệp khác là: biến thể ablation, output artifacts, hoặc dev tooling.
