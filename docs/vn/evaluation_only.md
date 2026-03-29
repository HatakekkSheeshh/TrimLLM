# Đánh Giá Không Cần Huấn Luyện

> **Bạn không cần huấn luyện để đánh giá.** TrimLLM đi kèm với bản đầy đủ EleutherAI lm-evaluation-harness — bạn có thể đánh giá bất kỳ model HuggingFace nào (LLaMA, OPT, GPT-2, v.v.) trên 60+ benchmarks chỉ với một lệnh. Không fine-tuning, không cần GPU cluster, không cần CondensationTrainer.

---

## Zero-Shot Evaluation (Không Huấn Luyện)

Đây là cách nhanh nhất để có benchmark numbers — chỉ cần load model từ HuggingFace và chạy harness.

### 1. Cài đặt harness

```bash
cd lm_eval/lm-evaluation-harness
pip install -e .
```

### 2. Chạy đánh giá

```bash
# Đánh giá một task duy nhất
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

# Đánh giá nhiều tasks cùng lúc
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag,piqa,arc_challenge,openbookqa,sciq \
    --device cuda:0 \
    --batch_size 8

# In kết quả chi tiết đầy đủ
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf \
    --tasks hellaswag \
    --device cuda:0 \
    --output_path ./results/llama7b_hellaswag.json
```

That's it. Harness sẽ tải model, chạy inference trên dataset đánh giá, và xuất ra accuracy/F1/etc.

---

## Những Model Nào Có Thể Đánh Giá?

Bất kỳ model nào trên HuggingFace sử dụng standard causal language modeling architecture:

| Model | Tầng | Memory (fp16) | GPU |
|---|---|---|---|
| `decapoda-research/llama-7b-hf` | 32 | ~14 GB | 1× A10G hoặc tốt hơn |
| `facebook/opt-1.3b` | 24 | ~2.6 GB | 1× consumer GPU ✅ |
| `facebook/opt-6.7b` | 32 | ~13 GB | 1× A10G/A100 |
| `gpt2-medium` | 24 | ~1.5 GB | 1× bất kỳ GPU nào ✅ |
| `gpt2-large` | 36 | ~3 GB | 1× bất kỳ GPU nào ✅ |
| `mistralai/Mistral-7B-v0.1` | 32 | ~14 GB | 1× A10G/A100 |
| `microsoft/phi-2` | 2 | ~1.4 GB | 1× bất kỳ GPU nào ✅ |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 22 | ~2 GB | 1× bất kỳ GPU nào ✅ |

---

## Bộ Nhớ GPU Theo Kích Thước Model

Nếu GPU local không chứa được full model, dùng quantization:

```bash
# 8-bit quantization — giảm VRAM một nửa
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf,load_in_8bit=True \
    --tasks hellaswag \
    --device cuda:0

# 4-bit quantization (QLoRA-style) — ~25% VRAM của fp16
python main.py \
    --model hf-causal \
    --model_args pretrained=decapoda-research/llama-7b-hf,load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16 \
    --tasks hellaswag \
    --device cuda:0
```

**So sánh quantization:**

| Precision | VRAM cho LLaMA-7B | Giảm chất lượng |
|---|---|---|
| fp16 (full) | ~14 GB | Không |
| int8 (8-bit) | ~7 GB | Tối thiểu |
| int4 (4-bit, NF4) | ~3.5 GB | Trung bình |
| fp32 | ~28 GB | Không (thừa) |

---

## Đánh Giá Một Fine-Tuned Checkpoint

Sau khi chạy huấn luyện TrimLLM, checkpoint được lưu tại:

```
outputs/{MODEL}/{TASK}_bs_{bs}/layerwise_condense_sparse_exhausive_sr{sr}_trial_{trial}_max_fro/
```

Đánh giá nó cùng cách:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=./outputs/decapoda-research/llama-7b-hf/piqa_bs_2/layerwise_condense_sparse_exhausive_sr0.625_trial_4_max_fro/checkpoint-24500 \
    --tasks piqa,hellaswag,arc_challenge \
    --device cuda:0 \
    --batch_size 8
```

**Lưu ý:** Vì `modeling_llama.py` (kiến trúc được sửa đổi) không được đăng ký trong HuggingFace model type tiêu chuẩn, bạn phải đánh giá các checkpoint trong thư mục `TrimLLM/scripts/` nơi custom model class có thể import. Hoặc sao chép `modeling_llama.py` vào môi trường đánh giá của bạn và đăng ký nó.

---

## Tất Cả Các Tham Số lm_eval

```bash
python main.py \
    --model hf-causal              # Loại model: hf-causal, hf-seq2seq, gpt2, v.v.
    --model_args pretrained=...     # HuggingFace model path hoặc local checkpoint
                                  # Comma-separated: pretrained=...,load_in_8bit=True
    --tasks hellaswag              # Tên task(s), ngăn cách bằng dấu phẩy
    --num_fewshot 0                # Số in-context examples (0 = zero-shot)
    --batch_size 8                 # Tăng nếu bộ nhớ GPU cho phép
    --device cuda:0                # Thiết bị
    --output_path ./results.json   # Lưu kết quả ra JSON
    --limit 100                    # Chạy chỉ N samples đầu (để test nhanh)
    --verbosity INFO               # DEBUG, INFO, WARNING
```

---

## Các Task Có Sẵn (Chọn Lọc)

| Task | Tên trong lm_eval | Lĩnh vực | Few-shot |
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

Danh sách đầy đủ: `python main.py --tasks help`

---

## Test Nhanh (Single GPU, ~5 phút)

Để xác minh mọi thứ hoạt động trước khi chạy full benchmarks:

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

Chạy 50 samples của HellaSwag trên GPT-2 (~2 GB VRAM, bất kỳ GPU nào).
