# Thuật Toán — TrimLLM Hoạt Động Như Thế Nào

## Trực Quan Cấp Cao

Fine-tuning truyền thống giữ nguyên tất cả các tầng model có thể train được. TrimLLM làm điều khác biệt: **trong quá trình fine-tuning, nó dần loại bỏ các tầng transformer** — thay thế chúng bằng `torch.nn.Identity` (module không tham số, chỉ forward nguyên trạng). Các tầng bị loại bỏ là những tầng mà mô hình "không cần" cho task mục tiêu.

Đây là nén có cấu trúc (không phải weight pruning), nghĩa là:
- Mô hình kết quả **thực sự ngắn hơn** khi inference
- Không cần tính toán thêm để "bỏ qua" các tầng đã loại bỏ
- Kiến trúc thay đổi thực sự, không chỉ magnitude của trọng số

---

## Các Bước Thuật Toán

### Bước 0 — Khởi Tạo

Load một pretrained LLaMA model với `N` tầng transformer (ví dụ: `N=32` cho LLaMA-7B).

### Bước 1 — Baseline Zero-Shot

Đánh giá mô hình trên task mục tiêu **trước bất kỳ fine-tuning nào**. Điều này thiết lập baseline zero-shot để so sánh.

### Bước 2 — Vòng Lặp Condensation

Với mỗi condensation epoch (được điều khiển bởi `--condense_epoch`):

```
FOR mỗi condensation step:
    ├── 2a. Fine-tune các tầng có thể train
    │      (embeddings + K tầng cuối; các tầng bị đóng băng bị bỏ qua qua Identity)
    │
    ├── 2b. Đánh giá sau sub-epoch
    │
    ├── 2c. Tính điểm quan trọng của từng tầng
    │      Chiến lược "naive":       xếp hạng theo vị trí tầng (đầu = ít quan trọng hơn)
    │      Chiến lược "activation":  tính ||activation||_frobenius norm cho mỗi tầng
    │                                xếp hạng tầng, loại bỏ tầng có điểm thấp nhất
    │
    ├── 2d. Loại bỏ tầng có điểm thấp nhất
    │      Thay LlamaDecoderLayer.self_attn  → torch.nn.Identity()
    │      HOẶC
    │      Thay LlamaDecoderLayer.mlp        → torch.nn.Identity()
    │
    └── 2e. Kiểm tra điều kiện dừng
           DỪNG nếu: removed_count >= max_budget
               HOẶC: hiệu suất task giảm quá ngưỡng
```

### Bước 3 — Đánh Giá Cuối Cùng

Chạy EleutherAI lm-evaluation-harness trên 60+ benchmark để tạo ra so sánh kỹ lưỡng với các baseline.

---

## Tính Điểm Quan Trọng Của Tầng

TrimLLM hỗ trợ nhiều chiến lược xếp hạng mức độ quan trọng của từng tầng:

### Dựa Trên Activation (Khuyến Nghị)

Với mỗi tầng transformer `i`, tính **Frobenius norm** của output activations trên một batch dữ liệu huấn luyện:

```
score[i] = ||activation[i]||_Frobenius
```

Các tầng có **activation norm thấp hơn** được coi là ít quan trọng hơn và bị loại bỏ trước.

Các metrics được hỗ trợ: `mean`, `std`, `kurtosis`, `l1_norm`, `l2_norm`, `fro_norm`.

`CondensationTrainer` trong `condensation_trainer.py` lưu activation statistics mỗi epoch trong `figs/activation/{metric}/`.

### Naive (Dựa Trên Vị Trí)

Đơn giản loại bỏ tầng từ đầu hoặc cuối mạng — không cần activation statistics. Hữu ích như baseline.

---

## Cách Xử Lý Tầng Bị Loại Bỏ

`modeling_llama.py` sửa đổi `LlamaDecoderLayer.forward()` để phát hiện các module `Identity`:

```python
def forward(self, hidden_states, ...):
    if isinstance(self.self_attn, torch.nn.Identity):
        # Bỏ qua attention hoàn toàn — giảm KV cache, FLOPs
        ...
    if isinstance(self.mlp, torch.nn.Identity):
        # Bỏ qua MLP hoàn toàn
        ...
    # Normal residual-connected forward pass
```

Điều này có nghĩa là các tầng bị loại bỏ là **true no-ops** khi inference — không tính attention patterns, không lưu MLP activations vào KV cache.

---

## Các Chế Độ Sparsity

| Chế Độ | `--sparse_update` | Hành Vi |
|---|---|---|
| **Static Sparse** | `static` | Sparsity ratio cố định; điểm quan trọng tính một lần lúc bắt đầu |
| **Dynamic Sparse** | `dynamic` | Điểm quan trọng được đánh giá lại mỗi condensation epoch |
| **Exhaustive Search** | `static` + `--static_mask` | Thử tất cả ứng viên loại bỏ trong budget |

### Static Sparsity (`sparsity_ratio`)

```bash
--sparsity_ratio 0.75   # 75% params bị đóng băng, 25% có thể train
```

Kiểm soát phần trăm model bị đóng băng so với có thể train tại bất kỳ thời điểm nào.

### Dynamic Sparsity

Mức độ quan trọng của tầng thay đổi khi huấn luyện tiến triển — mô hình liên tục đánh giá lại tầng nào ít hữu ích nhất.

---

## Tie-Breaker Strategy

Khi hai hoặc nhiều tầng có điểm quan trọng bằng nhau, `--tie_breaker_strategy` xác định tầng nào bị loại:

- `naive` — loại theo vị trí (đầu/cuối trước)
- `activation` — dùng secondary activation metric để phân định

---

## Cấu Trúc Output

Các lần chạy huấn luyện tạo ra:

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
│           └── layer_scores_per_epoch.png   # Visualization
├── checkpoint/                              # Best model checkpoint
└── eval_results.json                        # Per-epoch evaluation metrics
```
