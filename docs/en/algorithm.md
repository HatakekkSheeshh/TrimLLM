# Algorithm — How TrimLLM Works

## High-Level Intuition

Traditional fine-tuning keeps all model layers trainable. TrimLLM does something different: **as fine-tuning progresses, it gradually removes transformer layers** — replacing them with `torch.nn.Identity` (zero-parameter pass-through modules). The removed layers are the ones the model "doesn't need" for the target task.

This is structural compression (not weight pruning), meaning:
- The resulting model is **genuinely shorter** at inference time
- No extra computation is needed to "skip" removed layers
- The architecture itself changes, not just weight magnitudes

---

## Algorithm Steps

### Step 0 — Initialization

Load a pretrained LLaMA model with `N` transformer layers (e.g., `N=32` for LLaMA-7B).

### Step 1 — Zero-Shot Baseline Evaluation

Evaluate the model on the target task **before any fine-tuning**. This establishes the zero-shot baseline for comparison.

### Step 2 — Condensation Loop

For each condensation epoch (controlled by `--condense_epoch`):

```
FOR each condensation step:
    ├── 2a. Fine-tune trainable layers
    │      (embeddings + last K layers; frozen layers skipped via Identity)
    │
    ├── 2b. Evaluate after sub-epoch
    │
    ├── 2c. Score layer importance
    │      Strategy "naive":      rank by layer position (front = less important)
    │      Strategy "activation": compute ||activation||_frobenius norm per layer
    │                              rank layers, drop the lowest-scoring one
    │
    ├── 2d. Remove the lowest-importance layer
    │      Replace LlamaDecoderLayer.self_attn  → torch.nn.Identity()
    │      OR
    │      Replace LlamaDecoderLayer.mlp        → torch.nn.Identity()
    │
    └── 2e. Check stopping criteria
           STOP if: removed_count >= max_budget
               OR: task performance degrades beyond threshold
```

### Step 3 — Final Evaluation

Run the EleutherAI lm-evaluation-harness on 60+ benchmarks to produce a thorough comparison against baselines.

---

## Layer Importance Scoring

TrimLLM supports multiple strategies for ranking layer importance:

### Activation-Based (Recommended)

For each transformer layer `i`, compute the **Frobenius norm** of its output activations over a batch of training data:

```
score[i] = ||activation[i]||_Frobenius
```

Layers with **lower** activation norms are considered less important and removed first.

Supported metrics: `mean`, `std`, `kurtosis`, `l1_norm`, `l2_norm`, `fro_norm`.

The `CondensationTrainer` in `condensation_trainer.py` stores per-epoch activation statistics in `figs/activation/{metric}/`.

### Naive (Position-Based)

Simply removes layers from the front or back of the network — no activation statistics needed. Useful as a baseline.

---

## How Removed Layers Are Handled

`modeling_llama.py` modifies `LlamaDecoderLayer.forward()` to detect `Identity` modules:

```python
def forward(self, hidden_states, ...):
    if isinstance(self.self_attn, torch.nn.Identity):
        # Skip attention entirely — reduce KV cache, FLOPs
        ...
    if isinstance(self.mlp, torch.nn.Identity):
        # Skip MLP entirely
        ...
    # Normal residual-connected forward pass
```

This means removed layers are **true no-ops** at inference time — no attention patterns computed, no MLP activations stored in KV cache.

---

## Sparsity Modes

| Mode | `--sparse_update` | Behavior |
|---|---|---|
| **Static Sparse** | `static` | Sparsity ratio fixed; layer importance scored once at start |
| **Dynamic Sparse** | `dynamic` | Layer importance re-evaluated each condensation epoch |
| **Exhaustive Search** | `static` + `--static_mask` | Tries all layer removal candidates within budget |

### Static Sparsity (`sparsity_ratio`)

```bash
--sparsity_ratio 0.75   # 75% of params frozen, 25% trainable
```

Controls what fraction of the model is frozen vs. trainable at any given time.

### Dynamic Sparsity

Layer importance changes as training progresses — the model continuously re-evaluates which layer is least useful.

---

## Tie-Breaker Strategy

When two or more layers have equal importance scores, `--tie_breaker_strategy` determines which to remove:

- `naive` — remove by position (front/back first)
- `activation` — use a secondary activation metric to break ties

---

## Output Structure

Training runs produce:

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
