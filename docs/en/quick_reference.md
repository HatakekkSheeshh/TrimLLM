# Quick Reference — Essential Files Only

## Minimal File Set

**Total: ~13 files needed to run the project.**

### Training (`scripts/`)

| File | Role |
|---|---|
| `condensation_trainer.py` | Core — layer-dropping trainer |
| `modeling_llama.py` | Modified LLaMA with Identity skip |
| `data.py` | Dataset loading |
| `tasks.py` | Task prompt formatters |
| `run_clm_llama_lwcd.py` | Main training entry |
| `run_clm_llama_lwcd_static_sparse.py` | Static sparsity variant |

### Evaluation (`lm_eval/`)

| File | Role |
|---|---|
| `evaluator.py` | Core eval engine |
| `base.py` | LM interface |
| `main.py` | CLI entry |
| `models/huggingface.py` | HF model wrapper |

### Config

| File | Role |
|---|---|
| `requirements.txt` | All dependencies |

---

## Everything Else

| Category | Count | What they are |
|---|---|---|
| Ablation run scripts (`run_clm_llama_lr_*.sh`) | ~20 | LR sweep variants — reuse one script |
| SLURM HPC scripts (`slurm/`) | ~14 | HPC cluster launchers |
| Task definitions (`lm_eval/tasks/*.py`) | ~60 | Evaluation benchmarks |
| Legacy files (`*_legacy.py`) | ~2 | Superseded versions |
| Dev tooling (`.gitignore`, `.flake8`, etc.) | ~6 | Linting, coverage, pre-commit |
| Output artifacts (`lm_eval/results/`) | many | Generated JSONs — not code |

---

## Key Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--sparsity_ratio` | — | Fraction frozen (0–1) |
| `--max_budget` | — | Max layers to remove |
| `--condense_epoch` | 1 | Epochs per condensation step |
| `--sparse_update` | `static` | `static` or `dynamic` |
| `--tie_breaker_strategy` | `naive` | `naive` or `activation` |
| `--static_mask` | `None` | Predefined removal list |

---

## Activation Norm Metrics

Supported in `figs/activation/{metric}/`:

```
mean · std · kurtosis · l1_norm · l2_norm · fro_norm
```

Use `fro_norm` (Frobenius norm) as the primary layer importance metric.
