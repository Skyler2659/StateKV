# Architecture

StateKV keeps the active runtime small and separates it from completed experimental programs.
The code below is the supported path for current policy evaluation and R2 research.

## Active components

| Area | Responsibility |
|---|---|
| `statekv/core/` | Backend-independent state, action, risk, and decision contracts. |
| `statekv/backend_mlx.py` | MLX model and cache adapter. |
| `statekv/storage.py` | Atomic artifact writes and stable artifact naming. |
| `statekv/oracle_*`, `cheap_policy*`, `retest_freegen.py` | Recoverable free-generation policies and baseline panel. |
| `statekv/causal_*`, `structured_student.py` | R2 rollouts, causal data, and student models. |
| `statekv/{output_metrics,summary_statistics,text_metrics,dynamic_horizon_metrics}.py` | Shared metrics and statistics. |
| `benchmarks/` | Minimal MLX runner plus torch compatibility types and task adapters. |

The causal runtime is independent of the earlier Fisher and trajectory-model hierarchy.
This keeps the R2 path focused on a temporal model, cache state, and configured evaluation panel.

## Project boundaries

- `results/` holds canonical summaries and local raw artifacts.
- `experiments/` holds frozen manifests and completed-study artifacts.
- `configs/` contains active and historical experiment configurations.
- The active test suite covers cache invariants, output metrics, free-generation policies,
  causal data collection, R2 students, QK decomposition, and temporal metrics.

## Verification

```bash
PYTHONPATH=benchmarks/mlx .venv/bin/python -m pytest -q
PYTHONPATH=benchmarks/mlx .venv/bin/python scripts/smoke_test.py
```

See [Running StateKV](REPRODUCIBILITY.md) for environment setup and model-scale entry points.
