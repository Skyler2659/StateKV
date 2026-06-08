# L1 KV Cache Eviction Research Framework

This repository studies whether KV cache token importance contains a geometric
irreplaceability component that historical attention saliency does not fully
explain. The working hypothesis is that L1 leverage score can approximate this
component and complement attention-based token selection under aggressive KV
cache compression.

The current repository is a runnable research-framework prototype, not a
finished paper artifact. The priority is reliable quick tests, fair baseline
construction, saved per-sample results, and mechanism-analysis hooks.

## Entry Points

- `benchmark.py` is the legacy toy/demo runner. Keep it for old smoke tests and
  historical comparisons only.
- `scripts/run_benchmark.py` is the main paper-framework benchmark runner.
- `scripts/run_analysis.py` reads saved benchmark results and runs overlap,
  rank correlation, evidence recall, and case-study export.
- `scripts/run_profile.py` profiles latency, throughput, memory, and eviction
  overhead.
- `scripts/run_ablation.py` is reserved for larger sweeps and is still
  experimental.

## Install

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current local venv is Python 3.8.20 with torch 2.4.1+cpu and transformers
4.33.0. `requirements.txt` pins a newer transformers range (`>=4.46.3,<4.47`)
because Qwen2.5/Llama-3.1/Mistral-family experiments need newer model and
tokenizer support. Install the appropriate PyTorch wheel for your hardware from
the official PyTorch index when using CUDA.

## Devices

The main loaders support:

- `--device cpu`
- `--device cuda`
- `--device mps`
- `--device auto`

CPU and MPS paths avoid CUDA-only memory/timing calls. CUDA runs synchronize
timing and record peak CUDA memory. Apple MPS support is experimental and meant
for small smoke tests.

## Quick Start

CPU quick benchmark:

```bash
python scripts/run_benchmark.py --config configs/experiment/quick.yaml --device cpu
```

CUDA quick benchmark:

```bash
python scripts/run_benchmark.py --config configs/experiment/quick.yaml --device cuda
```

Apple MPS quick benchmark:

```bash
python scripts/run_benchmark.py --config configs/experiment/quick.yaml --device mps
```

Quick analysis, replacing `<run_id>` with the created timestamp directory:

```bash
python scripts/run_analysis.py --input results/quick_test/<run_id> --config configs/analysis/basic.yaml
```

Quick profile:

```bash
python scripts/run_profile.py --config configs/experiment/quick.yaml --device cpu
```

No-pytest smoke test:

```bash
python scripts/smoke_test.py
```

Legacy demo:

```bash
python benchmark.py --device cpu --comparison_mode needle --text_source needle
```

## Quick Config

`configs/experiment/quick.yaml` is a full runnable config. It uses local
`sshleifer/tiny-gpt2`, synthetic NIAH samples, budget 16, and these methods:

- `recency`
- `attention`
- `l1_leverage`
- `l2_leverage`
- `attention+l1`

Fragment configs under `configs/model/`, `configs/benchmark/`,
`configs/eviction/`, and `configs/analysis/` are not standalone benchmark
configs. Full experiment configs live under `configs/experiment/`.

## Results

Benchmark runs create timestamped directories:

```text
results/{experiment_name}/{run_id}/
  config.yaml
  results.json
  results.jsonl
  samples/*.json
  selected/*.pt
  scores/*.pt
  analysis/
```

Each sample result records method, budget, model, benchmark, context length,
prompt hash, ground truth, PPL/loss, latency, tokens/s, peak memory, evidence
positions, metadata, selected-token path, and score path. Selected tokens are
saved per layer as original token positions when the eviction wrapper can track
them.

## Current Semantics

KV compression uses `torch.index_select` on the KV sequence dimension. This is
important for GQA/MQA models where KV head count can differ from query head
count. For RoPE models, the framework preserves already-rotated cached keys in
their retained order; it does not make removed historical positions contiguous.
Large RoPE-family experiments still need model-specific smoke tests and, where
necessary, pos-shift validation.

Attention/H2O-style baselines now consume `outputs.attentions` when requested.
H2O/SnapKV/PyramidKV implementations should still be treated as style baselines,
not official reproductions.

## Known Issues / Roadmap

- LongBench standard generation metrics are not fully implemented.
- Official RULER reproduction is not complete; current RULER support is
  synthetic RULER-style coverage.
- Apple MPS is experimental.
- Current transformers 4.33.0 local env cannot load Qwen2.5 tokenizer; install
  `requirements.txt` to use Qwen2.5/Llama-3.1 class models.
- Evidence recall now uses NIAH token spans, but selected-token tracking remains
  a framework mechanism that should be stress-tested on long generation traces.
- Deletion/restoration analyses need a clear distinction between text-level
  ablation and true KV-level counterfactual eviction.
- Existing results from tiny-gpt2 quick tests do not support paper conclusions.
  They only validate framework plumbing.
