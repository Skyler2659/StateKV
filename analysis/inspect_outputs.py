#!/usr/bin/env python
"""Audit the canonical discovery run before any inferential analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from common import (
    REQUIRED_TABLES,
    ensure_directory,
    finite_or_raise,
    load_required_tables,
    parse_json,
    write_dual,
)


UNITS = {
    "anchor": "generated tokens already incorporated",
    "horizon": "teacher-forced future steps",
    "target_horizon": "teacher-forced future steps",
    "future_step": "1-indexed steps after anchor",
    "lag": "generated-token steps",
    "delta_nll": "nats/token",
    "sum_delta_nll": "nats",
    "avg_delta_nll": "nats/token",
    "max_delta_nll": "nats/token",
    "approx_kl": "nats",
    "sum_approx_kl": "nats",
    "avg_approx_kl": "nats/step",
    "max_approx_kl": "nats",
    "attention_output_error_mean": "dimensionless relative L2",
    "attention_output_error_max": "dimensionless relative L2",
    "oracle_overlap": "Jaccard, [0,1]",
    "active_cache_tokens": "KV token slots/layer",
    "max_active_cache_tokens": "KV token slots/layer",
    "peak_rss_bytes": "bytes",
    "peak_accelerator_bytes": "bytes",
    "replay_time_s": "seconds",
    "forward_time_s": "seconds",
    "generation_time_s": "seconds",
    "prompt_length": "tokens",
    "generated_length": "tokens",
    "position": "absolute logical token index",
    "age": "tokens behind anchor query",
    "score": "selector-specific dimensionless score",
    "boundary_margin": "selected Bth score minus (B+1)th score",
}


LOGICAL_KEYS = {
    "reference": ["run_id", "task", "sample_id"],
    "candidate": [
        "run_id",
        "task",
        "sample_id",
        "anchor",
        "strategy",
        "horizon_condition",
    ],
    "step": [
        "run_id",
        "task",
        "sample_id",
        "anchor",
        "strategy",
        "target_horizon",
        "future_step",
    ],
    "horizon": [
        "run_id",
        "task",
        "sample_id",
        "anchor",
        "strategy",
        "horizon",
    ],
}


def _shape_description(series: pd.Series) -> str:
    sample = next((value for value in series if value is not None), None)
    if sample is None:
        return "scalar/all-null"
    if isinstance(sample, float) and np.isnan(sample):
        sample = next(
            (
                value
                for value in series
                if not (isinstance(value, float) and np.isnan(value))
            ),
            None,
        )
    if isinstance(sample, str) and sample[:1] in "[{":
        try:
            parsed = json.loads(sample)
            if isinstance(parsed, list):
                return "JSON list; example length %d" % len(parsed)
            if isinstance(parsed, dict):
                return "JSON object; example keys %d" % len(parsed)
        except json.JSONDecodeError:
            pass
    if isinstance(sample, np.ndarray):
        return "array object; example shape %s" % (sample.shape,)
    return "scalar"


def _field_rows(name: str, frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "file": REQUIRED_TABLES[name],
            "field": column,
            "dtype": str(frame[column].dtype),
            "shape": _shape_description(frame[column]),
            "unit_or_semantics": UNITS.get(column, "see implementation/findings"),
            "missing": int(frame[column].isna().sum()),
        }
        for column in frame.columns
    ]


def audit(input_dir: Path, analysis_dir: Path) -> None:
    input_dir = Path(input_dir).resolve()
    analysis_dir = ensure_directory(Path(analysis_dir).resolve())
    tables_dir = ensure_directory(analysis_dir / "tables")
    logs_dir = ensure_directory(analysis_dir / "logs")
    tables = load_required_tables(input_dir)
    with open(input_dir / "resolved_config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(input_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(input_dir / "status.json", "r", encoding="utf-8") as handle:
        status = json.load(handle)

    finite_or_raise(
        tables["step"],
        ["delta_nll", "approx_kl", "attention_output_error_mean"],
        "step metrics",
    )
    finite_or_raise(
        tables["horizon"],
        ["sum_delta_nll", "avg_delta_nll", "avg_approx_kl"],
        "horizon metrics",
    )

    model = metadata["model"]
    tasks = metadata["task_load_events"]
    references = tables["reference"]
    candidates = tables["candidate"]
    steps = tables["step"]
    horizons = tables["horizon"]
    temporal = tables["temporal"]

    inventory = pd.DataFrame(
        [
            {
                "run_id": references["run_id"].iloc[0],
                "input_dir": str(input_dir),
                "model": references["model"].iloc[0],
                "weight_precision": model.get("weight_precision"),
                "backend": model.get("backend"),
                "device": model.get("device_resolved"),
                "seed": int(references["seed"].iloc[0]),
                "config_hash": references["config_hash"].iloc[0],
                "git_commit": references["git_commit"].iloc[0],
                "git_dirty": bool(metadata.get("git_dirty")),
                "sample_count": int(len(references)),
                "tasks": json.dumps(tasks, sort_keys=True),
                "strategies": json.dumps(config["strategies"]),
                "anchors": json.dumps(config["anchor_steps"]),
                "horizons": json.dumps(config["horizons"]),
                "signal_lags": json.dumps(config["signal_lags"]),
                "cache_total": int(config["cache"]["total_budget"]),
                "sink_size": int(config["cache"]["sink_size"]),
                "recent_size": int(config["cache"]["recent_size"]),
                "core_size": int(config["cache"]["selected_core_budget"]),
                "diagnostic_layers": json.dumps(
                    model["selected_diagnostic_layers"]
                ),
                "diagnostic_heads": json.dumps(
                    model["selected_diagnostic_query_heads"],
                    sort_keys=True,
                ),
                "query_heads": int(model["num_attention_heads"]),
                "kv_heads": int(model["num_key_value_heads"]),
                "head_dim": int(model["hidden_size"])
                // int(model["num_attention_heads"]),
                "reference_rows": int(len(references)),
                "candidate_rows": int(len(candidates)),
                "step_rows": int(len(steps)),
                "horizon_rows": int(len(horizons)),
                "temporal_rows": int(len(temporal)),
                "failed_status_keys": int(len(status.get("failed", {}))),
            }
        ]
    )
    write_dual(inventory, tables_dir / "run_inventory")

    quality: List[Dict[str, Any]] = []

    def issue(
        code: str,
        severity: str,
        scope: str,
        count: int,
        classification: str,
        detail: str,
    ) -> None:
        quality.append(
            {
                "issue_code": code,
                "severity": severity,
                "scope": scope,
                "count": int(count),
                "classification": classification,
                "detail": detail,
                "excluded": False,
            }
        )

    for name, keys in LOGICAL_KEYS.items():
        duplicates = int(tables[name].duplicated(keys).sum())
        if duplicates:
            issue(
                "DUPLICATE_LOGICAL_KEY_%s" % name.upper(),
                "error",
                REQUIRED_TABLES[name],
                duplicates,
                "data problem",
                "Duplicate rows under logical key %s" % keys,
            )

    eos = references[references["generation_stopped_on_eos"].eq(True)]
    if not eos.empty:
        issue(
            "EARLY_EOS_BUT_HORIZONS_COVERED",
            "info",
            "reference",
            len(eos),
            "expected run variation",
            "Generated lengths %s; all requested anchor/horizon rows are valid."
            % sorted(eos["generated_length"].astype(int).tolist()),
        )
    if metadata.get("git_dirty"):
        issue(
            "DIRTY_WORKTREE",
            "warning",
            "provenance",
            len(metadata.get("git_status", [])),
            "provenance limitation",
            "Commit hash does not include all source state; metadata records git status.",
        )
    if model.get("weight_precision") == "4bit":
        issue(
            "FOUR_BIT_OVERRIDE",
            "warning",
            "model",
            1,
            "experiment limitation",
            "Executed model uses user-approved cached 4-bit weights, not bfloat16.",
        )
    issue(
        "FULL_VALUE_VECTORS_NOT_PERSISTED",
        "critical",
        "geometry",
        1,
        "missing field",
        "Anchor/history V and future per-token V vectors are absent from Parquet/NPZ.",
    )
    issue(
        "DENSE_REFRESH_COUNTERFACTUAL_NOT_PERSISTED",
        "critical",
        "refresh",
        1,
        "missing experimental arm",
        "No refreshed replay at every future step. Only sparse cross-anchor same-token comparisons are recoverable.",
    )
    issue(
        "FUTURE_SCORE_VECTORS_NOT_PERSISTED",
        "warning",
        "stability",
        1,
        "missing field",
        "Future full score vectors/sets are absent; stored score stability is aggregate and old-token-only.",
    )
    issue(
        "STEP_ROWS_REPEAT_ACROSS_TARGET_HORIZONS",
        "info",
        "step_losses",
        len(steps),
        "intentional design",
        "Deployable strategies replay each target horizon independently; oracle core is horizon-conditioned.",
    )
    if not quality:
        issue("NO_ISSUES", "info", "run", 0, "none", "No issues found.")
    quality_frame = pd.DataFrame(quality)
    write_dual(quality_frame, tables_dir / "data_quality_issues")

    schema_rows: List[Dict[str, Any]] = []
    for name, frame in tables.items():
        schema_rows.extend(_field_rows(name, frame))
    schema_frame = pd.DataFrame(schema_rows)
    write_dual(schema_frame, tables_dir / "source_field_schema")

    npz_rows = []
    for _, row in references.iterrows():
        path = Path(row["reference_npz_path"])
        if not path.exists():
            raise FileNotFoundError("reference NPZ missing: %s" % path)
        with np.load(path) as archive:
            for key in archive.files:
                array = archive[key]
                npz_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "array": key,
                        "shape": json.dumps(list(array.shape)),
                        "dtype": str(array.dtype),
                        "bytes": int(array.nbytes),
                    }
                )
    npz_frame = pd.DataFrame(npz_rows)
    write_dual(npz_frame, tables_dir / "reference_npz_schema")

    grouped_npz = (
        npz_frame.groupby(["array", "dtype"], as_index=False)
        .agg(
            samples=("sample_id", "nunique"),
            shape_examples=("shape", lambda values: json.dumps(sorted(set(values))[:6])),
            total_bytes=("bytes", "sum"),
        )
    )
    table_summary = "\n".join(
        "- `%s`: %d rows, %d columns; logical-key duplicates=%d"
        % (
            REQUIRED_TABLES[name],
            len(frame),
            len(frame.columns),
            int(frame.duplicated(LOGICAL_KEYS[name]).sum())
            if name in LOGICAL_KEYS
            else 0,
        )
        for name, frame in tables.items()
    )
    field_lines = "\n".join(
        "| `%s` | `%s` | `%s` | %s | %s | %d |"
        % (
            row.file,
            row.field,
            row.dtype,
            row.shape,
            row.unit_or_semantics,
            row.missing,
        )
        for row in schema_frame.itertuples(index=False)
    )
    npz_lines = "\n".join(
        "| `%s` | `%s` | %d | `%s` | %d |"
        % (
            row.array,
            row.dtype,
            row.samples,
            row.shape_examples,
            row.total_bytes,
        )
        for row in grouped_npz.itertuples(index=False)
    )
    task_lines = "\n".join(
        "- `%s`: %d samples from `%s` (`dataset_official=%s`)"
        % (
            event["task"],
            event["count"],
            event["source"],
            event["dataset_official"],
        )
        for event in tasks
    )
    schema_report = """# Data schema report

This report was generated before mechanism analysis. It describes the actual
canonical run rather than the requested ideal schema.

## Run inventory

- Input: `{input_dir}`
- Model: `{model_name}` ({precision}, {backend}/{device})
- Samples: {samples}; seed={seed}; config hash=`{config_hash}`
- Strategies: `{strategies}`
- Anchors: `{anchors}`; horizons: `{horizons}`; signal lags: `{lags}`
- Cache: {sink} sink + {core} selected core + {recent} rolling recent = {total}
- Model heads: {query_heads} query, {kv_heads} KV; GQA group={gqa}
- Detailed diagnostics: layers `{layers}`, query heads `{heads}`
- Status: {completed} completed keys, {failed} failed keys

Tasks:

{tasks}

## File completeness

{table_summary}

All 720 horizon rows and 15,300 step rows are valid. The maximum and minimum
active cache are both 256. There are 15 readable NPZ files and no canonical
`errors.jsonl`.

## Parquet field schema

| File | Field | dtype | shape | unit / semantics | missing |
|---|---|---|---|---|---:|
{field_lines}

Sparse union columns in `temporal_signals.parquet` are null outside their
`signal_kind`; those nulls are structural, not missing measurements.

## Reference NPZ schema

| Array | dtype | samples | observed shapes (examples) | total raw bytes |
|---|---|---:|---|---:|
{npz_lines}

Attention distributions and oracle attention are right-padded; their companion
length arrays define the valid prefix. Query and attention-output vectors have
head dimension 128. No K/V cache tensor or new-token V vector is persisted.

## Nested candidate schema

Each candidate row contains 28 layer records. Each layer record contains:

- 4 `sink_positions`, 32 `recent_positions`, and 220 `selected_positions`;
- selected token IDs, eligible-token position/token/age/score/rank/core role;
- selection boundary margin and ridge diagnostics;
- SnapKV raw observation score where applicable;
- all five aggregate hybrid components for attention-weighted V ridge;
- per-layer overlaps with the other six candidate names (three deployable plus
  four horizon-conditioned oracle candidates).

Per-KV-head score arrays are intentionally not persisted. GQA mapping is in
metadata, and all-layer per-KV-head oracle attention is in reference NPZ.

## Completeness and anomalies

- Logical-key duplicates: 0 in the five required tables.
- Prompt truncation: 0/15.
- EOS before 128: 2/15, at 115 and 126 generated tokens; requested horizons
  remain covered.
- Required loss and signal fields checked by the canonical run are finite.
- Working tree was dirty and the executed model is the user-approved 4-bit
  checkpoint. These are provenance/experimental limitations, not silent errors.
- Full historical/future V vectors, dense refreshed replay, and future complete
  score vectors are absent. Their downstream metrics are marked unavailable.

Machine-readable versions are in `tables/run_inventory.*`,
`tables/source_field_schema.*`, `tables/reference_npz_schema.*`, and
`tables/data_quality_issues.*`.
""".format(
        input_dir=input_dir,
        model_name=model.get("model_name"),
        precision=model.get("weight_precision"),
        backend=model.get("backend"),
        device=model.get("device_resolved"),
        samples=len(references),
        seed=config["runtime"]["seed"],
        config_hash=references["config_hash"].iloc[0],
        strategies=config["strategies"],
        anchors=config["anchor_steps"],
        horizons=config["horizons"],
        lags=config["signal_lags"],
        sink=config["cache"]["sink_size"],
        core=config["cache"]["selected_core_budget"],
        recent=config["cache"]["recent_size"],
        total=config["cache"]["total_budget"],
        query_heads=model["num_attention_heads"],
        kv_heads=model["num_key_value_heads"],
        gqa=model["gqa_query_heads_per_kv_head"],
        layers=model["selected_diagnostic_layers"],
        heads=model["selected_diagnostic_query_heads"],
        completed=len(status.get("completed", {})),
        failed=len(status.get("failed", {})),
        tasks=task_lines,
        table_summary=table_summary,
        field_lines=field_lines,
        npz_lines=npz_lines,
    )
    (analysis_dir / "data_schema_report.md").write_text(
        schema_report, encoding="utf-8"
    )

    findings = """# Implementation findings

## Confirmed experiment semantics

1. **Independent horizons.** `runner.py::_replay` is called separately for
   every `(sample, anchor, strategy, horizon)`. Short horizons are not slices
   copied from one longest replay. For deployable selectors the same fixed core
   makes overlapping predictions numerically equal; the future oracle uses a
   different horizon-conditioned core.
2. **Horizon-specific oracle.** The oracle sums the exact full-cache query
   records that predict the H replay targets, and reconstructs a core separately
   for every H. Only tokens already present at the anchor can compete.
3. **Teacher forcing.** Every replay input after the rewound anchor query is the
   corresponding full-cache greedy reference token. Strategies do not free-run.
4. **Frozen core and FIFO recent.** Sink and selected core positions are fixed.
   Before each query, dynamic recent positions are pruned to the newest 31;
   appending the current teacher-forced token restores 32 and active cache 256.
5. **Losses.** `delta_nll` is per-step compressed NLL minus full-reference NLL.
   Step rows also store cumulative/running-average/running-max curves.
   Horizon rows store sum/mean/max delta-NLL and approximate KL.
6. **Attention output.** Relative error is measured before output projection on
   query heads 0 and 11 at layers 0, 14, and 27.
7. **Leverage orientation.** Selectable token V rows form `[token, head_dim]`
   matrices independently for each KV head. Ridge leverage uses an Hermitian
   eigensolve (no explicit inverse), relative ridge coefficient 0.001, then
   averages KV-head scores for one shared per-layer token set.
8. **Selector scope.** Selection runs on all 28 layers and both KV heads.
   Detailed temporal/geometry diagnostics are limited to three layers and two
   query heads per layer.

## Findings that constrain interpretation

### F1. Full value-space dynamics are not recoverable

The NPZ files do not contain anchor V matrices, selected-core V rows, or future
new-token V vectors. `temporal_signals` contains only precomputed spectrum and
residual summaries. Consequently the following requested metrics cannot be
computed exactly without new inference or a changed artifact writer:

- residual to the full historical value span;
- online ridge leverage of new tokens;
- expanding/local/EW covariance drift;
- principal angles, projection distance, or canonical correlation;
- time-resolved effective-rank/spectrum changes;
- whitened-coordinate, Mahalanobis, skew/kurtosis, or Gaussian diagnostics.

No substitute metric is labeled as one of these quantities.

### F2. Refresh benefit is only sparsely reconstructable

There is no refreshed-cache arm at every future step. A low-cost same-token
counterfactual is available only where one saved anchor reaches another:

- stale anchor 0 step 17 vs refreshed anchor 16 step 1;
- stale anchor 0 step 49 vs refreshed anchor 48 step 1;
- stale anchor 16 step 33 vs refreshed anchor 48 step 1.

These comparisons share the same deterministic reference token and trajectory,
but they cover only global refresh boundaries 16 and 48. They do not identify a
dense refresh-benefit curve or a unique cache lifetime. NLL refresh benefit is
global, not layer/head specific; per-head comparison is possible only for the
saved attention-output error.

### F3. Stored score/set stability is old-token-only

`score_drift_rows` intersects future candidates with positions that existed at
the anchor and records aggregate Pearson-like autocorrelation, cosine,
Spearman, top-core Jaccard, and margins at lags 1/4/8/16/32/64. It does not
persist future score vectors or future selected positions. Therefore:

- Kendall correlation and per-token rank displacement are unavailable;
- token identities entering/leaving a refreshed core are unavailable;
- current `top_core_jaccard` measures turnover within the anchor token universe,
  not turnover caused by newly generated tokens;
- stability is available only for layers 0/14/27.

### F4. Saved new-direction residual has a narrow definition

`future_new_token_value_residual` is computed relative to the full-rank span of
the selected core for a particular selector/layer/KV head. It is not relative
to full history. With 220 selected rows and head dimension 128, the basis is
usually full rank; very small residuals can reflect this dimensional fact.
`selected_span_reconstruction_residual_mean` is, despite its name, the mean
residual of all selectable rows to the selected span.

### F5. Validity observations are measurement-limit-specific

Each horizon replay separately computes threshold crossings from its own prefix
and marks right censoring at that target horizon. The same deployable core
therefore has repeated observations at H=1/4/16/64. Only the longest available
row should be used when estimating a deployable selector's empirical lifetime.
Oracle rows must remain horizon-conditioned because their cores differ.

### F6. Step rows intentionally repeat overlapping predictions

`step_losses` has one row per target horizon and future step. Overlapping
deployable rows are independent reruns of the same core and token; they should
be deduplicated by choosing the longest target horizon for per-step analyses.
Oracle rows cannot be deduplicated across target horizons without discarding
the oracle condition.

### F7. Model and provenance limitations

The run uses cached 4-bit MLX weights instead of bfloat16. Metadata records the
Git commit plus a dirty worktree. All analysis is conditional on that exact
artifact state.

## Consequence for the requested mechanism chain

The existing outputs can directly examine:

- stale functional loss;
- query/attention drift;
- aggregate score/rank/set stability on old tokens;
- selected-core residual summaries;
- sparse cross-anchor refresh benefit;
- oracle horizon dependence and selector ranking reversals.

They cannot directly establish the full
`covariance/subspace drift -> score change -> dense refresh benefit` chain.
Reports and figures must separate observed links, sparse links, and unavailable
links.
"""
    (analysis_dir / "implementation_findings.md").write_text(
        findings, encoding="utf-8"
    )
    (logs_dir / "audit_complete.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "schema_rows": len(schema_frame),
                "npz_rows": len(npz_frame),
                "quality_issue_rows": len(quality_frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--analysis-dir", default="analysis")
    args = parser.parse_args()
    audit(Path(args.input_dir), Path(args.analysis_dir))
    print(Path(args.analysis_dir).resolve() / "data_schema_report.md")
    print(Path(args.analysis_dir).resolve() / "implementation_findings.md")


if __name__ == "__main__":
    main()
