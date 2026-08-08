"""Model-backed arithmetic profile for the rolling direct cache policy."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import load_discovery_config
from statekv.direct_coreset_pilot import _record_attention
from statekv.direct_policy_replay import _shared_selection
from statekv.direct_policy_runtime import (
    RollingDirectPolicy,
    batch_blend_score,
    contribution_token_score,
    stable_topk_rows,
)
from statekv.functional_probe import _condition_cache
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks


def _timed(call: Callable[[], Any], repeats: int) -> Tuple[Any, np.ndarray]:
    call()
    elapsed = []
    result: Any = None
    for _ in range(int(repeats)):
        started = time.perf_counter_ns()
        result = call()
        elapsed.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, np.asarray(elapsed, dtype=np.float64)


def _eligible_rows(
    reference: Any, anchor: int, sink_size: int, recent_size: int
) -> np.ndarray:
    positions = [
        int(value)
        for value in reference.anchors[int(anchor)].position_maps[0].tolist()
    ]
    _, _, eligible_positions = mandatory_and_eligible(
        positions, int(sink_size), int(recent_size)
    )
    row_by_position = {position: row for row, position in enumerate(positions)}
    return np.asarray(
        [row_by_position[position] for position in eligible_positions],
        dtype=np.int64,
    )


def _payload(
    reference: Any,
    anchor: int,
    layers: Sequence[int],
    window: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    state = reference.anchors[int(anchor)]
    records = reference.query_records[max(0, int(anchor) - int(window)) : int(anchor)]
    if len(records) != int(window):
        raise RuntimeError("anchor does not have the configured history window")
    attention: Dict[int, np.ndarray] = {}
    values: Dict[int, np.ndarray] = {}
    for layer in layers:
        value = (
            state.values[int(layer)][0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        values[int(layer)] = value
        attention[int(layer)] = np.stack(
            [
                _record_attention(record, int(layer), int(value.shape[1])).astype(
                    np.float32
                )
                for record in records
            ],
            axis=0,
        )
    return attention, values


def _profile_unit(
    attention: Dict[int, np.ndarray],
    values: Dict[int, np.ndarray],
    eligible_rows: np.ndarray,
    layers: Sequence[int],
    window: int,
    weight: float,
    core_budget: int,
    repeats: int,
) -> Dict[str, float]:
    def batch_call() -> Tuple[np.ndarray, np.ndarray]:
        score = batch_blend_score(
            attention,
            values,
            eligible_rows,
            contribution_weight=weight,
            dtype=np.float32,
        )
        return score, stable_topk_rows(score, eligible_rows, core_budget)

    (batch_score, batch_rows), batch_ms = _timed(batch_call, repeats)

    update_measurements: List[float] = []
    completed: RollingDirectPolicy | None = None
    for _ in range(int(repeats)):
        rolling = RollingDirectPolicy(
            layers=layers,
            window=window,
            contribution_weight=weight,
            dtype=np.float32,
        )
        started = time.perf_counter_ns()
        for query in range(int(window)):
            for layer in layers:
                rolling.update_layer(
                    int(layer), attention[int(layer)][query], values[int(layer)]
                )
        elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
        update_measurements.append(elapsed_ms / int(window))
        completed = rolling
    if completed is None:
        raise RuntimeError("runtime profile did not build a rolling state")

    def refresh_call() -> Tuple[np.ndarray, np.ndarray]:
        score = completed.score(eligible_rows)
        return score, completed.select(eligible_rows, core_budget)

    (rolling_score, rolling_rows), refresh_ms = _timed(refresh_call, repeats)
    intersection = len(set(batch_rows.tolist()) & set(rolling_rows.tolist()))
    union = len(set(batch_rows.tolist()) | set(rolling_rows.tolist()))
    token_count = int(next(iter(values.values())).shape[1])
    return {
        "token_count": token_count,
        "eligible_token_count": int(eligible_rows.size),
        "batch_refresh_median_ms": float(np.median(batch_ms)),
        "batch_refresh_p95_ms": float(np.quantile(batch_ms, 0.95)),
        "rolling_update_per_token_median_ms": float(
            np.median(update_measurements)
        ),
        "rolling_update_per_token_p95_ms": float(
            np.quantile(update_measurements, 0.95)
        ),
        "rolling_refresh_median_ms": float(np.median(refresh_ms)),
        "rolling_refresh_p95_ms": float(np.quantile(refresh_ms, 0.95)),
        "score_max_absolute_error": float(
            np.max(np.abs(batch_score - rolling_score))
        ),
        "selection_jaccard": float(intersection / max(union, 1)),
        "rolling_working_set_bytes": int(completed.working_set_bytes),
        "rolling_bytes_per_context_token": float(
            completed.working_set_bytes / token_count
        ),
    }


def _capture_profile(
    runner: CandidatePullbackRunner,
    reference: Any,
    anchor: int,
    selection: Any,
    layers: Sequence[int],
    total_budget: int,
    recent_size: int,
    repeats: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Measure compressed-decode capture modes, including CPU fallback transfer."""

    if runner.cfg.model.backend != "mlx":
        return pd.DataFrame(), {}
    import mlx.core as mx

    cache_cfg = _condition_cache(runner.cfg, total_budget, recent_size)
    token_id = int(reference.anchors[int(anchor)].query_token_id)

    def one(mode: str) -> Dict[str, Any]:
        state, _ = runner.model.state_from_anchor(
            reference.anchors[int(anchor)], selection, cache_config=cache_cfg
        )
        attention_state = runner.model.runner.attention_state
        if mode in {"no_capture", "minimal_direct_capture"}:
            attention_state["enabled"] = False
            attention_state["temporal_record_diagnostics"] = False
            attention_state["temporal_record_direct_policy"] = (
                mode == "minimal_direct_capture"
            )
            attention_state["direct_policy_attention"] = {}
        transfer_ms = 0.0
        score_ms = 0.0
        try:
            started = time.perf_counter_ns()
            if mode == "current_research_diagnostics":
                logits, _, _ = runner.model.forward_one(
                    state, token_id, capture_attention=True
                )
                logits_np = logits.numpy().copy()
            else:
                logits_mx = runner.model.runner.model(
                    mx.array([[token_id]]), cache=state.cache
                )
                direct = attention_state.get("direct_policy_attention", {})
                mx.eval(logits_mx, *direct.values())
                logits_np = np.asarray(logits_mx[0, -1, :]).copy()
            forward_ms = (time.perf_counter_ns() - started) / 1.0e6
            direct_layer_count = 0
            if mode == "minimal_direct_capture":
                started = time.perf_counter_ns()
                copied_attention = {
                    int(layer): np.asarray(value).astype(np.float32).copy()
                    for layer, value in attention_state[
                        "direct_policy_attention"
                    ].items()
                }
                copied_values = {
                    int(layer): np.asarray(
                        state.cache[int(layer)].values[
                            0,
                            :,
                            : int(state.cache[int(layer)].offset),
                            :,
                        ]
                    )
                    .astype(np.float32)
                    .copy()
                    for layer in layers
                }
                transfer_ms = (time.perf_counter_ns() - started) / 1.0e6
                direct_layer_count = len(copied_attention)
                started = time.perf_counter_ns()
                for layer in layers:
                    contribution_token_score(
                        copied_attention[int(layer)], copied_values[int(layer)]
                    )
                score_ms = (time.perf_counter_ns() - started) / 1.0e6
            return {
                "mode": mode,
                "forward_ms": float(forward_ms),
                "transfer_ms": float(transfer_ms),
                "one_query_contribution_ms": float(score_ms),
                "direct_layer_count": int(direct_layer_count),
                "logits": logits_np,
            }
        finally:
            runner.model.release(state)

    modes = [
        "no_capture",
        "minimal_direct_capture",
        "current_research_diagnostics",
    ]
    for mode in modes:
        one(mode)
    raw: List[Dict[str, Any]] = []
    logits_by_mode: Dict[str, np.ndarray] = {}
    for mode in modes:
        for repeat in range(int(repeats)):
            result = one(mode)
            logits_by_mode[mode] = result.pop("logits")
            raw.append({"repeat": repeat, **result})
    raw_frame = pd.DataFrame(raw)
    records = []
    baseline_logits = logits_by_mode["no_capture"]
    for mode, current in raw_frame.groupby("mode", sort=False):
        records.append(
            {
                "mode": mode,
                "repeats": int(len(current)),
                "median_forward_ms": float(current["forward_ms"].median()),
                "p95_forward_ms": float(current["forward_ms"].quantile(0.95)),
                "median_transfer_ms": float(current["transfer_ms"].median()),
                "median_one_query_contribution_ms": float(
                    current["one_query_contribution_ms"].median()
                ),
                "direct_layer_count": int(current["direct_layer_count"].max()),
                "maximum_logit_absolute_difference_vs_no_capture": float(
                    np.max(np.abs(logits_by_mode[mode] - baseline_logits))
                ),
            }
        )
    summary = pd.DataFrame(records)
    indexed = summary.set_index("mode")
    no_capture_ms = float(indexed.loc["no_capture", "median_forward_ms"])
    minimal_ms = float(
        indexed.loc["minimal_direct_capture", "median_forward_ms"]
    )
    research_ms = float(
        indexed.loc["current_research_diagnostics", "median_forward_ms"]
    )
    derived = {
        "no_capture_median_forward_ms": no_capture_ms,
        "minimal_direct_capture_median_forward_ms": minimal_ms,
        "minimal_direct_capture_overhead_ms": minimal_ms - no_capture_ms,
        "minimal_direct_capture_overhead_fraction": (
            (minimal_ms - no_capture_ms) / no_capture_ms
        ),
        "current_research_diagnostics_median_forward_ms": research_ms,
        "minimal_vs_research_forward_time_reduction_fraction": (
            (research_ms - minimal_ms) / research_ms
        ),
        "minimal_cpu_transfer_median_ms": float(
            indexed.loc["minimal_direct_capture", "median_transfer_ms"]
        ),
        "minimal_cpu_one_query_contribution_median_ms": float(
            indexed.loc[
                "minimal_direct_capture", "median_one_query_contribution_ms"
            ]
        ),
        "maximum_logit_absolute_difference": float(
            summary["maximum_logit_absolute_difference_vs_no_capture"].max()
        ),
    }
    return summary, derived


def run_direct_policy_runtime_profile(
    config_path: Path, repository_root: Path
) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    if "task_overrides" in config:
        cfg.tasks = dict(config["task_overrides"])
    if "data_seed" in config:
        cfg.runtime.seed = int(config["data_seed"])
    if "runtime_run_id" in config:
        cfg.runtime.run_id = str(config["runtime_run_id"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)

    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    layers = [int(value) for value in config["diagnostic_layers"]]
    window = int(config.get("scenario_window", 4))
    weight = float(config.get("contribution_weight", 0.25))
    core_budget = int(config["core_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    repeats = int(config.get("timing_repeats", 30))
    capture_repeats = int(config.get("capture_timing_repeats", 0))
    capture_full_repeats = int(config.get("capture_full_timing_repeats", 0))
    capture_anchor = int(config.get("capture_anchor", anchors[-1]))
    refresh_intervals = [
        int(value) for value in config.get("refresh_intervals", [4, 8, 16, 32])
    ]

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected = [sample for sample in samples if str(sample.sample_id) in sample_ids]
    if {str(sample.sample_id) for sample in selected} != sample_ids:
        raise RuntimeError("configured profile samples were not loaded")

    records: List[Dict[str, Any]] = []
    capture_rows = pd.DataFrame()
    capture_summary: Dict[str, Any] = {}
    started = time.perf_counter()
    runner.model.load()
    try:
        for sample in selected:
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            try:
                for anchor in anchors:
                    attention, values = _payload(
                        reference, anchor, layers, window
                    )
                    eligible = _eligible_rows(
                        reference, anchor, sink_size, recent_size
                    )
                    records.append(
                        {
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "anchor": int(anchor),
                            **_profile_unit(
                                attention,
                                values,
                                eligible,
                                layers,
                                window,
                                weight,
                                core_budget,
                                repeats,
                            ),
                        }
                    )
                    if (
                        capture_repeats > 0
                        and capture_rows.empty
                        and int(anchor) == capture_anchor
                    ):
                        score = batch_blend_score(
                            attention,
                            values,
                            eligible,
                            contribution_weight=weight,
                            dtype=np.float32,
                        )
                        selection = _shared_selection(
                            reference,
                            anchor,
                            score,
                            core_budget,
                            sink_size,
                            recent_size,
                            "blend_attention_contribution_25_w4_shared",
                        )
                        compressed_rows, compressed_summary = _capture_profile(
                            runner,
                            reference,
                            anchor,
                            selection,
                            layers,
                            int(config["total_budget"]),
                            recent_size,
                            capture_repeats,
                        )
                        compressed_rows.insert(0, "cache_mode", "compressed")
                        capture_rows = compressed_rows
                        capture_summary = {"compressed": compressed_summary}
                        if capture_full_repeats > 0:
                            full_selection = _shared_selection(
                                reference,
                                anchor,
                                score,
                                int(eligible.size),
                                sink_size,
                                recent_size,
                                "full_cache_capture_profile",
                            )
                            full_rows, full_summary = _capture_profile(
                                runner,
                                reference,
                                anchor,
                                full_selection,
                                layers,
                                int(next(iter(values.values())).shape[1]),
                                recent_size,
                                capture_full_repeats,
                            )
                            full_rows.insert(0, "cache_mode", "full")
                            capture_rows = pd.concat(
                                [capture_rows, full_rows], ignore_index=True
                            )
                            capture_summary["full"] = full_summary
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    rows = pd.DataFrame(records)
    forward_time_s = None
    comparison_run = config.get("comparison_run")
    if comparison_run:
        metrics = pd.read_csv(repository_root / str(comparison_run) / "metrics.csv")
        policy = str(config.get("comparison_policy"))
        matched = metrics[metrics["policy"] == policy]
        if len(matched) != 1:
            raise RuntimeError("comparison policy is missing or ambiguous")
        forward_time_s = float(matched.iloc[0]["mean_forward_time_s"])

    batch_ms = float(rows["batch_refresh_median_ms"].median())
    update_ms = float(rows["rolling_update_per_token_median_ms"].median())
    refresh_ms = float(rows["rolling_refresh_median_ms"].median())
    for mode_summary in capture_summary.values():
        scheduled: Dict[str, Dict[str, float]] = {}
        per_capture_step = float(
            mode_summary["minimal_direct_capture_overhead_ms"]
            + mode_summary["minimal_cpu_transfer_median_ms"]
            + mode_summary["minimal_cpu_one_query_contribution_median_ms"]
        )
        for interval in refresh_intervals:
            per_decode_step = (
                per_capture_step * min(window, interval) / interval
                + refresh_ms / interval
            )
            scheduled[str(interval)] = {
                "cpu_fallback_ms_per_decode_step": float(per_decode_step),
                "fraction_of_no_capture_forward": float(
                    per_decode_step
                    / mode_summary["no_capture_median_forward_ms"]
                ),
            }
        mode_summary["scheduled_last_window_cpu_fallback"] = scheduled
    amortized: Dict[str, Dict[str, float]] = {}
    for interval in refresh_intervals:
        on_demand = batch_ms / interval
        scheduled = update_ms * min(window, interval) / interval + refresh_ms / interval
        always_rolling = update_ms + refresh_ms / interval
        amortized[str(interval)] = {
            "on_demand_batch_ms_per_decode_step": float(on_demand),
            "scheduled_last_window_rolling_ms_per_decode_step": float(scheduled),
            "always_rolling_ms_per_decode_step": float(always_rolling),
            "scheduled_fraction_of_forward_time": (
                float(scheduled / (1000.0 * forward_time_s))
                if forward_time_s is not None
                else None
            ),
        }
    bytes_per_token = float(rows["rolling_bytes_per_context_token"].max())
    summary = {
        "status": "runtime_capture_and_arithmetic_microbenchmark",
        "confirmatory_evidence": False,
        "policy": "blend_attention_contribution_25_w4_shared",
        "candidate_algorithms_run_per_decision": 0,
        "samples": sorted(sample_ids),
        "anchors": anchors,
        "diagnostic_layers": layers,
        "scenario_window": window,
        "contribution_weight": weight,
        "timing_repeats": repeats,
        "profile_units": int(len(rows)),
        "median_batch_refresh_ms": batch_ms,
        "median_rolling_update_per_token_ms": update_ms,
        "median_rolling_refresh_ms": refresh_ms,
        "minimum_selection_jaccard": float(rows["selection_jaccard"].min()),
        "maximum_score_absolute_error": float(
            rows["score_max_absolute_error"].max()
        ),
        "maximum_observed_working_set_bytes": int(
            rows["rolling_working_set_bytes"].max()
        ),
        "rolling_bytes_per_context_token": bytes_per_token,
        "projected_rolling_working_set_mib": {
            str(tokens): float(bytes_per_token * tokens / (1024.0**2))
            for tokens in (4096, 32768, 131072)
        },
        "comparison_mean_forward_time_s": forward_time_s,
        "compressed_decode_capture": capture_summary,
        "amortized_by_refresh_interval": amortized,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "scope": (
            "CPU NumPy score/top-k timing plus separate MLX forward capture and "
            "CPU-transfer timing on full and compressed caches. It excludes an "
            "integrated fused MLX score kernel and full free-generation runs; "
            "the measurements diagnose feasibility, not end-to-end speedup."
        ),
    }
    atomic_frame(rows, output_root / "profile_rows.csv")
    if not capture_rows.empty:
        atomic_frame(capture_rows, output_root / "capture_profile.csv")
    atomic_json(output_root / "summary.json", summary)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = ["run_direct_policy_runtime_profile"]
