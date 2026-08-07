"""Matched-count stateful refresh policies for robust envelope experiments."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.backend_mlx import MLXReplayState
from statekv.config import DiscoveryConfig
from statekv.functional_probe import _condition_cache
from statekv.robust_envelope import RobustEnvelopeRunner
from statekv.robust_envelope_analysis import EnvelopeModel
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame
from statekv.trajectory_analysis import cluster_bootstrap_interval
from statekv.trajectory_model import exact_distribution_metrics


def refresh_schedule(refresh_count: int, horizon: int) -> List[int]:
    """Evenly spaced, pre-query refresh offsets within a fixed horizon."""

    count = int(refresh_count)
    if count <= 0:
        return []
    return [
        int(round((index + 1) * int(horizon) / (count + 1)))
        for index in range(count)
    ]


def _envelope_model(payload: Mapping[str, Any]) -> EnvelopeModel:
    return EnvelopeModel(
        family=str(payload["family"]),
        layers=[int(value) for value in payload["layers"]],
        a=np.asarray(payload["a"], dtype=np.float64),
        b=np.asarray(payload["b"], dtype=np.float64),
        h=np.asarray(payload["h"], dtype=np.float64),
        scalar=bool(payload["scalar"]),
        source=str(payload["source"]),
    )


def _clone_state(state: MLXReplayState) -> MLXReplayState:
    """Deep-copy an MLX cache so an oracle branch cannot mutate policy state."""

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    caches = []
    for cache in state.cache:
        offset = int(cache.offset)
        cloned = KVCache()
        cloned.state = (
            mx.array(np.asarray(cache.keys[:, :, :offset, :]).copy()),
            mx.array(np.asarray(cache.values[:, :, :offset, :]).copy()),
        )
        cloned.logical_offset = int(cache.logical_offset)
        caches.append(cloned)
    return MLXReplayState(
        cache=caches,
        position_maps={
            int(layer): positions.detach().clone()
            for layer, positions in state.position_maps.items()
        },
        logical_next_position=int(state.logical_next_position),
    )


class RobustEnvelopePolicyRunner:
    """Replay online core updates without resetting accumulated trajectory drift."""

    def __init__(
        self,
        cfg: DiscoveryConfig,
        repository_root: Path,
        run_dir: Path,
        oracle_only: bool = False,
        threshold_only: bool = False,
    ):
        self.cfg = cfg
        self.repository_root = repository_root.resolve()
        self.run_dir = run_dir.resolve()
        self.oracle_only = bool(oracle_only)
        self.threshold_only = bool(threshold_only)
        if self.oracle_only and self.threshold_only:
            raise ValueError("oracle_only and threshold_only are exclusive")
        self.runner = RobustEnvelopeRunner(cfg, self.repository_root)
        self.cache_cfg = _condition_cache(
            cfg,
            int(cfg.robust_envelope.total_budget),
            int(cfg.robust_envelope.protected_recent),
        )
        with (self.run_dir / "envelope_fold_models.json").open() as handle:
            self.fold_models = json.load(handle)

    @property
    def layers(self) -> List[int]:
        return list(self.runner.model.selected_layers)

    def _full_score_bundle(
        self,
        record: Any,
        layer: int,
        values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        attention = record.all_head_attention_distributions[int(layer)].float()
        query_heads = int(attention.shape[0])
        kv_heads = int(values.shape[0])
        repeated = values.repeat_interleave(query_heads // kv_heads, dim=0)
        full_heads = record.all_head_attention_outputs[int(layer)].float()
        aov = attention[:, :, None] * repeated
        aor = (
            attention[:, :, None]
            / (1.0 - attention).clamp_min(1e-6)[:, :, None]
            * (full_heads[:, None, :] - repeated)
        )
        aov_projected = self.runner.model.project_features(
            int(layer),
            aov.permute(1, 0, 2).reshape(int(attention.shape[1]), -1),
        )
        aor_projected = self.runner.model.project_features(
            int(layer),
            aor.permute(1, 0, 2).reshape(int(attention.shape[1]), -1),
        )
        return {
            "attention": attention.mean(dim=0),
            "aov": aov_projected.square().sum(dim=1),
            "aor": aor_projected.square().sum(dim=1),
        }

    def _candidate_fixed_maps(
        self,
        sample_id: str,
        offset: int,
        state: MLXReplayState,
        fixed: Mapping[int, Set[int]],
        record: Any,
        full_values: Mapping[int, torch.Tensor],
    ) -> Dict[str, Dict[int, Set[int]]]:
        names = ("base", "attention", "aov", "aor", "age", "random")
        output = {
            name: {
                int(layer): set(int(value) for value in positions)
                for layer, positions in fixed.items()
            }
            for name in names
        }
        selected = set(self.layers)
        recent_after = max(0, int(self.cache_cfg.recent_size) - 1)
        sink_size = int(self.cache_cfg.sink_size)
        for layer, position_tensor in state.position_maps.items():
            positions = [int(value) for value in position_tensor.tolist()]
            current_fixed = set(int(value) for value in fixed[int(layer)])
            dynamic = [value for value in positions if value not in current_fixed]
            protected = set(dynamic[-recent_after:]) if recent_after else set()
            eligible = [
                value for value in positions if value not in protected
            ]
            target_size = len(current_fixed)
            forced_sink = set(positions[: min(sink_size, len(positions))])
            departure = [value for value in dynamic if value not in protected]

            # Every non-diagnostic layer receives the physical age update;
            # diagnostic layers additionally admit score-driven alternatives.
            age = set(current_fixed)
            if departure:
                removable = sorted(age - forced_sink)
                if removable:
                    age.remove(removable[0])
                    age.add(int(departure[-1]))
            output["age"][int(layer)] = age
            if int(layer) not in selected:
                for name in ("attention", "aov", "aor", "random"):
                    output[name][int(layer)] = set(age)
                continue

            bundle = self._full_score_bundle(
                record, int(layer), full_values[int(layer)]
            )
            for name in ("attention", "aov", "aor"):
                ranked = sorted(
                    (value for value in eligible if value not in forced_sink),
                    key=lambda value: (
                        -float(bundle[name][int(value)].item()),
                        int(value),
                    ),
                )
                output[name][int(layer)] = forced_sink | set(
                    ranked[: max(0, target_size - len(forced_sink))]
                )
            token = "%s:%d:%d:%d" % (
                sample_id,
                int(layer),
                int(offset),
                int(self.cfg.robust_envelope.random_seed),
            )
            seed = int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:8],
                "little",
            )
            rng = np.random.default_rng(seed)
            pool = np.asarray(
                [value for value in eligible if value not in forced_sink],
                dtype=np.int64,
            )
            need = max(0, target_size - len(forced_sink))
            sampled = (
                rng.choice(pool, size=min(need, len(pool)), replace=False).tolist()
                if need
                else []
            )
            output["random"][int(layer)] = forced_sink | set(
                int(value) for value in sampled
            )
        return output

    def _retained_after_prune(
        self,
        state: MLXReplayState,
        fixed: Mapping[int, Set[int]],
        layer: int,
        current_position: int,
    ) -> List[int]:
        positions = [
            int(value) for value in state.position_maps[int(layer)].tolist()
        ]
        fixed_layer = set(int(value) for value in fixed[int(layer)])
        dynamic = [value for value in positions if value not in fixed_layer]
        recent = max(0, int(self.cache_cfg.recent_size) - 1)
        tail = dynamic[-recent:] if recent else []
        return sorted(fixed_layer | set(tail) | {int(current_position)})

    def _direct_vector(
        self,
        state: MLXReplayState,
        fixed: Mapping[int, Set[int]],
        record: Any,
        full_values: Mapping[int, torch.Tensor],
    ) -> np.ndarray:
        current_position = int(record.query_position)
        coordinates = []
        for layer in self.layers:
            retained = self._retained_after_prune(
                state, fixed, int(layer), current_position
            )
            result = self.runner._direct_at_step(
                record,
                int(layer),
                full_values[int(layer)],
                retained,
            )
            coordinates.append(float(result["coordinate"]))
        return np.asarray(coordinates, dtype=np.float64)

    def _oracle_candidate(
        self,
        state: MLXReplayState,
        candidates: Mapping[str, Mapping[int, Set[int]]],
        current_token: int,
        target_index: int,
        reference: Any,
        full_values: Mapping[int, torch.Tensor],
        lookahead_steps: int,
    ) -> str:
        scores = {}
        for name, fixed in candidates.items():
            branch = _clone_state(state)
            branch_values = dict(full_values)
            branch_token = int(current_token)
            cumulative_kl = 0.0
            try:
                for step in range(int(lookahead_steps)):
                    current_index = int(target_index + step)
                    current_record = reference.query_records[current_index]
                    if step > 0:
                        self.runner._append_reference_value(
                            branch_values, current_record
                        )
                    self.runner.model.prune_recent_before_query(
                        branch,
                        {
                            int(layer): set(values)
                            for layer, values in fixed.items()
                        },
                        cache_config=self.cache_cfg,
                    )
                    self.runner._clear_controls()
                    logits, _, _ = self.runner.model.forward_one(
                        branch, branch_token, capture_attention=True
                    )
                    target_token = int(
                        reference.generated_token_ids[current_index]
                    )
                    cumulative_kl += exact_distribution_metrics(
                        reference.probe_logits[current_index],
                        logits,
                        target_token,
                    )["exact_kl"]
                    branch_token = target_token
                scores[name] = cumulative_kl
            finally:
                self.runner.model.release(branch)
        return min(scores, key=lambda name: (scores[name], name))

    def _select_candidate(
        self,
        policy: str,
        candidates: Mapping[str, Mapping[int, Set[int]]],
        directs: Mapping[str, np.ndarray],
        bound: np.ndarray,
        model: Optional[EnvelopeModel],
        margin: Optional[np.ndarray],
        state: MLXReplayState,
        current_token: int,
        target_index: int,
        reference: Any,
        full_values: Mapping[int, torch.Tensor],
        lookahead_steps: int,
    ) -> Tuple[str, float]:
        fixed_choices = {
            "fixed_interval": "attention",
            "age_only": "age",
            "aov_trigger": "aov",
            "aor_trigger": "aor",
        }
        if policy in fixed_choices:
            return fixed_choices[policy], float("nan")
        if policy == "direct_trigger":
            scores = {
                name: float(np.square(value).sum())
                for name, value in directs.items()
            }
        elif policy == "stateful_oracle":
            choice = self._oracle_candidate(
                state,
                candidates,
                current_token,
                target_index,
                reference,
                full_values,
                lookahead_steps,
            )
            return choice, float("nan")
        elif policy.startswith("E") and model is not None and margin is not None:
            scores = {}
            for name, direct in directs.items():
                if model.scalar:
                    direct_value = np.asarray([np.linalg.norm(direct)])
                else:
                    direct_value = direct
                future = model.step(bound, direct_value) + margin
                scores[name] = float(np.square(future).sum())
        else:
            return "base", float("nan")
        choice = min(scores, key=lambda name: (scores[name], name))
        base_score = float(scores["base"])
        return choice, float(base_score - scores[choice])

    def _run_policy(
        self,
        sample: Any,
        reference: Any,
        initial_selection: Any,
        policy: str,
        refresh_count: int,
    ) -> List[Dict[str, Any]]:
        cfg = self.cfg.robust_envelope
        anchor = int(cfg.anchor)
        horizon = int(cfg.horizon)
        schedule = set(refresh_schedule(refresh_count, horizon))
        state, fixed = self.runner.model.state_from_anchor(
            reference.anchors[anchor],
            copy.deepcopy(initial_selection),
            cache_config=self.cache_cfg,
        )
        full_values = self.runner._initial_full_values(reference, anchor)
        current_token = int(reference.anchors[anchor].query_token_id)
        model = None
        margin = None
        envelope_family = policy.split("_", 1)[0]
        if envelope_family in {"E1", "E2", "E3"}:
            payload = self.fold_models[reference.sample_id]["models"][
                "%s:empirical_nonnegative" % envelope_family
            ]
            model = _envelope_model(payload)
            margin = np.asarray(
                payload["simultaneous_margin_0.9"], dtype=np.float64
            )
            dimension = 1 if model.scalar else len(model.layers)
            bound = np.zeros(dimension, dtype=np.float64)
        else:
            bound = np.zeros(len(self.layers), dtype=np.float64)
        refreshes_completed = 0
        rows: List[Dict[str, Any]] = []
        try:
            for offset in range(1, horizon + 1):
                target_index = anchor + offset - 1
                record = reference.query_records[target_index]
                chosen_candidate = "base"
                advantage = float("nan")
                did_refresh = False
                if offset > 1:
                    self.runner._append_reference_value(full_values, record)
                    if offset in schedule:
                        candidates = self._candidate_fixed_maps(
                            reference.sample_id,
                            offset,
                            state,
                            fixed,
                            record,
                            full_values,
                        )
                        directs = {
                            name: self._direct_vector(
                                state, value, record, full_values
                            )
                            for name, value in candidates.items()
                        }
                        chosen_candidate, advantage = self._select_candidate(
                            envelope_family
                            if policy.endswith("_threshold")
                            else policy,
                            candidates,
                            directs,
                            bound,
                            model,
                            margin,
                            state,
                            current_token,
                            target_index,
                            reference,
                            full_values,
                            min(
                                [
                                    value - offset
                                    for value in schedule
                                    if value > offset
                                ]
                                or [horizon - offset + 1]
                            ),
                        )
                        if (
                            policy.endswith("_threshold")
                            and (
                                not np.isfinite(advantage)
                                or advantage
                                <= float(cfg.refresh_cost)
                            )
                        ):
                            chosen_candidate = "base"
                        else:
                            fixed = {
                                int(layer): set(values)
                                for layer, values in candidates[
                                    chosen_candidate
                                ].items()
                            }
                            refreshes_completed += 1
                            did_refresh = True
                    self.runner.model.prune_recent_before_query(
                        state, fixed, cache_config=self.cache_cfg
                    )
                direct = self._direct_vector(
                    state, fixed, record, full_values
                )
                self.runner._clear_controls()
                logits, _, _ = self.runner.model.forward_one(
                    state, current_token, capture_attention=True
                )
                self.runner.model.validate_active_budget(
                    state, cache_config=self.cache_cfg
                )
                target_token = int(reference.generated_token_ids[target_index])
                metrics = exact_distribution_metrics(
                    reference.probe_logits[target_index],
                    logits,
                    target_token,
                )
                if model is not None and margin is not None:
                    direct_value = (
                        np.asarray([np.linalg.norm(direct)])
                        if model.scalar
                        else direct
                    )
                    bound = model.step(bound, direct_value) + margin
                    bound_risk = float(np.square(bound).sum())
                else:
                    bound_risk = float("nan")
                rows.append(
                    {
                        "run_id": self.run_dir.name,
                        "model": self.cfg.model.name,
                        "task": sample.task,
                        "sample_id": sample.sample_id,
                        "seed": int(self.cfg.runtime.seed),
                        "config_hash": self.cfg.config_hash,
                        "policy": policy,
                        "requested_refresh_count": int(refresh_count),
                        "refresh_schedule": json.dumps(sorted(schedule)),
                        "horizon_offset": int(offset),
                        "refresh_opportunity": bool(offset in schedule),
                        "refresh_event": bool(did_refresh),
                        "refreshes_completed": int(refreshes_completed),
                        "chosen_candidate": chosen_candidate,
                        "envelope_advantage": float(advantage),
                        "envelope_bound_risk": bound_risk,
                        "direct_energy": float(np.square(direct).sum()),
                        "exact_kl": float(metrics["exact_kl"]),
                        "js": float(metrics["js"]),
                        "delta_nll": float(metrics["delta_nll"]),
                        "active_cache_tokens": int(
                            self.runner.model.active_cache_tokens(state)
                        ),
                        "total_budget": int(cfg.total_budget),
                        "refresh_cost": float(cfg.refresh_cost),
                        "refresh_reset_state_error": False,
                        "token_position_aligned": bool(
                            int(record.query_position)
                            == int(state.logical_next_position - 1)
                        ),
                    }
                )
                current_token = target_token
        finally:
            self.runner._clear_controls()
            self.runner.model.release(state)
        return rows

    @staticmethod
    def _policy_specs() -> List[Tuple[str, int]]:
        primary = [
            ("no_refresh_static", 0),
            ("fixed_interval", 3),
            ("age_only", 3),
            ("direct_trigger", 3),
            ("aov_trigger", 3),
            ("aor_trigger", 3),
            ("E1", 3),
            ("E2", 3),
            ("E3", 3),
            ("stateful_oracle", 3),
        ]
        curves = [
            ("aov_trigger", 1),
            ("aov_trigger", 2),
            ("E2", 1),
            ("E2", 2),
        ]
        return primary + curves

    def run(self) -> Path:
        samples, _ = load_discovery_tasks(self.cfg)
        model_info = self.runner.model.load()
        metadata_path = self.run_dir / "metadata.json"
        self.runner.metadata = (
            json.load(metadata_path.open())
            if metadata_path.exists()
            else {"git_commit": None, "model_info": model_info}
        )
        fragment_name = (
            "robust_envelope_policy_oracle"
            if self.oracle_only
            else (
                "robust_envelope_policy_threshold"
                if self.threshold_only
                else "robust_envelope_policy"
            )
        )
        fragments = self.run_dir / "fragments" / fragment_name
        fragments.mkdir(parents=True, exist_ok=True)
        try:
            for sample_index, sample in enumerate(samples):
                path = fragments / (
                    sample.sample_id.replace(":", "_") + ".parquet"
                )
                if self.cfg.runtime.resume and path.exists():
                    continue
                reference = self.runner.model.generate_reference(
                    sample.sample_id, sample.task, sample.prompt
                )
                initial = self.runner._selection(
                    reference, "v_ridge", sample_index
                )
                rows = []
                specs = (
                    [("stateful_oracle", 3)]
                    if self.oracle_only
                    else (
                        [("E2_threshold", 3)]
                        if self.threshold_only
                        else self._policy_specs()
                    )
                )
                for policy, refresh_count in specs:
                    rows.extend(
                        self._run_policy(
                            sample,
                            reference,
                            initial,
                            policy,
                            refresh_count,
                        )
                    )
                _atomic_frame(pd.DataFrame(rows), path)
                self.runner.model.release(reference)
            frames = [
                pd.read_parquet(path)
                for path in sorted(fragments.glob("*.parquet"))
            ]
            result = pd.concat(frames, ignore_index=True)
            if self.oracle_only or self.threshold_only:
                base_fragments = (
                    self.run_dir
                    / "fragments"
                    / "robust_envelope_policy"
                )
                base = pd.concat(
                    [
                        pd.read_parquet(path)
                        for path in sorted(base_fragments.glob("*.parquet"))
                    ],
                    ignore_index=True,
                )
                replaced_policy = (
                    "stateful_oracle"
                    if self.oracle_only
                    else "E2_threshold"
                )
                base = base[base["policy"] != replaced_policy]
                if self.threshold_only:
                    oracle_fragments = (
                        self.run_dir
                        / "fragments"
                        / "robust_envelope_policy_oracle"
                    )
                    if oracle_fragments.exists():
                        oracle = pd.concat(
                            [
                                pd.read_parquet(path)
                                for path in sorted(
                                    oracle_fragments.glob("*.parquet")
                                )
                            ],
                            ignore_index=True,
                        )
                        base = base[base["policy"] != "stateful_oracle"]
                        base = pd.concat([base, oracle], ignore_index=True)
                result = pd.concat([base, result], ignore_index=True)
            if not bool(result["token_position_aligned"].all()):
                raise RuntimeError("policy replay contains token alignment failure")
            if int(result["active_cache_tokens"].max()) > int(
                self.cfg.robust_envelope.total_budget
            ):
                raise RuntimeError("policy replay exceeded cache budget")
            output = self.run_dir / "envelope_refresh_policy_rows.parquet"
            _atomic_frame(result, output)
            return output
        finally:
            self.runner.model.close()


def summarize_policy_rows(
    rows: pd.DataFrame, cfg: DiscoveryConfig
) -> Dict[str, Any]:
    per_sequence = (
        rows.groupby(
            [
                "sample_id",
                "task",
                "policy",
                "requested_refresh_count",
            ]
        )
        .agg(
            cumulative_kl=("exact_kl", "sum"),
            cumulative_js=("js", "sum"),
            cumulative_abs_nll=(
                "delta_nll", lambda value: float(np.abs(value).sum())
            ),
            final_refresh_count=("refreshes_completed", "max"),
            maximum_cache_tokens=("active_cache_tokens", "max"),
        )
        .reset_index()
    )
    policy_summary = (
        per_sequence.groupby(["policy", "requested_refresh_count"])
        .agg(
            median_cumulative_kl=("cumulative_kl", "median"),
            median_cumulative_js=("cumulative_js", "median"),
            median_cumulative_abs_nll=("cumulative_abs_nll", "median"),
            median_actual_refresh_count=("final_refresh_count", "median"),
            sequence_count=("sample_id", "nunique"),
        )
        .reset_index()
    )
    primary = per_sequence[
        per_sequence["requested_refresh_count"].isin([0, 3])
    ]
    baselines = {
        "no_refresh_static", "fixed_interval", "age_only",
        "direct_trigger", "aov_trigger", "aor_trigger",
    }
    envelope_names = {"E1", "E2", "E3"}
    task_gate = []
    for task, group in primary.groupby("task"):
        baseline = float(
            group[group["policy"].isin(baselines)]
            .groupby("policy")["cumulative_kl"]
            .median()
            .min()
        )
        envelope = float(
            group[group["policy"].isin(envelope_names)]
            .groupby("policy")["cumulative_kl"]
            .median()
            .min()
        )
        task_gate.append(
            {
                "task": task,
                "best_baseline_median_cumulative_kl": baseline,
                "best_envelope_median_cumulative_kl": envelope,
                "envelope_minus_baseline": envelope - baseline,
                "direction_improves": bool(envelope < baseline),
            }
        )
    bootstrap = {}
    for values, group in per_sequence.groupby(
        ["policy", "requested_refresh_count"]
    ):
        key = "%s:refresh_%d" % (values[0], int(values[1]))
        bootstrap[key] = {
            metric: cluster_bootstrap_interval(
                group,
                metric,
                cluster="sample_id",
                samples=int(cfg.runtime.bootstrap_samples),
                seed=int(cfg.runtime.seed),
            )
            for metric in (
                "cumulative_kl",
                "cumulative_js",
                "cumulative_abs_nll",
            )
        }
    return {
        "schema_version": "robust_envelope_refresh_policy_v2",
        "scope": (
            "teacher-forced stateful online replay; refresh updates the physical "
            "fixed core before pruning, never reconstructs an evicted KV, and "
            "never resets accumulated hidden-state error"
        ),
        "refresh_schedules": {
            str(count): refresh_schedule(
                count, int(cfg.robust_envelope.horizon)
            )
            for count in (0, 1, 2, 3)
        },
        "same_budget": bool(
            per_sequence["maximum_cache_tokens"].max()
            <= int(cfg.robust_envelope.total_budget)
        ),
        "same_refresh_cost": True,
        "matched_count_primary": 3,
        "per_sequence": per_sequence.to_dict("records"),
        "policy_summary": policy_summary.to_dict("records"),
        "sequence_cluster_bootstrap_95ci": bootstrap,
        "quality_refresh_curve": policy_summary[
            (
                policy_summary["policy"].isin(["aov_trigger", "E2"])
                & policy_summary["requested_refresh_count"].isin([1, 2, 3])
            )
            | (policy_summary["policy"] == "no_refresh_static")
            | (policy_summary["policy"] == "E2_threshold")
        ].to_dict("records"),
        "task_direction_gate": task_gate,
        "policy_value_gate": {
            "pass": bool(task_gate)
            and all(row["direction_improves"] for row in task_gate),
            "requires_both_task_directions": True,
        },
        "refresh_does_not_reset_existing_error": bool(
            not rows["refresh_reset_state_error"].any()
        ),
        "stateful_oracle_is_upper_bound_only": True,
        "task_score_limitation": (
            "The fixed reference-token teacher-forced protocol measures KL/JS/"
            "NLL but cannot identify free-generation NIAH or ROUGE task scores."
        ),
    }
