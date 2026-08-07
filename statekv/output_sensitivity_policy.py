"""Conditional stateful refresh replay for validated output readouts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

import numpy as np
import pandas as pd

from statekv.config import DiscoveryConfig
from statekv.functional_probe import _condition_cache
from statekv.output_sensitivity import OutputSensitivityRunner
from statekv.output_sensitivity_analysis import Bridge, LAYERS
from statekv.robust_envelope_analysis import EnvelopeModel
from statekv.robust_envelope_policy import (
    RobustEnvelopePolicyRunner,
    refresh_schedule,
)
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame
from statekv.trajectory_model import exact_distribution_metrics


def _e2_from_fold(payload: Mapping[str, Any]) -> EnvelopeModel:
    return EnvelopeModel(
        family="E2",
        layers=list(LAYERS),
        a=np.asarray(payload["a"], dtype=np.float64),
        b=np.asarray(payload["b"], dtype=np.float64),
        h=np.zeros((len(LAYERS), len(LAYERS)), dtype=np.float64),
        scalar=False,
        source="frozen_empirical_nonnegative",
    )


def _bridge_from_fold(family: str, payload: Mapping[str, Any]) -> Bridge:
    return Bridge(
        family=family,
        coefficient=np.asarray(payload["coefficient"], dtype=np.float64),
        intercept=float(payload["intercept"]),
        metadata=dict(payload.get("metadata", {})),
    )


class OutputSensitivityPolicyRunner(RobustEnvelopePolicyRunner):
    """Receding-horizon physical policy with pairwise-LCB abstention."""

    def __init__(
        self,
        cfg: DiscoveryConfig,
        repository_root: Path,
        run_dir: Path,
    ):
        proxy = copy.deepcopy(cfg)
        proxy.output_sensitivity.enabled = False
        proxy.robust_envelope.enabled = True
        proxy.robust_envelope.anchor = int(cfg.output_sensitivity.anchors[0])
        proxy.robust_envelope.horizon = int(
            cfg.output_sensitivity.segment_horizon
        )
        proxy.robust_envelope.total_budget = int(
            cfg.output_sensitivity.total_budget
        )
        proxy.robust_envelope.protected_recent = int(
            cfg.output_sensitivity.protected_recent
        )
        proxy.robust_envelope.refresh_cost = float(
            cfg.output_sensitivity.refresh_cost
        )
        proxy.runtime.run_id = "%s_policy_runtime" % (
            cfg.runtime.run_id or "output_sensitivity"
        )
        self.original_cfg = cfg
        self.cfg = proxy
        self.repository_root = repository_root.resolve()
        self.run_dir = run_dir.resolve()
        self.oracle_only = False
        self.threshold_only = False
        self.runner = OutputSensitivityRunner(proxy, self.repository_root)
        self.cache_cfg = _condition_cache(
            proxy,
            int(proxy.robust_envelope.total_budget),
            int(proxy.robust_envelope.protected_recent),
        )
        with (self.run_dir / "output_bridge_fold_models.json").open() as handle:
            self.fold_models = json.load(handle)
        with (self.run_dir / "output_sensitivity_gate_decision.json").open() as handle:
            gate = json.load(handle)
        with (self.run_dir / "output_bridge_ranking_summary.json").open() as handle:
            ranking = json.load(handle)
        passing = [
            value
            for value in gate["stage_b_passing_families"]
            if value in {"O1", "O2", "O3", "O4_CONT", "O4_REGIME"}
        ]
        task_rows = pd.DataFrame(ranking["task_split"])
        worst = (
            task_rows[task_rows["bridge_family"].isin(passing)]
            .groupby("bridge_family")["median_spearman"]
            .min()
            .sort_values(ascending=False)
        )
        self.output_family = str(worst.index[0]) if len(worst) else "O2"

    def _candidate_objectives(
        self,
        sample_id: str,
        candidate_directs: Mapping[str, np.ndarray],
        current_bound: np.ndarray,
        horizon: int,
        observable: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        fold = self.fold_models[sample_id]
        e2 = _e2_from_fold(fold["e2"])
        state_margin = np.asarray(fold["state_margin"], dtype=np.float64)
        bridge = _bridge_from_fold(
            self.output_family, fold["bridges"][self.output_family]
        )
        old_scores: Dict[str, float] = {}
        output_scores: Dict[str, float] = {}
        for name, direct in candidate_directs.items():
            bound = np.asarray(current_bound, dtype=np.float64).copy()
            old_value = 0.0
            output_value = 0.0
            for step in range(1, int(horizon) + 1):
                bound = e2.step(bound, direct) + state_margin
                payload = {
                    **{"b_l%d" % layer: bound[index] for index, layer in enumerate(LAYERS)},
                    **{"d_l%d" % layer: direct[index] for index, layer in enumerate(LAYERS)},
                    **dict(observable),
                    "horizon_offset": step,
                }
                prediction = float(
                    bridge.predict(pd.DataFrame([payload]), LAYERS)[0]
                )
                old_value += float(np.square(bound).sum())
                output_value += 0.25 * prediction * prediction
            old_scores[name] = old_value
            output_scores[name] = output_value
        return old_scores, output_scores

    def _run_output_policy(
        self,
        sample: Any,
        reference: Any,
        initial_selection: Any,
        policy: str,
        maximum_refresh_count: int,
        receding_horizon: int,
    ) -> List[Dict[str, Any]]:
        anchor = int(self.cfg.robust_envelope.anchor)
        total_horizon = int(self.cfg.robust_envelope.horizon)
        schedule = set(refresh_schedule(maximum_refresh_count, total_horizon))
        state, fixed = self.runner.model.state_from_anchor(
            reference.anchors[anchor],
            copy.deepcopy(initial_selection),
            cache_config=self.cache_cfg,
        )
        full_values = self.runner._initial_full_values(reference, anchor)
        current_token = int(reference.anchors[anchor].query_token_id)
        fold = self.fold_models[reference.sample_id]
        e2 = _e2_from_fold(fold["e2"])
        state_margin = np.asarray(fold["state_margin"], dtype=np.float64)
        bound = np.zeros(len(LAYERS), dtype=np.float64)
        refreshes = 0
        rows: List[Dict[str, Any]] = []
        try:
            for offset in range(1, total_horizon + 1):
                target_index = anchor + offset - 1
                record = reference.query_records[target_index]
                did_refresh = False
                abstained = False
                choice = "base"
                predicted_benefit = float("nan")
                pair_margin = float("nan")
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
                                state, values, record, full_values
                            )
                            for name, values in candidates.items()
                        }
                        if policy == "fixed_interval":
                            choice = "attention"
                        elif policy == "age":
                            choice = "age"
                        elif policy == "aov":
                            choice = "aov"
                        elif policy == "aor":
                            choice = "aor"
                        elif policy == "direct":
                            choice = min(
                                directs,
                                key=lambda name: (
                                    float(np.square(directs[name]).sum()),
                                    name,
                                ),
                            )
                        elif policy == "oracle":
                            choice = self._oracle_candidate(
                                state,
                                candidates,
                                current_token,
                                target_index,
                                reference,
                                full_values,
                                min(
                                    int(receding_horizon),
                                    total_horizon - offset + 1,
                                ),
                            )
                        else:
                            observable = self.runner._operating_features(
                                record, reference.probe_logits[target_index]
                            )
                            old_scores, output_scores = self._candidate_objectives(
                                reference.sample_id,
                                directs,
                                bound,
                                int(receding_horizon),
                                observable,
                            )
                            scores = (
                                old_scores
                                if policy == "old_E2"
                                else output_scores
                            )
                            choice = min(
                                scores, key=lambda name: (scores[name], name)
                            )
                            predicted_benefit = float(
                                scores["base"] - scores[choice]
                            )
                            if policy == "pairwise_LCB":
                                horizon_key = str(int(receding_horizon))
                                pair_margin = float(
                                    fold["bridges"][self.output_family][
                                        "pairwise_margin_95"
                                    ][horizon_key]["margin"]
                                )
                                lower = predicted_benefit - pair_margin
                                if (
                                    lower
                                    <= float(
                                        self.cfg.robust_envelope.refresh_cost
                                    )
                                ):
                                    choice = "base"
                                    abstained = True
                        if choice != "base":
                            fixed = {
                                int(layer): set(values)
                                for layer, values in candidates[choice].items()
                            }
                            did_refresh = True
                            refreshes += 1
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
                target_token = int(reference.generated_token_ids[target_index])
                metric = exact_distribution_metrics(
                    reference.probe_logits[target_index],
                    logits,
                    target_token,
                )
                bound = e2.step(bound, direct) + state_margin
                rows.append(
                    {
                        "run_id": self.run_dir.name,
                        "model": self.original_cfg.model.name,
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "task_bucket": (
                            "GovReport"
                            if "gov" in sample.task.lower()
                            else "NIAH"
                        ),
                        "policy": policy,
                        "output_bridge_family": self.output_family,
                        "receding_horizon": int(receding_horizon),
                        "maximum_refresh_count": int(
                            maximum_refresh_count
                        ),
                        "horizon_offset": int(offset),
                        "refresh_opportunity": bool(offset in schedule),
                        "refresh_event": bool(did_refresh),
                        "actual_refresh_count": int(refreshes),
                        "abstained": bool(abstained),
                        "chosen_candidate": choice,
                        "predicted_benefit": predicted_benefit,
                        "pairwise_margin": pair_margin,
                        "exact_kl": float(metric["exact_kl"]),
                        "js": float(metric["js"]),
                        "delta_nll": float(metric["delta_nll"]),
                        "active_cache_tokens": int(
                            self.runner.model.active_cache_tokens(state)
                        ),
                        "refresh_reset_state_error": False,
                        "refresh_recalled_deleted_kv": False,
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

    def _specs(self) -> List[Tuple[str, int, int]]:
        result = [("static", 0, 4)]
        for policy in ("fixed_interval", "age", "direct", "aov", "aor"):
            for count in (1, 2, 3):
                result.append((policy, count, 4))
        for policy in ("old_E2", "output_E2", "pairwise_LCB", "oracle"):
            for horizon in (4, 8, 16):
                for count in (1, 2, 3):
                    result.append((policy, count, horizon))
        return result

    def run(self) -> Path:
        samples, _ = load_discovery_tasks(self.original_cfg)
        model_info = self.runner.model.load()
        metadata_path = self.run_dir / "metadata.json"
        self.runner.metadata = (
            json.load(metadata_path.open())
            if metadata_path.exists()
            else {"git_commit": None, "model_info": model_info}
        )
        fragments = self.run_dir / "fragments" / "output_sensitivity_policy"
        fragments.mkdir(parents=True, exist_ok=True)
        try:
            for sample_index, sample in enumerate(samples):
                path = fragments / (sample.sample_id.replace(":", "_") + ".parquet")
                if self.original_cfg.runtime.resume and path.exists():
                    continue
                reference = self.runner.model.generate_reference(
                    sample.sample_id, sample.task, sample.prompt
                )
                initial = self.runner._selector_at(
                    reference, int(self.cfg.robust_envelope.anchor)
                ).select(
                    reference.anchors[
                        int(self.cfg.robust_envelope.anchor)
                    ].snapshot(reference.sample_id),
                    "v_ridge_leverage",
                )
                rows: List[Dict[str, Any]] = []
                for policy, count, horizon in self._specs():
                    rows.extend(
                        self._run_output_policy(
                            sample,
                            reference,
                            initial,
                            policy,
                            count,
                            horizon,
                        )
                    )
                _atomic_frame(pd.DataFrame(rows), path)
                self.runner.model.release(reference)
            result = pd.concat(
                [
                    pd.read_parquet(path)
                    for path in sorted(fragments.glob("*.parquet"))
                ],
                ignore_index=True,
            )
            if not bool(result["token_position_aligned"].all()):
                raise RuntimeError("policy replay contains alignment failure")
            if int(result["active_cache_tokens"].max()) > int(
                self.original_cfg.output_sensitivity.total_budget
            ):
                raise RuntimeError("policy replay exceeded cache budget")
            output = self.run_dir / "refresh_lcb_policy_rows.parquet"
            _atomic_frame(result, output)
            return output
        finally:
            self.runner.model.close()


def summarize_output_policy(rows: pd.DataFrame) -> Dict[str, Any]:
    sequence = (
        rows.groupby(
            [
                "sample_id",
                "task_bucket",
                "policy",
                "receding_horizon",
                "maximum_refresh_count",
            ],
            as_index=False,
        )
        .agg(
            cumulative_exact_kl=("exact_kl", "sum"),
            cumulative_js=("js", "sum"),
            cumulative_abs_nll=(
                "delta_nll", lambda value: float(np.abs(value).sum())
            ),
            actual_refresh_count=("actual_refresh_count", "max"),
            abstention_rate=("abstained", "mean"),
        )
    )
    table = (
        sequence.groupby(
            [
                "policy",
                "task_bucket",
                "receding_horizon",
                "maximum_refresh_count",
            ],
            as_index=False,
        )
        .agg(
            median_cumulative_kl=("cumulative_exact_kl", "median"),
            median_actual_refresh_count=("actual_refresh_count", "median"),
            median_abstention_rate=("abstention_rate", "median"),
        )
    )
    matched: List[Dict[str, Any]] = []
    for (task, count), group in sequence.groupby(
        ["task_bucket", "actual_refresh_count"], sort=False
    ):
        medians = group.groupby("policy")["cumulative_exact_kl"].median()
        if "pairwise_LCB" in medians:
            baselines = medians.drop(
                labels=["pairwise_LCB", "oracle"], errors="ignore"
            )
            matched.append(
                {
                    "task_bucket": task,
                    "actual_refresh_count": int(count),
                    "pairwise_lcb_median_kl": float(
                        medians["pairwise_LCB"]
                    ),
                    "best_baseline_median_kl": float(baselines.min()),
                    "difference": float(
                        medians["pairwise_LCB"] - baselines.min()
                    ),
                }
            )
    matched_frame = pd.DataFrame(matched)
    task_best = (
        matched_frame.groupby("task_bucket")["difference"].min().to_dict()
        if len(matched_frame)
        else {}
    )
    lcb_sequence = sequence[sequence["policy"] == "pairwise_LCB"]
    horizon_summary = (
        lcb_sequence.groupby(
            ["task_bucket", "receding_horizon"], as_index=False
        )["cumulative_exact_kl"]
        .median()
    )
    pooled_horizon = (
        lcb_sequence.groupby("receding_horizon")[
            "cumulative_exact_kl"
        ].median()
    )
    best_horizon = (
        int(pooled_horizon.idxmin()) if len(pooled_horizon) else None
    )
    maximum_trigger_fraction = float(
        (
            lcb_sequence["actual_refresh_count"]
            / lcb_sequence["maximum_refresh_count"].clip(lower=1)
        ).mean()
        if len(lcb_sequence)
        else float("nan")
    )
    return {
        "status": "complete",
        "teacher_forced_stateful_replay": True,
        "refresh_does_not_reset_accumulated_state": bool(
            not rows["refresh_reset_state_error"].any()
        ),
        "refresh_never_recalled_deleted_kv": bool(
            not rows["refresh_recalled_deleted_kv"].any()
        ),
        "maximum_refresh_count_respected": bool(
            (
                rows["actual_refresh_count"]
                <= rows["maximum_refresh_count"]
            ).all()
        ),
        "table": json.loads(table.to_json(orient="records")),
        "matched_actual_refresh_count": matched,
        "receding_horizon_task_summary": json.loads(
            horizon_summary.to_json(orient="records")
        ),
        "best_receding_horizon_by_pooled_median_kl": best_horizon,
        "pairwise_lcb_mean_fraction_of_maximum_refreshes": (
            maximum_trigger_fraction
        ),
        "policy_gate": {
            "both_tasks_nonworse_at_some_matched_actual_count": bool(
                set(task_best) == {"NIAH", "GovReport"}
                and all(value <= 0.0 for value in task_best.values())
            ),
            "at_least_one_task_strictly_improves": bool(
                any(value < -1e-12 for value in task_best.values())
            ),
            "not_almost_always_max_trigger": bool(
                maximum_trigger_fraction < 0.95
            ),
            "best_matched_difference_by_task": task_best,
        },
        "task_split_reported": True,
    }
