"""Conditional free-generation external validation for selected cache actions."""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from statekv.config import DiscoveryConfig
from statekv.functional_probe import _condition_cache
from statekv.output_sensitivity import OutputSensitivityRunner
from statekv.selectors import CoreSelection, LayerSelection
from statekv.tasks import load_discovery_tasks
from src.evaluation.official_metrics import (
    longbench_score,
    normalize_answer,
    rouge_l_score,
    ruler_score,
)


def _ngram_f1(prediction: str, reference: str, n: int) -> float:
    left = prediction.split()
    right = reference.split()
    left_counts = Counter(
        tuple(left[index : index + n])
        for index in range(max(0, len(left) - n + 1))
    )
    right_counts = Counter(
        tuple(right[index : index + n])
        for index in range(max(0, len(right) - n + 1))
    )
    overlap = sum((left_counts & right_counts).values())
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    if not overlap or not left_total or not right_total:
        return 0.0
    precision = overlap / left_total
    recall = overlap / right_total
    return float(2.0 * precision * recall / (precision + recall))


def _repetition_rate(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 4:
        return 0.0
    grams = [
        tuple(tokens[index : index + 4])
        for index in range(len(tokens) - 3)
    ]
    return float(1.0 - len(set(grams)) / max(len(grams), 1))


class OutputSensitivityFreeGenerationRunner:
    def __init__(
        self,
        cfg: DiscoveryConfig,
        repository_root: Path,
        run_dir: Path,
    ):
        self.cfg = copy.deepcopy(cfg)
        self.repository_root = repository_root.resolve()
        self.run_dir = run_dir.resolve()
        self.runner = OutputSensitivityRunner(self.cfg, self.repository_root)
        self.cache_cfg = _condition_cache(
            self.cfg,
            int(self.cfg.output_sensitivity.total_budget),
            int(self.cfg.output_sensitivity.protected_recent),
        )
        self.inventory = pd.read_parquet(
            self.run_dir / "output_candidate_inventory.parquet"
        )
        self.bridge_rows = pd.read_parquet(
            self.run_dir / "output_bridge_rows.parquet"
        )
        with (self.run_dir / "output_bridge_fold_models.json").open() as handle:
            self.folds = json.load(handle)

    def _selection(
        self,
        sample_id: str,
        anchor: int,
        candidate_id: str,
    ) -> CoreSelection:
        row = self.inventory[
            (self.inventory["sample_id"] == sample_id)
            & (self.inventory["anchor"] == int(anchor))
            & (self.inventory["candidate_id"] == candidate_id)
        ].iloc[0]
        masks = json.loads(row["selected_positions_json"])
        by_layer = {
            int(layer): LayerSelection(
                layer=int(layer),
                selected_positions=[int(value) for value in positions],
                eligible_positions=[],
                aggregate_scores=[],
                metadata={
                    "physical_shared_mask": True,
                    "per_query_head_selection": False,
                    "source": row["candidate_source"],
                },
            )
            for layer, positions in masks.items()
        }
        return CoreSelection(
            strategy=str(row["candidate_source"]),
            horizon_condition=None,
            by_layer=by_layer,
        )

    def _method_candidates(self, sample_id: str, anchor: int) -> Dict[str, str]:
        rows = self.bridge_rows[
            (self.bridge_rows["held_out_sequence"] == sample_id)
            & (self.bridge_rows["anchor"] == int(anchor))
            & (self.bridge_rows["horizon_offset"] <= 16)
        ]
        action = (
            rows.groupby(
                ["bridge_family", "candidate_id", "candidate_source"],
                as_index=False,
            )
            .agg(
                objective=("induced_kl_bound", "sum"),
                true_kl=("exact_kl", "sum"),
            )
        )
        def chosen(family: str) -> str:
            current = action[action["bridge_family"] == family]
            return str(current.loc[current["objective"].idxmin(), "candidate_id"])

        attention = str(
            action[
                (action["bridge_family"] == "O0")
                & (action["candidate_source"] == "attention")
            ]["candidate_id"].iloc[0]
        )
        aov = str(
            action[
                (action["bridge_family"] == "O0")
                & (action["candidate_source"] == "aov")
            ]["candidate_id"].iloc[0]
        )
        old = chosen("O0")
        new = chosen("O2")
        o2 = action[action["bridge_family"] == "O2"].set_index("candidate_id")
        benefit = float(o2.loc[attention, "objective"] - o2.loc[new, "objective"])
        pair_payload = self.folds[sample_id]["bridges"]["O2"][
            "pairwise_margin_95"
        ]
        pair_horizon = max(
            (int(value) for value in pair_payload if int(value) <= 16),
            default=min(int(value) for value in pair_payload),
        )
        pair_margin = float(
            pair_payload[str(pair_horizon)]["margin"]
        )
        lcb = benefit - pair_margin
        pair = (
            new
            if lcb > float(self.cfg.output_sensitivity.refresh_cost)
            else attention
        )
        oracle_rows = action[action["bridge_family"] == "O2"]
        oracle = str(
            oracle_rows.loc[oracle_rows["true_kl"].idxmin(), "candidate_id"]
        )
        return {
            "static_attention": attention,
            "strongest_fixed_aov": aov,
            "old_E2_O0": old,
            "new_output_O2": new,
            "pairwise_LCB": pair,
            "candidate_oracle_upper_bound": oracle,
        }

    def _generate(
        self,
        reference: Any,
        selection: CoreSelection,
        anchor: int,
        answer_token_ids: Sequence[int],
    ) -> Dict[str, Any]:
        state, fixed = self.runner.model.state_from_anchor(
            reference.anchors[int(anchor)],
            copy.deepcopy(selection),
            cache_config=self.cache_cfg,
        )
        current_token = int(reference.anchors[int(anchor)].query_token_id)
        generated = [
            int(value)
            for value in reference.generated_token_ids[: int(anchor)]
        ]
        answer_probabilities: List[float] = []
        try:
            for offset in range(
                int(anchor), int(self.cfg.generation.max_new_tokens)
            ):
                if offset > int(anchor):
                    self.runner.model.prune_recent_before_query(
                        state, fixed, cache_config=self.cache_cfg
                    )
                self.runner._clear_controls()
                logits, _, _ = self.runner.model.forward_one(
                    state, current_token, capture_attention=True
                )
                probability = torch.softmax(logits.float(), dim=-1)
                if answer_token_ids:
                    answer_probabilities.append(
                        float(
                            probability[
                                torch.as_tensor(
                                    answer_token_ids, dtype=torch.long
                                )
                            ].sum().item()
                        )
                    )
                next_token = int(torch.argmax(logits.float()).item())
                generated.append(next_token)
                current_token = next_token
        finally:
            self.runner._clear_controls()
            self.runner.model.release(state)
        return {
            "token_ids": generated,
            "text": self.runner.model.tokenizer.decode(
                generated, skip_special_tokens=True
            ),
            "answer_token_probability_max": float(
                max(answer_probabilities) if answer_probabilities else float("nan")
            ),
            "answer_token_probability_mean": float(
                np.mean(answer_probabilities)
                if answer_probabilities
                else float("nan")
            ),
        }

    def _metric_row(
        self,
        sample: Any,
        method: str,
        generation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        text = str(generation["text"])
        references = [str(value) for value in sample.references]
        result = {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "task_bucket": (
                "GovReport" if "gov" in sample.task.lower() else "NIAH"
            ),
            "method": method,
            "generation_text": text,
            "generation_length_tokens": len(generation["token_ids"]),
            "repetition_4gram_rate": _repetition_rate(text),
            "truncated_at_max_tokens": bool(
                len(generation["token_ids"])
                >= int(self.cfg.generation.max_new_tokens)
            ),
            "answer_token_probability_max": generation[
                "answer_token_probability_max"
            ],
            "answer_token_probability_mean": generation[
                "answer_token_probability_mean"
            ],
            "cache_budget": (
                "full"
                if method == "full_cache"
                else int(self.cfg.output_sensitivity.total_budget)
            ),
            "maximum_refresh_count": 1,
        }
        if "gov" in sample.task.lower():
            result.update(
                {
                    "rouge_l": float(
                        max(
                            rouge_l_score(text, reference)
                            for reference in references
                        )
                    ),
                    "rouge_1": float(
                        max(_ngram_f1(text, reference, 1) for reference in references)
                    ),
                    "rouge_2": float(
                        max(_ngram_f1(text, reference, 2) for reference in references)
                    ),
                    "official_longbench_score": float(
                        longbench_score("gov_report", text, references) or 0.0
                    ),
                    "factual_consistency": None,
                    "failure_type": (
                        "low_rouge"
                        if max(
                            rouge_l_score(text, reference)
                            for reference in references
                        )
                        < 0.1
                        else "none"
                    ),
                }
            )
        else:
            normalized = normalize_answer(text)
            exact = any(
                normalize_answer(reference) == normalized
                for reference in references
            )
            retrieval = any(
                normalize_answer(reference) in normalized
                for reference in references
            )
            result.update(
                {
                    "exact_match": bool(exact),
                    "needle_retrieval_accuracy": float(retrieval),
                    "official_ruler_score": float(
                        ruler_score(sample.task, text, references) or 0.0
                    ),
                    "failure_type": (
                        "none" if retrieval else "needle_not_retrieved"
                    ),
                }
            )
        return result

    def run(self) -> Dict[str, Any]:
        samples, _ = load_discovery_tasks(self.cfg)
        model_info = self.runner.model.load()
        metadata_path = self.run_dir / "metadata.json"
        self.runner.metadata = (
            json.load(metadata_path.open())
            if metadata_path.exists()
            else {"git_commit": None, "model_info": model_info}
        )
        anchor = int(self.cfg.output_sensitivity.anchors[0])
        rows: List[Dict[str, Any]] = []
        try:
            for sample in samples:
                reference = self.runner.model.generate_reference(
                    sample.sample_id, sample.task, sample.prompt
                )
                reference_tokens = [
                    int(value) for value in reference.generated_token_ids
                ]
                reference_text = self.runner.model.tokenizer.decode(
                    reference_tokens, skip_special_tokens=True
                )
                answer_ids: List[int] = []
                for answer in sample.references:
                    answer_ids.extend(
                        int(value)
                        for value in self.runner.model.tokenizer.encode(
                            str(answer), add_special_tokens=False
                        )
                    )
                full = {
                    "token_ids": reference_tokens,
                    "text": reference_text,
                    "answer_token_probability_max": float("nan"),
                    "answer_token_probability_mean": float("nan"),
                }
                rows.append(self._metric_row(sample, "full_cache", full))
                generated_by_candidate: Dict[str, Dict[str, Any]] = {}
                for method, candidate_id in self._method_candidates(
                    sample.sample_id, anchor
                ).items():
                    if candidate_id not in generated_by_candidate:
                        generated_by_candidate[candidate_id] = self._generate(
                            reference,
                            self._selection(
                                sample.sample_id, anchor, candidate_id
                            ),
                            anchor,
                            sorted(set(answer_ids)),
                        )
                    rows.append(
                        self._metric_row(
                            sample,
                            method,
                            generated_by_candidate[candidate_id],
                        )
                    )
                self.runner.model.release(reference)
        finally:
            self.runner.model.close()
        frame = pd.DataFrame(rows)
        numeric = [
            column
            for column in [
                "exact_match",
                "needle_retrieval_accuracy",
                "official_ruler_score",
                "rouge_l",
                "rouge_1",
                "rouge_2",
                "official_longbench_score",
                "generation_length_tokens",
                "repetition_4gram_rate",
            ]
            if column in frame
        ]
        summary = json.loads(
            frame.groupby(["method", "task_bucket"])[numeric]
            .mean(numeric_only=True)
            .reset_index()
            .to_json(orient="records")
        )
        summary_frame = pd.DataFrame(summary)
        direction = {}
        for task, metric in (
            ("NIAH", "needle_retrieval_accuracy"),
            ("GovReport", "rouge_l"),
        ):
            task_rows = summary_frame[
                summary_frame["task_bucket"] == task
            ].set_index("method")
            direction[task] = {
                "metric": metric,
                "new_minus_old": float(
                    task_rows.loc["new_output_O2", metric]
                    - task_rows.loc["old_E2_O0", metric]
                ),
                "pairwise_lcb_minus_static": float(
                    task_rows.loc["pairwise_LCB", metric]
                    - task_rows.loc["static_attention", metric]
                ),
            }
        return {
            "status": "complete",
            "protocol": (
                "free generation after a common 16-token full-cache warm-up; "
                "one physical selection/refresh action; no teacher forcing "
                "after intervention"
            ),
            "same_model_generation_and_seed": True,
            "compressed_methods_same_budget": True,
            "full_cache_is_uncompressed_reference": True,
            "maximum_refresh_count": 1,
            "rows": json.loads(frame.to_json(orient="records")),
            "summary": summary,
            "external_validity_gate": {
                "task_directions": direction,
                "new_vs_old_nonconflicting": bool(
                    all(
                        value["new_minus_old"] >= 0.0
                        for value in direction.values()
                    )
                ),
                "pairwise_vs_static_nonconflicting": bool(
                    all(
                        value["pairwise_lcb_minus_static"] >= 0.0
                        for value in direction.values()
                    )
                ),
                "at_least_one_nontrivial_improvement": bool(
                    any(
                        value["new_minus_old"] > 1e-12
                        or value["pairwise_lcb_minus_static"] > 1e-12
                        for value in direction.values()
                    )
                ),
            },
            "factual_consistency_note": (
                "No repository factual-consistency metric was available; "
                "the field is null rather than substituted post hoc."
            ),
        }
