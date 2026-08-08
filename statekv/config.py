"""Strict configuration for the opt-in temporal discovery pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml


@dataclass
class ModelDiscoveryConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    backend: str = "torch"
    quant_bits: Optional[int] = None
    deterministic: bool = True
    temperature: float = 0.0
    do_sample: bool = False
    revision: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    attn_implementation: str = "eager"
    prompt_format: str = "chat_template"
    system_prompt: Optional[str] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheDiscoveryConfig:
    total_budget: int = 256
    sink_size: int = 4
    recent_size: int = 32
    selected_core_budget: int = 220


@dataclass
class GenerationDiscoveryConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    stop_on_eos: bool = True


@dataclass
class DiagnosticsDiscoveryConfig:
    num_layers: int = 3
    heads_per_layer: int = 2
    layer_selection: str = "evenly_spaced"
    explicit_layers: List[int] = field(default_factory=list)
    explicit_heads: List[int] = field(default_factory=list)


@dataclass
class MetricsDiscoveryConfig:
    logits_top_k: int = 128
    probability_floor: float = 1e-12
    attention_error_epsilon: float = 1e-8
    large_loss_spike_threshold: float = 0.25


@dataclass
class SelectorDiscoveryConfig:
    observation_window: int = 32
    snapkv_pooling_kernel: int = 63
    snapkv_pooling: str = "max"
    ridge_lambda: float = 1e-3
    ridge_lambda_mode: str = "relative"
    attention_weighted_ridge_lambda: float = 1e-3
    attention_weight_epsilon: float = 1e-4
    shared_token_selection: bool = True


@dataclass
class RuntimeDiscoveryConfig:
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    prefill_chunk_size: int = 32
    resume: bool = True
    fail_on_error: bool = False
    max_prompt_tokens: int = 1536
    output_root: str = "results/temporal_cache_discovery"
    run_id: Optional[str] = None
    bootstrap_samples: int = 1000


@dataclass
class ValidityThresholdsConfig:
    avg_delta_nll: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.25]
    )
    max_delta_nll: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5]
    )
    avg_approx_kl: List[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05]
    )


@dataclass
class MechanismDiscoveryConfig:
    """Opt-in targeted measurements; disabled for the original protocol."""

    enabled: bool = False
    base_anchor_steps: List[int] = field(default_factory=lambda: [0, 16, 48])
    refresh_lags: List[int] = field(
        default_factory=lambda: [1, 8, 16, 24, 32, 40, 48, 64]
    )
    online_leverage_scopes: List[str] = field(
        default_factory=lambda: [
            "full_history",
            "selector_candidate_history",
        ]
    )
    recent_exit_enabled: bool = True
    recent_exit_base_anchor: int = 0
    recent_exit_search_max_offset: int = 31
    recent_exit_relative_lags: List[int] = field(
        default_factory=lambda: [-1, 0, 1]
    )


@dataclass
class FunctionalProbeDiscoveryConfig:
    """Opt-in Stage-1 functional-staleness experiment."""

    enabled: bool = False
    total_budgets: List[int] = field(default_factory=lambda: [128, 256])
    protected_recent_sizes: List[int] = field(
        default_factory=lambda: [0, 32]
    )
    selectors: List[str] = field(
        default_factory=lambda: ["v_ridge_leverage", "snapkv"]
    )
    base_anchor_steps: List[int] = field(default_factory=lambda: [0, 48])
    probe_lags: List[int] = field(
        default_factory=lambda: [1, 8, 16, 24, 32, 40, 48, 64]
    )
    feature_variants: List[str] = field(
        default_factory=lambda: ["raw_v", "projected_v", "aov", "aor"]
    )
    feature_granularities: List[str] = field(
        default_factory=lambda: ["layer", "diagnostic_head"]
    )
    projection_chunk_size: int = 128
    identity_epsilon: float = 1e-12
    identity_denominator_floor: float = 1e-6
    save_full_reference_npz: bool = False


@dataclass
class TheoryClosingDiscoveryConfig:
    """Opt-in, small theory-closing mechanism matrix."""

    enabled: bool = False
    subset_anchor_step: int = 0
    subset_probe_step: int = 32
    subset_recent_size: int = 32
    candidate_pool_size: int = 16
    candidate_subset_size: int = 4
    candidate_random_seed: int = 42
    horizon_anchor_step: int = 0
    horizon_start_step: int = 32
    horizons: List[int] = field(
        default_factory=lambda: [1, 8, 16, 24, 32, 40, 48, 64]
    )
    total_budget: int = 128
    protected_recent_sizes: List[int] = field(
        default_factory=lambda: [0, 32]
    )
    selector: str = "v_ridge_leverage"
    feature_variants: List[str] = field(
        default_factory=lambda: ["raw_v", "projected_v", "aov", "aor"]
    )
    ridge_coefficient: float = 1e-3
    ridge_mode: str = "relative_pool_fixed"
    identity_epsilon: float = 1e-12
    identity_denominator_floor: float = 1e-6
    prediction_windows: List[int] = field(
        default_factory=lambda: [8, 16, 32]
    )
    ema_gammas: List[float] = field(
        default_factory=lambda: [0.5, 0.8, 0.9, 0.95, 0.98]
    )
    primary_objective_metric: str = "median_unit_spearman_proj_head"
    objective_partial_gate: float = 0.30
    horizon_partial_gate: float = 0.30
    monitoring_proxy_gate_spearman: float = 0.30
    monitoring_proxy_gate_delta_auprc: float = 0.05


@dataclass
class TrajectoryModelDiscoveryConfig:
    """Opt-in trajectory stochastic-model identification experiment."""

    enabled: bool = False
    anchors: List[int] = field(
        default_factory=lambda: [16, 32, 48, 64, 80]
    )
    horizon: int = 64
    diagnostic_layers: List[int] = field(
        default_factory=lambda: [0, 7, 14, 15, 21, 27]
    )
    betas: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    )
    protected_recent_sizes: List[int] = field(
        default_factory=lambda: [0, 32]
    )
    mask_types: List[str] = field(
        default_factory=lambda: [
            "attention",
            "aov",
            "aor",
            "v_ridge",
            "random",
            "old_core",
        ]
    )
    total_budget: int = 128
    scaling_relative_tolerance: float = 0.20
    scaling_r2_gate: float = 0.80
    superposition_relative_tolerance: float = 0.25
    latent_ranks: List[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32]
    )
    latent_variance_gate: float = 0.90
    markov_history_delta_r2_gate: float = 0.03
    rollout_r2_gate: float = 0.20
    ridge_alpha: float = 1e-3
    state_storage_dtype: str = "float16"
    random_seed: int = 42


@dataclass
class RobustEnvelopeDiscoveryConfig:
    """Opt-in robust perturbation-envelope experiment."""

    enabled: bool = False
    anchor: int = 32
    horizon: int = 64
    diagnostic_layers: List[int] = field(
        default_factory=lambda: [0, 7, 14, 15, 21, 27]
    )
    total_budget: int = 128
    protected_recent: int = 32
    static_strategies: List[str] = field(
        default_factory=lambda: [
            "attention",
            "v_ridge",
            "aov",
            "aor",
            "random",
        ]
    )
    subset_count: int = 12
    subset_horizon: int = 32
    jacobian_beta: float = 0.25
    coverage_levels: List[float] = field(
        default_factory=lambda: [0.90, 0.95]
    )
    evaluation_horizons: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64]
    )
    pointwise_coverage_gate: float = 0.90
    looseness_h8_gate: float = 5.0
    looseness_h32_gate: float = 10.0
    action_spearman_increment_gate: float = 0.05
    refresh_count: int = 3
    refresh_cost: float = 1.0
    random_seed: int = 42


@dataclass
class OutputSensitivityDiscoveryConfig:
    """Opt-in output-sensitivity and decision-calibration experiment."""

    enabled: bool = False
    anchors: List[int] = field(default_factory=lambda: [16, 32, 48])
    segment_horizon: int = 32
    state_reference_anchor: int = 32
    state_reference_horizon: int = 64
    diagnostic_layers: List[int] = field(
        default_factory=lambda: [0, 7, 14, 15, 21, 27]
    )
    total_budget: int = 128
    protected_recent: int = 32
    candidate_count: int = 24
    evaluation_horizons: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32]
    )
    state_evaluation_horizons: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64]
    )
    coverage_levels: List[float] = field(
        default_factory=lambda: [0.90, 0.95]
    )
    calibration_sequences: int = 8
    state_margin_sequences: int = 5
    jacobian_directions: int = 8
    jacobian_radii: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.05]
    )
    output_pointwise_coverage_gate: float = 0.90
    output_looseness_gate: float = 5.0
    action_spearman_increment_gate: float = 0.05
    pairwise_sign_accuracy_gate: float = 0.65
    partial_pairwise_sign_accuracy_gate: float = 0.60
    maximum_refresh_count: int = 3
    refresh_cost: float = 1.0
    random_seed: int = 42


@dataclass
class GaugeGeometryDiscoveryConfig:
    """Opt-in gauge-aware output geometry and Fisher-pullback experiment."""

    enabled: bool = False
    source_run_id: str = "output_sensitivity_4bit_24seq_seed42_v1"
    anchors: List[int] = field(default_factory=lambda: [16, 32, 48])
    segment_horizon: int = 32
    diagnostic_layers: List[int] = field(
        default_factory=lambda: [0, 7, 14, 15, 21, 27]
    )
    total_budget: int = 128
    protected_recent: int = 32
    candidate_count: int = 24
    evaluation_horizons: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32]
    )
    topk_values: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32, 64, 128, 256]
    )
    topk_tail_probability: float = 1.0e-6
    truncated_range_quantiles: List[float] = field(
        default_factory=lambda: [1.0e-5, 1.0e-4, 1.0e-3]
    )
    gauss_legendre_orders: List[int] = field(
        default_factory=lambda: [2, 3, 5]
    )
    dense_path_points: int = 9
    pullback_radii: List[float] = field(
        default_factory=lambda: [1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2]
    )
    pullback_sketch_sizes: List[int] = field(
        default_factory=lambda: [8, 16, 32, 64]
    )
    pullback_ranks: List[int] = field(
        default_factory=lambda: [4, 8, 16]
    )
    periodic_q_intervals: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16]
    )
    coverage_levels: List[float] = field(
        default_factory=lambda: [0.90, 0.95]
    )
    geometry_ratio_improvement_gate: float = 100.0
    quadrature_relative_error_gate: float = 1.0e-3
    q_pointwise_coverage_gate: float = 0.90
    q_median_looseness_gate: float = 5.0
    q_action_spearman_increment_gate: float = 0.05
    maximum_refresh_count: int = 3
    random_seed: int = 42


@dataclass
class IndependentFisherDiscoveryConfig:
    """Independent midpoint-Fisher replication and gated pullback experiment."""

    enabled: bool = False
    prior_run_ids: List[str] = field(
        default_factory=lambda: [
            "output_sensitivity_4bit_24seq_seed42_v1",
            "gauge_geometry_4bit_24seq_seed42_v1",
        ]
    )
    anchors: List[int] = field(default_factory=lambda: [16, 32, 48])
    segment_horizon: int = 16
    evaluation_horizons: List[int] = field(
        default_factory=lambda: [1, 4, 8, 16]
    )
    diagnostic_layers: List[int] = field(
        default_factory=lambda: [0, 7, 14, 15, 21, 27]
    )
    total_budget: int = 128
    protected_recent: int = 32
    candidate_count: int = 24
    stage_b_candidate_count: int = 8
    stage_b_candidate_sources: List[str] = field(
        default_factory=lambda: [
            "attention",
            "direct_energy_greedy",
            "v_ridge",
            "aor",
            "preceding_anchor_old_core",
            "stratified_random_00",
            "stratified_random_01",
            "stratified_random_02",
        ]
    )
    topk_values: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32, 64, 128, 256]
    )
    trust_top_k: int = 16
    adaptive_relative_tolerance: float = 1.0e-8
    adaptive_absolute_tolerance: float = 1.0e-10
    adaptive_subdivision_limit: int = 50
    pullback_radii: List[float] = field(
        default_factory=lambda: [1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2]
    )
    pullback_directions: int = 8
    pullback_power_iterations: int = 5
    pullback_sketch_sizes: List[int] = field(
        default_factory=lambda: [16, 32, 64]
    )
    pullback_ranks: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32]
    )
    periodic_q_intervals: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16]
    )
    maximum_refresh_count: int = 3
    refresh_horizons: List[int] = field(
        default_factory=lambda: [4, 8, 16]
    )
    g3_kl_spearman_gate: float = 0.90
    g3_action_spearman_gate: float = 0.85
    g3_median_symmetric_ratio_gate: float = 1.50
    g3_action_increment_gate: float = 0.10
    trust_precision_gate: float = 0.90
    trust_coverage_gate: float = 0.50
    trust_relative_error_gate: float = 0.10
    oracle_midpoint_kl_spearman_gate: float = 0.90
    oracle_midpoint_action_spearman_gate: float = 0.85
    fisher_direct_action_increment_gate: float = 0.05
    q_pointwise_coverage_gate: float = 0.90
    q_median_looseness_gate: float = 5.0
    q_action_spearman_increment_gate: float = 0.05
    random_seed: int = 20260726


@dataclass
class DiscoveryConfig:
    experiment_name: str = "temporal_cache_discovery"
    model: ModelDiscoveryConfig = field(default_factory=ModelDiscoveryConfig)
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    generation: GenerationDiscoveryConfig = field(
        default_factory=GenerationDiscoveryConfig
    )
    cache: CacheDiscoveryConfig = field(default_factory=CacheDiscoveryConfig)
    anchor_steps: List[int] = field(default_factory=lambda: [0, 16, 48])
    horizons: List[int] = field(default_factory=lambda: [1, 4, 16, 64])
    strategies: List[str] = field(
        default_factory=lambda: [
            "snapkv",
            "v_ridge_leverage",
            "attention_weighted_v_ridge_leverage",
            "future_attention_oracle",
        ]
    )
    signal_lags: List[int] = field(
        default_factory=lambda: [1, 4, 8, 16, 32, 64]
    )
    diagnostics: DiagnosticsDiscoveryConfig = field(
        default_factory=DiagnosticsDiscoveryConfig
    )
    metrics: MetricsDiscoveryConfig = field(default_factory=MetricsDiscoveryConfig)
    selectors: SelectorDiscoveryConfig = field(
        default_factory=SelectorDiscoveryConfig
    )
    validity_thresholds: ValidityThresholdsConfig = field(
        default_factory=ValidityThresholdsConfig
    )
    mechanism: MechanismDiscoveryConfig = field(
        default_factory=MechanismDiscoveryConfig
    )
    functional_probe: FunctionalProbeDiscoveryConfig = field(
        default_factory=FunctionalProbeDiscoveryConfig
    )
    theory_closing: TheoryClosingDiscoveryConfig = field(
        default_factory=TheoryClosingDiscoveryConfig
    )
    trajectory_model: TrajectoryModelDiscoveryConfig = field(
        default_factory=TrajectoryModelDiscoveryConfig
    )
    robust_envelope: RobustEnvelopeDiscoveryConfig = field(
        default_factory=RobustEnvelopeDiscoveryConfig
    )
    output_sensitivity: OutputSensitivityDiscoveryConfig = field(
        default_factory=OutputSensitivityDiscoveryConfig
    )
    gauge_geometry: GaugeGeometryDiscoveryConfig = field(
        default_factory=GaugeGeometryDiscoveryConfig
    )
    independent_fisher: IndependentFisherDiscoveryConfig = field(
        default_factory=IndependentFisherDiscoveryConfig
    )
    runtime: RuntimeDiscoveryConfig = field(default_factory=RuntimeDiscoveryConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        if self.model.backend == "torch":
            if self.model.name != "Qwen/Qwen2.5-1.5B-Instruct":
                raise ValueError(
                    "the Torch discovery backend requires "
                    "Qwen/Qwen2.5-1.5B-Instruct"
                )
            if self.model.dtype != "bfloat16":
                raise ValueError("the Torch discovery backend requires bfloat16")
            if self.model.quant_bits is not None:
                raise ValueError("the Torch discovery backend is not quantized")
        elif self.model.backend == "mlx":
            supported_mlx_models = {
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "mlx-community/Qwen3-8B-4bit",
            }
            if self.model.name not in supported_mlx_models:
                raise ValueError(
                    "the MLX discovery backend requires a supported 4-bit "
                    "checkpoint: %s" % sorted(supported_mlx_models)
                )
            if self.model.dtype not in {"4bit", "int4"}:
                raise ValueError("the MLX discovery backend requires dtype=4bit")
            if int(self.model.quant_bits or 0) != 4:
                raise ValueError("the MLX discovery backend requires quant_bits=4")
        else:
            raise ValueError("model.backend must be torch or mlx")
        if self.model.do_sample or self.generation.do_sample:
            raise ValueError("temporal discovery requires do_sample=false")
        if self.model.temperature != 0.0 or self.generation.temperature != 0.0:
            raise ValueError("temporal discovery requires temperature=0.0")
        expected = (
            int(self.cache.sink_size)
            + int(self.cache.recent_size)
            + int(self.cache.selected_core_budget)
        )
        if expected != int(self.cache.total_budget):
            raise ValueError(
                "sink_size + recent_size + selected_core_budget must equal total_budget"
            )
        if min(
            self.cache.total_budget,
            self.cache.sink_size,
            self.cache.recent_size,
            self.cache.selected_core_budget,
        ) < 0:
            raise ValueError("cache sizes must be non-negative")
        if not self.tasks:
            raise ValueError("at least one task is required")
        required = {
            "snapkv",
            "v_ridge_leverage",
            "attention_weighted_v_ridge_leverage",
            "future_attention_oracle",
        }
        if set(self.strategies) != required or len(self.strategies) != 4:
            raise ValueError("the discovery experiment requires exactly four strategies")
        if not self.anchor_steps or min(self.anchor_steps) < 0:
            raise ValueError("anchor_steps must be non-negative")
        if not self.horizons or min(self.horizons) <= 0:
            raise ValueError("horizons must be positive")
        if sorted(set(self.anchor_steps)) != sorted(self.anchor_steps):
            raise ValueError("anchor_steps must be sorted and unique")
        if sorted(set(self.horizons)) != sorted(self.horizons):
            raise ValueError("horizons must be sorted and unique")
        if self.generation.max_new_tokens < max(self.horizons):
            raise ValueError("max_new_tokens must cover the largest horizon")
        if self.diagnostics.layer_selection not in {
            "evenly_spaced",
            "explicit",
        }:
            raise ValueError(
                "diagnostic layer_selection must be evenly_spaced or explicit"
            )
        if self.diagnostics.layer_selection == "explicit":
            if (
                not self.diagnostics.explicit_layers
                or min(self.diagnostics.explicit_layers) < 0
                or sorted(set(self.diagnostics.explicit_layers))
                != sorted(self.diagnostics.explicit_layers)
            ):
                raise ValueError(
                    "explicit diagnostic layers must be sorted, unique, and non-negative"
                )
        if self.diagnostics.explicit_heads:
            if (
                min(self.diagnostics.explicit_heads) < 0
                or sorted(set(self.diagnostics.explicit_heads))
                != sorted(self.diagnostics.explicit_heads)
            ):
                raise ValueError(
                    "explicit diagnostic heads must be sorted, unique, and non-negative"
                )
        if self.selectors.snapkv_pooling not in {"max", "avg"}:
            raise ValueError("SnapKV pooling must be max or avg")
        if self.selectors.ridge_lambda_mode not in {"relative", "absolute"}:
            raise ValueError("ridge_lambda_mode must be relative or absolute")
        if self.runtime.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if self.mechanism.enabled:
            if not self.mechanism.base_anchor_steps:
                raise ValueError("mechanism.base_anchor_steps must be non-empty")
            if sorted(set(self.mechanism.base_anchor_steps)) != sorted(
                self.mechanism.base_anchor_steps
            ):
                raise ValueError(
                    "mechanism.base_anchor_steps must be sorted and unique"
                )
            if not self.mechanism.refresh_lags or min(
                self.mechanism.refresh_lags
            ) <= 0:
                raise ValueError("mechanism.refresh_lags must be positive")
            if sorted(set(self.mechanism.refresh_lags)) != sorted(
                self.mechanism.refresh_lags
            ):
                raise ValueError(
                    "mechanism.refresh_lags must be sorted and unique"
                )
            allowed_scopes = {
                "full_history",
                "selector_candidate_history",
            }
            if not set(self.mechanism.online_leverage_scopes).issubset(
                allowed_scopes
            ):
                raise ValueError(
                    "mechanism.online_leverage_scopes contains an unknown scope"
                )
            if self.mechanism.recent_exit_base_anchor not in (
                self.mechanism.base_anchor_steps
            ):
                raise ValueError(
                    "recent_exit_base_anchor must be a mechanism base anchor"
                )
            if self.mechanism.recent_exit_search_max_offset <= 0:
                raise ValueError(
                    "recent_exit_search_max_offset must be positive"
                )
            maximum_required = max(
                int(anchor) + int(lag) + 1
                for anchor in self.mechanism.base_anchor_steps
                for lag in self.mechanism.refresh_lags
            )
            if maximum_required > int(self.generation.max_new_tokens):
                raise ValueError(
                    "generation.max_new_tokens must cover mechanism base+lag+1"
                )
        if self.functional_probe.enabled:
            probe = self.functional_probe
            if not probe.total_budgets or min(probe.total_budgets) <= 0:
                raise ValueError(
                    "functional_probe.total_budgets must be positive"
                )
            if sorted(set(probe.total_budgets)) != sorted(
                probe.total_budgets
            ):
                raise ValueError(
                    "functional_probe.total_budgets must be sorted and unique"
                )
            if not probe.protected_recent_sizes or min(
                probe.protected_recent_sizes
            ) < 0:
                raise ValueError(
                    "functional_probe.protected_recent_sizes must be non-negative"
                )
            if sorted(set(probe.protected_recent_sizes)) != sorted(
                probe.protected_recent_sizes
            ):
                raise ValueError(
                    "functional_probe.protected_recent_sizes must be sorted and unique"
                )
            allowed_selectors = {
                "v_ridge_leverage",
                "snapkv",
                "attention_weighted_v_ridge_leverage",
            }
            if (
                not probe.selectors
                or not set(probe.selectors).issubset(allowed_selectors)
            ):
                raise ValueError(
                    "functional_probe.selectors contains an unknown selector"
                )
            if sorted(set(probe.base_anchor_steps)) != sorted(
                probe.base_anchor_steps
            ) or not probe.base_anchor_steps:
                raise ValueError(
                    "functional_probe.base_anchor_steps must be sorted, unique, and non-empty"
                )
            if min(probe.base_anchor_steps) < 0:
                raise ValueError(
                    "functional_probe.base_anchor_steps must be non-negative"
                )
            if (
                sorted(set(probe.probe_lags)) != sorted(probe.probe_lags)
                or not probe.probe_lags
                or min(probe.probe_lags) <= 0
            ):
                raise ValueError(
                    "functional_probe.probe_lags must be sorted, unique, and positive"
                )
            allowed_features = {"raw_v", "projected_v", "aov", "aor"}
            if set(probe.feature_variants) != allowed_features:
                raise ValueError(
                    "functional_probe requires raw_v/projected_v/aov/aor"
                )
            allowed_granularities = {"layer", "diagnostic_head"}
            if (
                not probe.feature_granularities
                or not set(probe.feature_granularities).issubset(
                    allowed_granularities
                )
            ):
                raise ValueError(
                    "functional_probe.feature_granularities is invalid"
                )
            if probe.projection_chunk_size <= 0:
                raise ValueError(
                    "functional_probe.projection_chunk_size must be positive"
                )
            if probe.identity_epsilon <= 0:
                raise ValueError(
                    "functional_probe.identity_epsilon must be positive"
                )
            if probe.identity_denominator_floor <= 0:
                raise ValueError(
                    "functional_probe.identity_denominator_floor must be positive"
                )
            maximum_required = max(
                int(anchor) + int(lag) + 1
                for anchor in probe.base_anchor_steps
                for lag in probe.probe_lags
            )
            if maximum_required > int(self.generation.max_new_tokens):
                raise ValueError(
                    "generation.max_new_tokens must cover functional base+lag+1"
                )
            for budget in probe.total_budgets:
                for protected_recent in probe.protected_recent_sizes:
                    effective_recent = max(1, int(protected_recent))
                    core = (
                        int(budget)
                        - int(self.cache.sink_size)
                        - effective_recent
                    )
                    if core <= 0:
                        raise ValueError(
                            "functional cache allocation leaves no historical core"
                        )
        if self.theory_closing.enabled:
            theory = self.theory_closing
            if self.functional_probe.enabled:
                raise ValueError(
                    "theory_closing and functional_probe cannot be enabled together"
                )
            if theory.subset_anchor_step < 0:
                raise ValueError(
                    "theory_closing.subset_anchor_step must be non-negative"
                )
            if theory.subset_probe_step <= theory.subset_anchor_step:
                raise ValueError(
                    "theory_closing subset probe must follow its anchor"
                )
            if theory.candidate_pool_size <= 0:
                raise ValueError(
                    "theory_closing candidate_pool_size must be positive"
                )
            if not 0 < theory.candidate_subset_size <= theory.candidate_pool_size:
                raise ValueError(
                    "theory_closing candidate subset size is invalid"
                )
            if (
                not theory.horizons
                or min(theory.horizons) <= 0
                or sorted(set(theory.horizons)) != sorted(theory.horizons)
            ):
                raise ValueError(
                    "theory_closing horizons must be sorted, unique, and positive"
                )
            if (
                not theory.protected_recent_sizes
                or min(theory.protected_recent_sizes) < 0
            ):
                raise ValueError(
                    "theory_closing protected_recent_sizes must be non-negative"
                )
            if theory.selector not in {
                "v_ridge_leverage",
                "snapkv",
                "attention_weighted_v_ridge_leverage",
            }:
                raise ValueError("theory_closing selector is unsupported")
            if set(theory.feature_variants) != {
                "raw_v",
                "projected_v",
                "aov",
                "aor",
            }:
                raise ValueError(
                    "theory_closing requires Raw-V/projected-V/AOV/AOR"
                )
            if theory.ridge_coefficient <= 0:
                raise ValueError(
                    "theory_closing ridge coefficient must be positive"
                )
            if theory.ridge_mode != "relative_pool_fixed":
                raise ValueError(
                    "theory_closing uses the pre-registered relative_pool_fixed ridge"
                )
            if (
                not theory.prediction_windows
                or min(theory.prediction_windows) <= 0
            ):
                raise ValueError(
                    "theory_closing prediction windows must be positive"
                )
            if (
                not theory.ema_gammas
                or min(theory.ema_gammas) <= 0
                or max(theory.ema_gammas) >= 1
            ):
                raise ValueError(
                    "theory_closing EMA gammas must lie strictly in (0,1)"
                )
            maximum_required = max(
                theory.subset_probe_step,
                theory.horizon_start_step + max(theory.horizons),
            )
            if maximum_required > int(self.generation.max_new_tokens):
                raise ValueError(
                    "generation.max_new_tokens must cover theory-closing horizons"
                )
            for recent in theory.protected_recent_sizes:
                effective_recent = max(1, int(recent))
                if (
                    int(theory.total_budget)
                    - int(self.cache.sink_size)
                    - effective_recent
                    <= 0
                ):
                    raise ValueError(
                        "theory_closing cache allocation leaves no core"
                    )
        if self.trajectory_model.enabled:
            trajectory = self.trajectory_model
            if self.functional_probe.enabled or self.theory_closing.enabled:
                raise ValueError(
                    "trajectory_model cannot share a run with functional/theory-closing"
                )
            if (
                not trajectory.anchors
                or min(trajectory.anchors) < 0
                or sorted(set(trajectory.anchors))
                != sorted(trajectory.anchors)
            ):
                raise ValueError(
                    "trajectory anchors must be sorted, unique, and non-negative"
                )
            if trajectory.horizon <= 0:
                raise ValueError("trajectory horizon must be positive")
            if (
                not trajectory.diagnostic_layers
                or min(trajectory.diagnostic_layers) < 0
            ):
                raise ValueError(
                    "trajectory diagnostic layers must be non-negative"
                )
            if (
                not trajectory.betas
                or 0.0 not in trajectory.betas
                or 1.0 not in trajectory.betas
                or min(trajectory.betas) < 0
            ):
                raise ValueError(
                    "trajectory betas must include 0 and 1 and be non-negative"
                )
            allowed_masks = {
                "attention",
                "aov",
                "aor",
                "v_ridge",
                "random",
                "old_core",
            }
            if set(trajectory.mask_types) != allowed_masks:
                raise ValueError(
                    "trajectory mask types must be the pre-registered six"
                )
            if trajectory.state_storage_dtype not in {
                "float16",
                "float32",
            }:
                raise ValueError(
                    "trajectory state storage dtype must be float16/float32"
                )
            if (
                not trajectory.latent_ranks
                or min(trajectory.latent_ranks) <= 0
            ):
                raise ValueError("trajectory latent ranks must be positive")
            maximum_required = (
                max(trajectory.anchors) + trajectory.horizon
            )
            if maximum_required > int(self.generation.max_new_tokens):
                raise ValueError(
                    "generation.max_new_tokens must cover trajectory anchor+horizon"
                )
        if self.robust_envelope.enabled:
            envelope = self.robust_envelope
            if (
                self.functional_probe.enabled
                or self.theory_closing.enabled
                or self.trajectory_model.enabled
                or self.output_sensitivity.enabled
                or self.gauge_geometry.enabled
            ):
                raise ValueError(
                    "robust_envelope cannot share a run with other opt-in protocols"
                )
            if envelope.anchor < 0 or envelope.horizon <= 0:
                raise ValueError("robust envelope anchor/horizon are invalid")
            if (
                envelope.anchor + envelope.horizon
                > int(self.generation.max_new_tokens)
            ):
                raise ValueError(
                    "generation.max_new_tokens must cover robust envelope horizon"
                )
            if (
                not envelope.diagnostic_layers
                or min(envelope.diagnostic_layers) < 0
            ):
                raise ValueError(
                    "robust envelope diagnostic layers are invalid"
                )
            if set(envelope.static_strategies) != {
                "attention",
                "v_ridge",
                "aov",
                "aor",
                "random",
            }:
                raise ValueError(
                    "robust envelope strategies must be the pre-registered five"
                )
            if (
                envelope.total_budget
                - self.cache.sink_size
                - envelope.protected_recent
                <= 0
            ):
                raise ValueError(
                    "robust envelope cache allocation leaves no core"
                )
            if (
                envelope.subset_count < 4
                or envelope.subset_horizon <= 0
                or not 0 < envelope.jacobian_beta <= 1
            ):
                raise ValueError(
                    "robust envelope subset/probe design is invalid"
                )
            if any(
                not 0 < float(value) < 1
                for value in envelope.coverage_levels
            ):
                raise ValueError(
                    "robust envelope coverage levels must lie in (0,1)"
                )
        if self.output_sensitivity.enabled:
            output = self.output_sensitivity
            if (
                self.functional_probe.enabled
                or self.theory_closing.enabled
                or self.trajectory_model.enabled
                or self.robust_envelope.enabled
                or self.gauge_geometry.enabled
            ):
                raise ValueError(
                    "output_sensitivity cannot share a run with other opt-in protocols"
                )
            if len(output.anchors) != len(set(output.anchors)):
                raise ValueError("output-sensitivity anchors must be unique")
            if not output.anchors or min(output.anchors) < 0:
                raise ValueError("output-sensitivity anchors are invalid")
            maximum_required = max(
                max(output.anchors) + int(output.segment_horizon),
                int(output.state_reference_anchor)
                + int(output.state_reference_horizon),
            )
            if maximum_required > int(self.generation.max_new_tokens):
                raise ValueError(
                    "generation.max_new_tokens must cover output-sensitivity horizons"
                )
            if int(output.candidate_count) < 24:
                raise ValueError(
                    "output-sensitivity requires at least 24 candidates"
                )
            if int(output.segment_horizon) < max(output.evaluation_horizons):
                raise ValueError("output-sensitivity segment horizon is too short")
            if int(output.jacobian_directions) < 8:
                raise ValueError(
                    "output-sensitivity requires at least 8 directions"
                )
            if (
                len(output.jacobian_radii) < 3
                or min(output.jacobian_radii) <= 0
            ):
                raise ValueError(
                    "output-sensitivity requires 3 positive radii"
                )
            if output.diagnostic_layers != self.diagnostics.explicit_layers:
                raise ValueError(
                    "output-sensitivity layers must match explicit diagnostics"
                )
            if (
                output.total_budget
                - self.cache.sink_size
                - output.protected_recent
                <= 0
            ):
                raise ValueError(
                    "output-sensitivity cache allocation leaves no core"
                )
            if any(
                not 0 < float(value) < 1
                for value in output.coverage_levels
            ):
                raise ValueError(
                    "output-sensitivity coverage levels must lie in (0,1)"
                )
        if self.gauge_geometry.enabled:
            gauge = self.gauge_geometry
            if (
                self.functional_probe.enabled
                or self.theory_closing.enabled
                or self.trajectory_model.enabled
                or self.robust_envelope.enabled
                or self.output_sensitivity.enabled
            ):
                raise ValueError(
                    "gauge_geometry cannot share a run with other opt-in protocols"
                )
            if not gauge.anchors or min(gauge.anchors) < 0:
                raise ValueError("gauge-geometry anchors are invalid")
            if sorted(set(gauge.anchors)) != sorted(gauge.anchors):
                raise ValueError(
                    "gauge-geometry anchors must be sorted and unique"
                )
            if (
                max(gauge.anchors) + int(gauge.segment_horizon)
                > int(self.generation.max_new_tokens)
            ):
                raise ValueError(
                    "generation.max_new_tokens must cover gauge geometry horizons"
                )
            if int(gauge.candidate_count) != 24:
                raise ValueError(
                    "gauge geometry requires exactly 24 inherited candidates"
                )
            if (
                int(gauge.segment_horizon)
                < max(int(value) for value in gauge.evaluation_horizons)
            ):
                raise ValueError("gauge geometry segment horizon is too short")
            if gauge.diagnostic_layers != self.diagnostics.explicit_layers:
                raise ValueError(
                    "gauge geometry layers must match explicit diagnostics"
                )
            if (
                gauge.total_budget
                - self.cache.sink_size
                - gauge.protected_recent
                <= 0
            ):
                raise ValueError(
                    "gauge geometry cache allocation leaves no core"
                )
            if (
                sorted(set(gauge.topk_values)) != sorted(gauge.topk_values)
                or min(gauge.topk_values) < 4
            ):
                raise ValueError(
                    "gauge geometry top-k values must be sorted and at least 4"
                )
            if set(gauge.gauss_legendre_orders) != {2, 3, 5}:
                raise ValueError(
                    "gauge geometry requires 2/3/5-point Gauss-Legendre"
                )
            if int(gauge.dense_path_points) != 9:
                raise ValueError("gauge geometry dense reference requires 9 points")
            if (
                len(gauge.pullback_radii) < 4
                or min(gauge.pullback_radii) <= 0
            ):
                raise ValueError(
                    "gauge geometry requires four positive pullback radii"
                )
            if any(
                not 0 < float(value) < 1
                for value in gauge.coverage_levels
            ):
                raise ValueError(
                    "gauge geometry coverage levels must lie in (0,1)"
                )
        if self.independent_fisher.enabled:
            independent = self.independent_fisher
            if (
                self.functional_probe.enabled
                or self.theory_closing.enabled
                or self.trajectory_model.enabled
                or self.robust_envelope.enabled
                or self.output_sensitivity.enabled
                or self.gauge_geometry.enabled
            ):
                raise ValueError(
                    "independent_fisher cannot share a run with other opt-in protocols"
                )
            if independent.anchors != [16, 32, 48]:
                raise ValueError(
                    "independent Fisher replication requires anchors 16/32/48"
                )
            if independent.evaluation_horizons != [1, 4, 8, 16]:
                raise ValueError(
                    "independent Fisher replication requires horizons 1/4/8/16"
                )
            if (
                max(independent.anchors) + int(independent.segment_horizon)
                > int(self.generation.max_new_tokens)
            ):
                raise ValueError(
                    "generation.max_new_tokens must cover independent Fisher horizons"
                )
            if int(independent.candidate_count) != 24:
                raise ValueError(
                    "independent Fisher Stage A requires exactly 24 candidates"
                )
            if (
                int(independent.stage_b_candidate_count) != 8
                or len(independent.stage_b_candidate_sources) != 8
                or len(set(independent.stage_b_candidate_sources)) != 8
            ):
                raise ValueError(
                    "independent Fisher Stage B requires eight fixed candidates"
                )
            if (
                independent.diagnostic_layers
                != self.diagnostics.explicit_layers
            ):
                raise ValueError(
                    "independent Fisher layers must match explicit diagnostics"
                )
            if (
                independent.total_budget
                - self.cache.sink_size
                - independent.protected_recent
                <= 0
            ):
                raise ValueError(
                    "independent Fisher cache allocation leaves no core"
                )
            if (
                len(independent.pullback_radii) != 4
                or min(independent.pullback_radii) <= 0
            ):
                raise ValueError(
                    "independent Fisher requires four positive pullback radii"
                )
            if int(independent.pullback_directions) < 8:
                raise ValueError(
                    "independent Fisher requires at least eight directions"
                )
            if (
                float(independent.adaptive_relative_tolerance) != 1.0e-8
                or float(independent.adaptive_absolute_tolerance) != 1.0e-10
            ):
                raise ValueError(
                    "independent Fisher adaptive tolerances are preregistered"
                )

    def captured_anchor_steps(self) -> List[int]:
        """All states that must be retained during the reference trajectory."""

        result = set(int(value) for value in self.anchor_steps)
        if self.mechanism.enabled:
            bases = [int(value) for value in self.mechanism.base_anchor_steps]
            lags = [int(value) for value in self.mechanism.refresh_lags]
            result.update(bases)
            result.update(base + lag for base in bases for lag in lags)
            if self.mechanism.recent_exit_enabled:
                base = int(self.mechanism.recent_exit_base_anchor)
                window = int(self.cache.recent_size)
                maximum = int(self.mechanism.recent_exit_search_max_offset)
                relative = [
                    int(value)
                    for value in self.mechanism.recent_exit_relative_lags
                ]
                for offset in range(1, maximum + 1):
                    exit_lag = offset + window
                    result.update(
                        base + exit_lag + delta for delta in relative
                    )
        if self.functional_probe.enabled:
            bases = [
                int(value)
                for value in self.functional_probe.base_anchor_steps
            ]
            lags = [
                int(value) for value in self.functional_probe.probe_lags
            ]
            result.update(bases)
            result.update(base + lag for base in bases for lag in lags)
        if self.theory_closing.enabled:
            theory = self.theory_closing
            result.update(
                {
                    int(theory.subset_anchor_step),
                    int(theory.subset_probe_step),
                    int(theory.horizon_anchor_step),
                    int(theory.horizon_start_step),
                }
            )
            result.update(
                range(
                    int(theory.horizon_start_step),
                    int(theory.horizon_start_step)
                    + max(int(value) for value in theory.horizons),
                )
            )
        if self.trajectory_model.enabled:
            # old/fresh masks at the first registered trajectory anchor need a
            # pre-intervention source state.  Capturing anchor zero is cheap and
            # keeps that source explicit instead of silently substituting the
            # current anchor.
            result.add(0)
            result.update(
                int(value) for value in self.trajectory_model.anchors
            )
        if self.robust_envelope.enabled:
            result.add(int(self.robust_envelope.anchor))
        if self.output_sensitivity.enabled:
            result.add(0)
            result.update(
                int(value) for value in self.output_sensitivity.anchors
            )
            result.add(
                int(self.output_sensitivity.state_reference_anchor)
            )
        if self.gauge_geometry.enabled:
            result.add(0)
            result.update(int(value) for value in self.gauge_geometry.anchors)
        if self.independent_fisher.enabled:
            result.add(0)
            result.update(
                int(value) for value in self.independent_fisher.anchors
            )
        return sorted(
            value
            for value in result
            if 0 <= value <= int(self.generation.max_new_tokens)
        )


T = TypeVar("T")


def _strict_dataclass(cls: Type[T], value: Dict[str, Any], path: str) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("unknown fields at %s: %s" % (path, unknown))
    return cls(**value)


def load_discovery_config(path: str) -> DiscoveryConfig:
    resolved = Path(os.path.expandvars(path)).resolve()
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    allowed = {item.name for item in fields(DiscoveryConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("unknown top-level discovery fields: %s" % unknown)
    cfg = DiscoveryConfig(
        experiment_name=str(
            data.get("experiment_name", "temporal_cache_discovery")
        ),
        model=_strict_dataclass(
            ModelDiscoveryConfig, data.get("model", {}), "model"
        ),
        tasks=dict(data.get("tasks", {})),
        generation=_strict_dataclass(
            GenerationDiscoveryConfig, data.get("generation", {}), "generation"
        ),
        cache=_strict_dataclass(
            CacheDiscoveryConfig, data.get("cache", {}), "cache"
        ),
        anchor_steps=[int(value) for value in data.get("anchor_steps", [0, 16, 48])],
        horizons=[int(value) for value in data.get("horizons", [1, 4, 16, 64])],
        strategies=list(data.get("strategies", DiscoveryConfig().strategies)),
        signal_lags=[
            int(value)
            for value in data.get("signal_lags", [1, 4, 8, 16, 32, 64])
        ],
        diagnostics=_strict_dataclass(
            DiagnosticsDiscoveryConfig, data.get("diagnostics", {}), "diagnostics"
        ),
        metrics=_strict_dataclass(
            MetricsDiscoveryConfig, data.get("metrics", {}), "metrics"
        ),
        selectors=_strict_dataclass(
            SelectorDiscoveryConfig, data.get("selectors", {}), "selectors"
        ),
        validity_thresholds=_strict_dataclass(
            ValidityThresholdsConfig,
            data.get("validity_thresholds", {}),
            "validity_thresholds",
        ),
        mechanism=_strict_dataclass(
            MechanismDiscoveryConfig,
            data.get("mechanism", {}),
            "mechanism",
        ),
        functional_probe=_strict_dataclass(
            FunctionalProbeDiscoveryConfig,
            data.get("functional_probe", {}),
            "functional_probe",
        ),
        theory_closing=_strict_dataclass(
            TheoryClosingDiscoveryConfig,
            data.get("theory_closing", {}),
            "theory_closing",
        ),
        trajectory_model=_strict_dataclass(
            TrajectoryModelDiscoveryConfig,
            data.get("trajectory_model", {}),
            "trajectory_model",
        ),
        robust_envelope=_strict_dataclass(
            RobustEnvelopeDiscoveryConfig,
            data.get("robust_envelope", {}),
            "robust_envelope",
        ),
        output_sensitivity=_strict_dataclass(
            OutputSensitivityDiscoveryConfig,
            data.get("output_sensitivity", {}),
            "output_sensitivity",
        ),
        gauge_geometry=_strict_dataclass(
            GaugeGeometryDiscoveryConfig,
            data.get("gauge_geometry", {}),
            "gauge_geometry",
        ),
        independent_fisher=_strict_dataclass(
            IndependentFisherDiscoveryConfig,
            data.get("independent_fisher", {}),
            "independent_fisher",
        ),
        runtime=_strict_dataclass(
            RuntimeDiscoveryConfig, data.get("runtime", {}), "runtime"
        ),
    )
    cfg.validate()
    return cfg
