#!/usr/bin/env python3
"""Generate the evidence matrix, full report, David update, and manifest."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    cluster_bootstrap_statistic,
    ensure_directory,
    json_dump,
    sha256_file,
    write_dual,
)


def _correlation(corr: pd.DataFrame, relationship: str) -> pd.Series:
    row = corr[corr["relationship"].eq(relationship)]
    if row.empty:
        raise KeyError(relationship)
    return row.iloc[0]


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _ci(row: pd.Series, prefix: str = "spearman") -> str:
    return (
        f"{_fmt(row[prefix])} "
        f"[{_fmt(row[prefix + '_ci_low'])}, {_fmt(row[prefix + '_ci_high'])}]"
    )


def _cluster_summary(
    frame: pd.DataFrame, value: str, statistic: str = "mean"
) -> tuple[float, float, float, int]:
    return cluster_bootstrap_statistic(
        frame,
        "sample_cluster",
        value,
        statistic,
        np.random.default_rng(20260724),
        2000,
    )


def _markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    view = frame.copy()
    if include_index:
        view = view.reset_index()
    columns = [str(column) for column in view.columns]

    def cell(value: object) -> str:
        if isinstance(value, float):
            value = f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def build(analysis_dir: Path, input_dir: Path) -> None:
    analysis_dir = Path(analysis_dir)
    table = analysis_dir / "tables"
    inventory = pd.read_parquet(table / "run_inventory.parquet").iloc[0]
    quality = pd.read_parquet(table / "data_quality_issues.parquet")
    horizon = pd.read_parquet(table / "per_horizon_metrics.parquet")
    score = pd.read_parquet(table / "score_stability.parquet")
    sets = pd.read_parquet(table / "set_stability.parquet")
    residual = pd.read_parquet(table / "future_selected_core_residuals.parquet")
    geometry = pd.read_parquet(table / "geometry_anchor_summaries.parquet")
    refresh = pd.read_parquet(table / "refresh_benefit_analysis.parquet")
    validity = pd.read_parquet(table / "validity_horizon_sensitivity.parquet")
    rankings = pd.read_parquet(table / "selector_horizon_rankings.parquet")
    oracle = pd.read_parquet(table / "future_oracle_horizon_overlap.parquet")
    reversals = pd.read_parquet(table / "per_sample_horizon_rank_reversals.parquet")
    corr = pd.read_parquet(table / "mechanism_correlation_summary.parquet")
    events = pd.read_parquet(table / "direction_shift_events.parquet")
    figures = pd.read_parquet(analysis_dir / "figures/figure_manifest.parquet")
    mechanism_dir = analysis_dir / "mechanism_targeted"
    mechanism_addendum_path = mechanism_dir / "final_report_addendum.md"
    david_addendum_path = mechanism_dir / "david_addendum.md"

    global_refresh = refresh[refresh["record_scope"].eq("global_output")].copy()
    refresh_all = _cluster_summary(global_refresh, "refresh_benefit", "median")
    refresh_by_lag = global_refresh.groupby("refresh_lag")["refresh_benefit"].agg(
        ["mean", "median", lambda x: (x > 0).mean()]
    )
    refresh_by_lag.columns = ["mean", "median", "fraction_positive"]
    refresh_by_task = global_refresh.groupby("task")["refresh_benefit"].agg(
        ["mean", "median"]
    )
    h64 = horizon[horizon["horizon"].eq(64)]
    h64_summary = {}
    for strategy, group in h64.groupby("strategy"):
        h64_summary[strategy] = _cluster_summary(group, "avg_delta_nll", "mean")

    stability64 = (
        score[score["lag"].eq(64)]
        .groupby("strategy")[["spearman_rank_correlation", "top_core_jaccard"]]
        .median()
    )
    oracle_pair = oracle.groupby(["horizon_left", "horizon_right"])[
        "mean_jaccard"
    ].median()
    residual_by_lag = residual.groupby("lag")["future_new_token_residual"].median()
    effective_rank = geometry.groupby("anchor")["effective_rank"].median()

    score_to_set = _correlation(corr, "score drift -> selected-set turnover")
    residual_to_score = _correlation(
        corr, "selected-core residual -> score instability"
    )
    residual_to_set = _correlation(corr, "selected-core residual -> set turnover")
    residual_to_refresh = _correlation(
        corr, "selected-core residual -> sparse refresh benefit"
    )
    set_to_loss = _correlation(corr, "selected-set turnover -> delta NLL")
    set_to_refresh = _correlation(
        corr, "selected-set turnover -> sparse refresh benefit"
    )
    attention_to_loss = _correlation(corr, "attention-output error -> delta NLL")
    attention_to_refresh = _correlation(
        corr, "attention drift -> sparse refresh benefit"
    )
    stale_to_refresh = _correlation(corr, "stale loss -> sparse refresh benefit")
    margin_to_life = _correlation(
        corr, "selection margin -> empirical validity horizon"
    )

    # Concrete mismatch cases show why set and functional stability must remain
    # separate. Thresholds are global quartiles and are explicitly exploratory.
    links = pd.read_parquet(table / "mechanism_links.parquet")
    set_loss = links[
        links["relationship"].eq("selected-set turnover -> delta NLL")
    ].dropna(subset=["x_value", "y_value"])
    turn_hi, turn_lo = set_loss["x_value"].quantile([0.75, 0.25])
    loss_abs = set_loss["y_value"].abs()
    loss_hi, loss_lo = loss_abs.quantile([0.75, 0.25])
    redundancy_cases = int(((set_loss["x_value"] >= turn_hi) & (loss_abs <= loss_lo)).sum())
    hidden_error_cases = int(((set_loss["x_value"] <= turn_lo) & (loss_abs >= loss_hi)).sum())

    rank_change = (
        reversals.groupby("strategy")["rank_reversal_1_to_64"].mean().sort_values()
    )
    threshold_medians = (
        validity[
            validity["definition"].isin(
                ["absolute_average_delta_nll", "sample_normalized_percentile"]
            )
        ]
        .groupby(["definition", "threshold_label", "strategy"])["observed_horizon"]
        .median()
    )

    evidence = pd.DataFrame(
        [
            {
                "Explanation": "A. Score persistence",
                "Supporting observations": (
                    "At lag 64, median Spearman/Jaccard were "
                    f"{stability64.loc['v_ridge_leverage','spearman_rank_correlation']:.3f}/"
                    f"{stability64.loc['v_ridge_leverage','top_core_jaccard']:.3f} for V-ridge "
                    "and "
                    f"{stability64.loc['attention_weighted_v_ridge_leverage','spearman_rank_correlation']:.3f}/"
                    f"{stability64.loc['attention_weighted_v_ridge_leverage','top_core_jaccard']:.3f} "
                    "for hybrid. Score drift and turnover were associated."
                ),
                "Contradicting observations": (
                    "SnapKV lag-64 stability was lower "
                    f"({stability64.loc['snapkv','spearman_rank_correlation']:.3f}/"
                    f"{stability64.loc['snapkv','top_core_jaccard']:.3f}), and sparse refresh "
                    "benefit sometimes remained material."
                ),
                "Strength": "moderate signal",
                "Main confounders": (
                    "Old-token-only score universe; only three diagnostic layers; "
                    "no dense refreshed arm."
                ),
            },
            {
                "Explanation": "B. Functional redundancy",
                "Supporting observations": (
                    f"Turnover→ΔNLL Spearman was only {_ci(set_to_loss)}; "
                    f"{redundancy_cases} exploratory high-turnover/low-|loss| layer records exist."
                ),
                "Contradicting observations": (
                    f"{hidden_error_cases} low-turnover/high-|loss| records also exist, and "
                    f"turnover→refresh Spearman was {_ci(set_to_refresh)}."
                ),
                "Strength": "mixed",
                "Main confounders": (
                    "Layer-level set turnover is paired with a global loss; token identities "
                    "and downstream states are absent."
                ),
            },
            {
                "Explanation": "C. Recent-window absorption",
                "Supporting observations": (
                    "The implementation guarantees a rolling 32-token recent window, "
                    "which makes the mechanism plausible."
                ),
                "Contradicting observations": (
                    "No token-level event identity or dense benefit measurement near recent-window exit."
                ),
                "Strength": "unavailable",
                "Main confounders": "The necessary exit-aligned counterfactual was not saved.",
            },
            {
                "Explanation": "D. Stable geometry with changing attention",
                "Supporting observations": (
                    "Anchor selected-core effective rank and selected-span residual summaries exist."
                ),
                "Contradicting observations": (
                    "No time-resolved covariance/subspace metric exists, so geometry and attention "
                    "timescales cannot be compared."
                ),
                "Strength": "unavailable",
                "Main confounders": (
                    "Only anchor selected-core spectra; no future-window V matrices/sketches."
                ),
            },
            {
                "Explanation": "E. Downstream insensitivity",
                "Supporting observations": (
                    f"Attention-output error→ΔNLL Spearman was {_ci(attention_to_loss)}, "
                    "leaving substantial dispersion compatible with attenuation."
                ),
                "Contradicting observations": (
                    "The positive association is not zero, and only six diagnostic query heads "
                    "before output projection were measured."
                ),
                "Strength": "mixed",
                "Main confounders": "No post-projection or downstream-layer perturbation tracing.",
            },
            {
                "Explanation": "F. Sparse regime changes",
                "Supporting observations": (
                    "Refresh-benefit means exceed medians at saved boundaries, indicating a "
                    "right-skewed distribution driven by a subset of sample/transition cases."
                ),
                "Contradicting observations": (
                    "Only three boundary lags are observed; event sparsity and temporal alignment "
                    "cannot be measured densely."
                ),
                "Strength": "weak signal",
                "Main confounders": "Sparse anchors and post-hoc event thresholds.",
            },
            {
                "Explanation": "G. Horizon-conditioned optimality",
                "Supporting observations": (
                    "Median future-oracle Jaccard declines from "
                    f"{oracle_pair.loc[(1,4)]:.3f} (H1/H4) to "
                    f"{oracle_pair.loc[(1,64)]:.3f} (H1/H64), and per-sample selector "
                    "ranks can change from H1 to H64."
                ),
                "Contradicting observations": (
                    f"Adjacent H16/H64 oracle overlap remains {oracle_pair.loc[(16,64)]:.3f}; "
                    "rank changes are heterogeneous."
                ),
                "Strength": "moderate signal",
                "Main confounders": (
                    "Oracle selects only anchor-existing tokens and is not deployable."
                ),
            },
        ]
    )
    evidence_updates_path = (
        mechanism_dir / "hypothesis_evidence_updates.parquet"
    )
    if evidence_updates_path.exists():
        evidence_updates = pd.read_parquet(evidence_updates_path)
        missing = sorted(
            set(evidence_updates["Explanation"])
            - set(evidence["Explanation"])
        )
        if missing:
            raise ValueError(
                "mechanism evidence updates do not match base rows: "
                + ", ".join(missing)
            )
        evidence = evidence.set_index("Explanation")
        for _, update in evidence_updates.iterrows():
            explanation = update["Explanation"]
            for column in evidence_updates.columns:
                if column != "Explanation":
                    evidence.loc[explanation, column] = update[column]
        evidence = evidence.reset_index()
    allowed_strength = {
        "strong signal",
        "moderate signal",
        "weak signal",
        "mixed",
        "unsupported",
        "unavailable",
    }
    if not set(evidence["Strength"]).issubset(allowed_strength):
        raise ValueError("invalid evidence strength label")
    evidence.to_csv(table / "hypothesis_evidence_matrix.csv", index=False)
    evidence.to_parquet(table / "hypothesis_evidence_matrix.parquet", index=False)

    h64_lines = []
    for strategy in [
        "snapkv",
        "v_ridge_leverage",
        "attention_weighted_v_ridge_leverage",
        "future_attention_oracle",
    ]:
        point, low, high, n = h64_summary[strategy]
        h64_lines.append(
            f"- `{strategy}`: mean avg-ΔNLL {point:.3f} "
            f"(95% sample-cluster bootstrap CI {low:.3f}–{high:.3f}; n={n})."
        )
    refresh_lines = [
        f"- lag {int(lag)}: mean {row['mean']:.3f}, median {row['median']:.3f}, "
        f"positive fraction {row['fraction_positive']:.2f}."
        for lag, row in refresh_by_lag.iterrows()
    ]
    rank_lines = [
        f"- `{strategy}`: {fraction:.0%} of sample-anchor rows changed rank by at least one "
        "between H=1 and H=64."
        for strategy, fraction in rank_change.items()
    ]
    evidence_md = _markdown_table(evidence)
    refresh_task_md = _markdown_table(refresh_by_task, include_index=True)

    report = f"""# Final analysis report

## 1. Experiment inventory

The canonical run is `{inventory['run_id']}` using
`mlx-community/Qwen2.5-1.5B-Instruct-4bit` on MLX/MPS. This is the
user-approved 4-bit run, not the originally discussed bfloat16 condition.
There are 15 samples: five cached official LongBench `gov_report`, five
repository-synthetic RULER NIAH, and five deterministic long reasoning
prompts. The run uses seed 42 and config hash
`{inventory['config_hash']}`.

Anchors are 0, 16, and 48; replay horizons are 1, 4, 16, and 64. The cache has
4 sink + 220 selected core + 32 rolling recent = 256 tokens per layer.
Selectors are SnapKV, V-space ridge leverage, attention-weighted V-space ridge
leverage, and a separately constructed future-attention oracle for each
horizon. Selection runs on all 28 layers and both KV heads; detailed diagnostics
cover layers 0/14/27 and query heads 0/11 (six query-head records).

Every horizon is a separate teacher-forced replay on the full-cache greedy
reference tokens. The core and sink are frozen within a replay and the recent
window rolls. The main per-step analysis uses only H=64 rows to avoid counting
overlapping deployable replays repeatedly. Future-oracle results are explicitly
H=64-conditioned in that table.

## 2. Data quality

All five canonical Parquet tables and all 15 NPZ files are readable. There are
15,300 valid step rows, 720 valid horizon rows, 94,410 temporal-signal rows, no
logical-key duplicates, no invalid replay rows, and active-cache size is always
256. Two generations ended before 128 tokens (115 and 126) but still cover all
requested anchor/horizon combinations. No sample was excluded.

The run records a dirty worktree, so the commit hash alone does not reproduce
the source state. Seven audit issues are retained in
`data_quality_issues`; three are material missing-field/arm issues. Full
historical/anchor V matrices and future per-token V vectors are absent; full
future score vectors/set identities are absent; and no cache refreshed at every
future step was replayed.

## 3. Descriptive findings

At H=64, stale-cache loss is heterogeneous:

{chr(10).join(h64_lines)}

Means are often much larger than medians, particularly for sparse refresh
benefit, so a single global mean is not representative. Negative ΔNLL or
negative refresh benefit can occur for a token because the compressed
distribution can assign that reference token more probability than the
full-cache distribution; it does not imply globally better generation.

## 4. Cache stability

Ridge-based scores are highly persistent on the saved old-token universe. At
lag 64, V-ridge has median Spearman
{stability64.loc['v_ridge_leverage','spearman_rank_correlation']:.3f} and core
Jaccard {stability64.loc['v_ridge_leverage','top_core_jaccard']:.3f}; the hybrid
has {stability64.loc['attention_weighted_v_ridge_leverage','spearman_rank_correlation']:.3f}
and {stability64.loc['attention_weighted_v_ridge_leverage','top_core_jaccard']:.3f}.
SnapKV changes more: {stability64.loc['snapkv','spearman_rank_correlation']:.3f}
and {stability64.loc['snapkv','top_core_jaccard']:.3f}. These are selected-core
statistics, excluding the fixed sink and rolling recent window.

Score drift is associated with set turnover (Spearman {_ci(score_to_set)}), but
set turnover is much less strongly associated with ΔNLL ({_ci(set_to_loss)}).
There are {redundancy_cases} high-turnover/low-|ΔNLL| and
{hidden_error_cases} low-turnover/high-|ΔNLL| exploratory layer records under
global quartile definitions. This directly cautions against treating set
overlap as functional validity.

Kendall correlation, per-token rank displacement, entering/leaving identities,
new-token score ranks, exact recent-window Jaccard, and whole-cache Jaccard are
unavailable. Saved score drift covers old anchor tokens after excluding
sink/recent eligibility.

## 5. Dataset and geometry dynamics

The available “new-direction residual” is relative to the full-rank span of the
220-token selected core in a 128-dimensional KV-head space. It is not a
residual to all history. Its overall median changes from
{residual_by_lag.loc[1]:.3g} at lag 1 to
{residual_by_lag.loc[64]:.3g} at lag 64. Selected-core effective-rank medians at
anchors 0/16/48 are {effective_rank.loc[0]:.1f},
{effective_rank.loc[16]:.1f}, and {effective_rank.loc[48]:.1f}; these are
cross-anchor summaries of newly selected cores, not a time-resolved spectrum
for one fixed dataset.

Selected-core residual is weakly associated with score instability
({_ci(residual_to_score)}) and essentially unassociated with set turnover
({_ci(residual_to_set)}). Its relationship to sparse refresh benefit is
{_ci(residual_to_refresh)}. The interval crossing zero means this run does not
show a stable direct relationship for this narrowly defined residual. It does
**not** test the requested full-history residual, online leverage, covariance
drift, principal angles, or local-window regime shift.

Strict Gaussianity, approximate ellipticity, second-order stationarity, and
temporal distribution stability cannot be separated because means,
covariances, whitened coordinates, Mahalanobis norms, skewness, and kurtosis
were not saved. No normality claim is made.

## 6. Refresh benefit

Only three same-token comparisons are identifiable: stale anchor 0 step 17
versus refreshed anchor 16 step 1; anchor 0 step 49 versus anchor 48 step 1; and
anchor 16 step 33 versus anchor 48 step 1. Token ID and reference position match
for all 180 global comparisons. Oracle comparisons use the H=64-conditioned
oracle on both sides and remain diagnostic rather than deployable.

Across all comparisons, the median benefit is {refresh_all[0]:.3f}
(95% sample-cluster bootstrap CI {refresh_all[1]:.3f}–{refresh_all[2]:.3f};
15 samples). By saved refresh lag:

{chr(10).join(refresh_lines)}

Task-level mean/median benefits are:

{refresh_task_md}

Refresh benefit is therefore heterogeneous and right-skewed. Averages do not
justify saying refresh is universally useless; conversely, three saved
boundaries do not establish a refresh schedule or causal lifetime.

Layer/head-specific NLL benefit is unavailable because logits are global. The
six saved diagnostic query heads support only attention-output-error benefit.
Compression-regime dependence is unavailable because this run has one budget
(256) and one quantization condition (4-bit).

## 7. Mechanism-chain analysis

The proposed chain is only partially observable:

1. **Data drift → score instability.** Selected-core residual → score drift is
   {_ci(residual_to_score)}. Full-history residual, online leverage, and
   covariance/subspace drift are unavailable.
2. **Score instability → set change.** This is the clearest saved link:
   score drift → turnover is {_ci(score_to_set)}. Smaller selection margin has
   only a weak relationship with turnover, and margin → H(ΔNLL≤0.1) is
   {_ci(margin_to_life)}.
3. **Set change → functional error.** Turnover → ΔNLL is {_ci(set_to_loss)},
   with both redundancy-like and low-turnover/high-error counterexamples.
4. **Functional error → refresh benefit.** Stale ΔNLL is the strongest observed
   correlate of sparse benefit: {_ci(stale_to_refresh)}.
5. **Data drift → refresh benefit.** Selected-core residual gives
   {_ci(residual_to_refresh)}; attention drift gives
   {_ci(attention_to_refresh)}; turnover gives {_ci(set_to_refresh)}. Their
   uncertainty intervals include zero.

These are exploratory associations with sample-cluster bootstrap intervals,
not independent token-level tests and not causal estimates. The limited
standardized OLS/leave-one-sample-out artifact is supplied for sensitivity,
but the report does not promote its coefficients to mechanism evidence.

## 8. Competing explanations

{evidence_md}

## 9. Horizon dependence

The future oracle is horizon-specific. Median all-layer oracle-core Jaccard is
{oracle_pair.loc[(1,4)]:.3f} for H1/H4,
{oracle_pair.loc[(1,16)]:.3f} for H1/H16, and
{oracle_pair.loc[(1,64)]:.3f} for H1/H64. This indicates progressively different
short- versus long-horizon content on average, but non-monotonic sample/layer
behavior remains.

Overall mean-loss ranks move from SnapKV/oracle/hybrid/V-ridge at H=1 to
oracle/hybrid/SnapKV/V-ridge at H=64. Per-sample-anchor rank-change fractions:

{chr(10).join(rank_lines)}

Empirical validity is threshold-sensitive. For example, median H under mean
ΔNLL≤0.01 ranges from 14 to 35 across selectors; at ≤0.25 all selector medians
are right-censored at 64. Sample-percentile definitions produce still different
rankings. “Lifetime” is therefore an operational definition in this run, not a
unique latent quantity.

## 10. What the current experiment cannot establish

- Generality beyond 15 samples, three task families, one 1.5B 4-bit model, one
  cache budget, and one seed.
- Free-running behavior: teacher forcing prevents trajectory divergence and
  can understate compounding failure.
- Causal effects of refresh or drift. Sparse anchor comparisons are matched on
  reference token but were not randomized interventions over every step.
- Full value-space dynamics, online leverage, covariance drift, principal
  angles, time-resolved rank, stationarity, elliptical/Gaussian diagnostics, or
  full-history new directions.
- Exact token-level entry/exit, recent-window absorption at the 32-token exit,
  Kendall/rank displacement, or whole-cache set stability.
- All-layer/head functional attribution. Only layers 0/14/27 and query heads
  0/11 have attention-output diagnostics, and these are pre-output-projection.
- A deployable oracle conclusion: the oracle is future-aware, horizon-specific,
  and restricted to anchor-existing tokens.
- Formal multiple-comparison claims. The reported correlations are exploratory;
  no p-value selection is used.

## 11. Recommended next experiments

| Priority | Minimal experiment | Distinguishes | Minimum scale | New inference? | Required fields |
|---|---|---|---|---|---|
| P0 | Add dense refresh at lags 1, 8, 16, 24, 32, 40, 48, 64 for the existing 15 samples and three deployable selectors | score persistence/redundancy vs true refresh benefit | existing 15 samples | yes, compressed replay only | stale/refreshed logits, cache IDs, token positions, per-head output errors |
| P0 | Save per-step V sketches plus anchor Cholesky/SVD factors on layers 0/14/27 and both KV heads | new directions vs attention/downstream explanations | 6 samples (2/task) for discovery, then 15 confirmation | yes | future V vectors or reproducible sketch, Gram factor, local-window covariance, projection residual |
| P0 | Exit-aligned intervention: refresh just before/after a candidate token leaves the 32-token recent window | recent-window absorption vs fixed-core sufficiency | 9 samples (3/task) | yes | token/core/recent identities, event token, exit step, matched losses |
| P1 | Repeat only budgets 192/256/384 on 9 samples | compression-regime dependence vs model/task effects | 9 samples | yes | same fields plus budget |
| P1 | Free-run short 32-token continuations only around the largest saved sparse-benefit cases | teacher-forcing artifact vs compounding functional failure | top 6 events plus 6 matched controls | yes | generated tokens, sequence likelihood/task outcome, divergence point |

These are targeted discriminating experiments, not a request for a broad
benchmark sweep.

{"The three P0 mechanism experiments above were subsequently completed on the same 15 samples; Section 13 reports the results." if mechanism_addendum_path.exists() else ""}

## 12. Candidate theoretical questions supported by data

### When do stale ridge scores remain stable despite decoding?

Current support: V-ridge and hybrid old-token ranks/Jaccards remain high through
lag 64. Counterexample: SnapKV changes faster, and stable score/set does not
guarantee low loss. Missing: new-token ranks and full value-space factors.
Worth further theory: **yes**, if formulated conditionally on candidate
universe, margin, and rolling recent coverage.

### Can online leverage or full-history residual predict refresh benefit?

Current support: only the selected-core residual is available, and its sparse
benefit association is near zero. Counterexample: large benefit can occur
without a large saved residual. Missing: the actual online leverage/full-history
residual and dense benefit. Worth further theory: **undetermined but high-value
to test**, because the present run does not measure the proposed variable.

### Does a rolling recent window delay fixed-core failure?

Current support: the implementation keeps every new token recent for 32 steps,
and benefit often grows at later saved lags. Counterexample: some lag-16
benefits are already material. Missing: token identity and exit-aligned refresh.
Worth further theory: **yes after the P0 exit intervention**, not before.

### Is cache validity better characterized by drift than elapsed steps?

Current support: elapsed lag alone does not explain sample heterogeneity, while
stale loss strongly tracks sparse benefit. Counterexample: saved residual,
attention drift, and turnover are not stable refresh predictors here. Missing:
covariance/subspace drift and dense refresh. Worth further theory:
**premature**, but the targeted measurement is well motivated.

## Artifact guide

There are {len(figures)} standalone figures, each with CSV and Parquet source
data. Six figures are deliberately marked unavailable. Definitions and
reproduction commands are in `analysis/README.md`; audit details are in
`data_schema_report.md` and `implementation_findings.md`.
"""
    if mechanism_addendum_path.exists():
        report = (
            report.rstrip()
            + "\n\n"
            + mechanism_addendum_path.read_text(encoding="utf-8").strip()
            + "\n"
        )
    (analysis_dir / "final_analysis_report.md").write_text(report, encoding="utf-8")

    david = f"""# Update for David

We completed an offline analysis of the small decoding-stage cache-reuse experiment. The run used the 4-bit MLX Qwen2.5-1.5B-Instruct model, 15 samples across cached LongBench gov_report, synthetic RULER NIAH, and deterministic long reasoning, with anchors at 0, 16, and 48 and reuse horizons 1, 4, 16, and 64. Each replay was teacher-forced on the same full-cache reference trajectory. The cache budget was 256 tokens: 4 sink, 220 fixed selected-core, and 32 rolling recent. We compared SnapKV, V-ridge leverage, attention-weighted V-ridge, and a separate future-attention oracle for each horizon.

Four observations seem most useful. First, ridge-based scores are remarkably persistent on the old-token candidate universe. At lag 64, median Spearman rank correlation/core Jaccard were {stability64.loc['v_ridge_leverage','spearman_rank_correlation']:.3f}/{stability64.loc['v_ridge_leverage','top_core_jaccard']:.3f} for V-ridge and {stability64.loc['attention_weighted_v_ridge_leverage','spearman_rank_correlation']:.3f}/{stability64.loc['attention_weighted_v_ridge_leverage','top_core_jaccard']:.3f} for the attention-weighted version. SnapKV changed more ({stability64.loc['snapkv','spearman_rank_correlation']:.3f}/{stability64.loc['snapkv','top_core_jaccard']:.3f}). Score drift was associated with selected-core turnover (sample-cluster bootstrap Spearman {_ci(score_to_set)}), so score persistence is a credible part of the explanation.

Second, set stability is not the same as functional stability. Turnover versus ΔNLL had only Spearman {_ci(set_to_loss)}, and we found both high-turnover/low-loss and low-turnover/high-loss records. Attention-output error versus ΔNLL was also modest ({_ci(attention_to_loss)}). This leaves room for functional redundancy and downstream attenuation, but neither is established.

Third, refresh benefit is heterogeneous and right-skewed. We can reconstruct only three same-token cross-anchor comparisons (lags 16, 32, and 48), not a dense refresh curve. Across these comparisons the median stale-minus-refreshed ΔNLL was {refresh_all[0]:.3f}, while stale loss was the strongest observed correlate of benefit (Spearman {_ci(stale_to_refresh)}). Therefore, a small average alone would be misleading: a minority of sample/transition cases drives meaningful benefits.

Fourth, horizon changes preferred content. Median future-oracle Jaccard falls from {oracle_pair.loc[(1,4)]:.3f} between H=1 and H=4 to {oracle_pair.loc[(1,64)]:.3f} between H=1 and H=64. Overall loss ranks also change: SnapKV is lowest at H=1, while the horizon-conditioned oracle is lowest at H=64 and the attention-weighted V-ridge moves ahead of SnapKV.

The part most directly related to your stable-distribution/new-direction suggestion remains unresolved. The saved “new-direction residual” is relative to the full-rank span of the 220-token selected core, not to all historical values. Its correlation with sparse refresh benefit was {_ci(residual_to_refresh)}. We cannot interpret that as evidence against the hypothesis because the run did not save full historical/future V vectors, online leverage, local covariance, or principal-angle drift. It also cannot test strict Gaussianity or second-order stationarity.

The smallest decisive follow-up is not a broad benchmark. On the same 15 samples, save dense refreshed replays at a few lags and per-step V sketches or stable Gram/SVD factors for layers 0, 14, and 27. Add a matched intervention immediately before and after a token leaves the 32-token recent window. This would separate three explanations: genuinely stable geometry, functional redundancy, and recent-window absorption.

Questions for you:

1. Should the theoretical object be stability over the old-token candidate universe, or must it explicitly include newly generated tokens entering eligibility?
2. Is the most useful target a bound on stale-cache loss, or a decision rule predicting positive refresh benefit?
3. Would you prioritize local second-order stationarity or online leverage spikes as the first geometric measurement?
"""
    if david_addendum_path.exists():
        david = (
            david.rstrip()
            + "\n\n"
            + david_addendum_path.read_text(encoding="utf-8").strip()
            + "\n"
        )
    words = re.findall(r"\b[\w–-]+\b", david)
    if not 500 <= len(words) <= 800:
        raise ValueError(f"david_update must be 500-800 words; got {len(words)}")
    (analysis_dir / "david_update.md").write_text(david, encoding="utf-8")

    # Machine-readable key findings used in both narrative reports.
    findings = pd.DataFrame(
        [
            {
                "finding": "score_drift_to_turnover",
                "effect": score_to_set["spearman"],
                "ci_low": score_to_set["spearman_ci_low"],
                "ci_high": score_to_set["spearman_ci_high"],
                "status": "observed_exploratory",
            },
            {
                "finding": "turnover_to_delta_nll",
                "effect": set_to_loss["spearman"],
                "ci_low": set_to_loss["spearman_ci_low"],
                "ci_high": set_to_loss["spearman_ci_high"],
                "status": "observed_exploratory",
            },
            {
                "finding": "selected_core_residual_to_sparse_refresh",
                "effect": residual_to_refresh["spearman"],
                "ci_low": residual_to_refresh["spearman_ci_low"],
                "ci_high": residual_to_refresh["spearman_ci_high"],
                "status": "narrow_residual_not_full_history",
            },
            {
                "finding": "stale_loss_to_sparse_refresh",
                "effect": stale_to_refresh["spearman"],
                "ci_low": stale_to_refresh["spearman_ci_low"],
                "ci_high": stale_to_refresh["spearman_ci_high"],
                "status": "observed_sparse_counterfactual",
            },
        ]
    )
    write_dual(findings, table / "key_findings")

    manifest_entries = []
    for path in sorted(analysis_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        manifest_entries.append(
            {
                "path": str(path.relative_to(analysis_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "analysis_seed": 20260724,
        "input_dir": str(Path(input_dir).resolve()),
        "run_id": inventory["run_id"],
        "config_hash": inventory["config_hash"],
        "n_files": len(manifest_entries),
        "n_figures": int(len(figures)),
        "n_unavailable_figures": int(figures["availability"].eq("unavailable").sum()),
        "david_update_word_count": len(words),
        "files": manifest_entries,
    }
    json_dump(analysis_dir / "manifest.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    ensure_directory(args.analysis_dir)
    build(args.analysis_dir, args.input_dir)


if __name__ == "__main__":
    main()
