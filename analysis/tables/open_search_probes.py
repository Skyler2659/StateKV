"""Open-search offline probes on the QK-V decomposition records.

Probes (all offline; no model runs):
  HF1 coverage stress : attention mass captured by top-K vs K, per layer/cycle
  HF1 regret conc.    : where qk_pool's per-cycle exact KL sits (step_rows)
  HF3 set stability   : core-set Jaccard across cycles; stale-set mass decay
  HF2 page recall     : page-max candidate generation recall of exact top-core

Input: results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1/
Output: analysis/tables/open_*.csv + stdout summary
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = "results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1"
OUT = "analysis/tables"

CORE_KS = [28, 92, 156, 220, 284]  # budgets 64/128/192/256/320 minus sink4+recent32
PAGE_SIZES = [4, 8, 16, 32]
STALE_HORIZONS = [1, 2, 4, 8]


def load_token_rows() -> pd.DataFrame:
    t = pq.read_table(
        f"{DATA}/token_rows.parquet",
        columns=["sample_id", "task", "cycle", "layer", "position", "attn", "in_core"],
    )
    return t.to_pandas()


def probe_coverage_and_stability(df: pd.DataFrame) -> None:
    # Per (sample, cycle, layer): rank positions by attn desc, cumulative mass.
    cov_rows = []
    stab_rows = []
    stale_rows = []
    keys = ["sample_id", "cycle", "layer"]
    df = df.sort_values(keys + ["attn"], ascending=[True, True, True, False])
    grouped = {k: g for k, g in df.groupby(keys, sort=True)}

    # core sets per (sample, cycle, layer) for stability
    core_sets = {}
    for (sid, cyc, lay), g in grouped.items():
        attn = g["attn"].to_numpy()
        total = attn.sum()
        cumsum = np.cumsum(attn)
        row = {"sample_id": sid, "cycle": cyc, "layer": lay, "pool": len(g)}
        for k in CORE_KS:
            kk = min(k, len(g))
            row[f"mass_k{k}"] = cumsum[kk - 1] / total if total > 0 else np.nan
        cov_rows.append(row)
        core = frozenset(g.loc[g["in_core"], "position"].tolist())
        core_sets[(sid, cyc, lay)] = core

    cov = pd.DataFrame(cov_rows)
    cov.to_csv(f"{OUT}/open_hf1_coverage_by_cycle.csv", index=False)
    agg = cov.groupby("layer")[[f"mass_k{k}" for k in CORE_KS]].mean()
    agg.to_csv(f"{OUT}/open_hf1_coverage_by_layer.csv")
    print("=== HF1 coverage stress (mean attention mass captured, by layer) ===")
    print(agg.round(4).to_string())
    overall = cov[[f"mass_k{k}" for k in CORE_KS]].mean()
    print("\noverall mean captured mass:")
    print(overall.round(5).to_string())
    print("\noverall p05 captured mass (hard cycles):")
    print(cov[[f"mass_k{k}" for k in CORE_KS]].quantile(0.05).round(5).to_string())

    # stability: Jaccard of consecutive core sets + stale mass decay
    samples = sorted(df["sample_id"].unique())
    cycles = sorted(df["cycle"].unique())
    layers = sorted(df["layer"].unique())
    attn_lookup = {(sid, cyc, lay): g for (sid, cyc, lay), g in grouped.items()}
    for sid in samples:
        for lay in layers:
            for i, cyc in enumerate(cycles[:-1]):
                a = core_sets.get((sid, cyc, lay))
                b = core_sets.get((sid, cycles[i + 1], lay))
                if a is None or b is None:
                    continue
                jac = len(a & b) / max(1, len(a | b))
                stab_rows.append({"sample_id": sid, "layer": lay, "cycle": cyc, "jaccard": jac})
            for i, cyc in enumerate(cycles):
                a = core_sets.get((sid, cyc, lay))
                if not a:
                    continue
                for h in STALE_HORIZONS:
                    fut = cyc + h
                    if fut > cycles[-1]:
                        continue
                    g = attn_lookup.get((sid, fut, lay))
                    if g is None:
                        continue
                    total = g["attn"].sum()
                    if total <= 0:
                        continue
                    mass = g.loc[g["position"].isin(a), "attn"].sum() / total
                    stale_rows.append(
                        {"sample_id": sid, "layer": lay, "cycle": cyc, "h": h, "stale_mass": mass}
                    )
    stab = pd.DataFrame(stab_rows)
    stab.groupby("layer")["jaccard"].mean().to_csv(f"{OUT}/open_hf3_jaccard_by_layer.csv")
    print("\n=== HF3 core-set Jaccard(c, c+1): overall mean "
          f"{stab['jaccard'].mean():.4f}, p05 {stab['jaccard'].quantile(0.05):.4f} ===")
    stale = pd.DataFrame(stale_rows)
    stale_agg = stale.groupby("h")["stale_mass"].agg(["mean", lambda s: s.quantile(0.05)])
    stale_agg.columns = ["mean", "p05"]
    stale_agg.to_csv(f"{OUT}/open_hf3_stale_mass_decay.csv")
    print("\n=== HF3 stale-set attention mass decay ===")
    print(stale_agg.round(5).to_string())


def probe_page_recall(df: pd.DataFrame) -> None:
    # exact top-220 core (by attn) vs page-max selection at various page sizes.
    rows = []
    for (sid, cyc, lay), g in df.groupby(["sample_id", "cycle", "layer"], sort=False):
        g = g.sort_values("position")
        attn = g["attn"].to_numpy()
        pos = g["position"].to_numpy()
        k = min(220, len(g))
        exact = set(pos[np.argsort(-attn)[:k]].tolist())
        row = {"sample_id": sid, "cycle": cyc, "layer": lay}
        for p in PAGE_SIZES:
            # page id by absolute position // p; pages never cross pool edge.
            # positions are contiguous from 0, so pages are contiguous slices.
            page_id = pos // p
            edges = np.flatnonzero(np.r_[True, page_id[1:] != page_id[:-1]])
            pmax = np.maximum.reduceat(attn, edges)
            n_pages = int(np.ceil(k / p))
            order = np.argsort(-pmax)[:n_pages]
            ends = np.r_[edges[1:], len(g)]
            mask = np.zeros(len(g), dtype=bool)
            for s, e in zip(edges[order], ends[order]):
                mask[s:e] = True
            sel = set(pos[mask].tolist())
            row[f"recall_p{p}"] = len(sel & exact) / k
        rows.append(row)
    rec = pd.DataFrame(rows)
    rec.to_csv(f"{OUT}/open_hf2_page_recall_by_cycle.csv", index=False)
    agg = rec[[f"recall_p{p}" for p in PAGE_SIZES]].mean()
    agg.to_csv(f"{OUT}/open_hf2_page_recall.csv")
    print("\n=== HF2 page-max recall of exact top-220 core (mean) ===")
    print(agg.round(4).to_string())


def probe_regret_concentration() -> None:
    st = pq.read_table(f"{DATA}/step_rows.parquet").to_pandas()
    st = st.sort_values("exact_kl", ascending=False)
    st.to_csv(f"{OUT}/open_hf1_step_kl_sorted.csv", index=False)
    total = st["exact_kl"].sum()
    top10 = st.head(int(0.1 * len(st)))["exact_kl"].sum() / total
    print("\n=== HF1 regret concentration (qk_pool per-cycle exact KL) ===")
    print(f"cycles={len(st)}  mean={st['exact_kl'].mean():.5f}  "
          f"p50={st['exact_kl'].median():.5f}  p95={st['exact_kl'].quantile(0.95):.5f}  "
          f"max={st['exact_kl'].max():.5f}")
    print(f"top-10% cycles carry {top10:.3f} of total KL mass")
    early = st[st["cycle"] < 8]["exact_kl"].mean()
    late = st[st["cycle"] >= 56]["exact_kl"].mean()
    mid = st[(st["cycle"] >= 8) & (st["cycle"] < 56)]["exact_kl"].mean()
    print(f"KL by phase: cycles 0-7 {early:.5f} | 8-55 {mid:.5f} | 56-63 {late:.5f}")
    by_sample = st.groupby("sample_id")["exact_kl"].mean().sort_values(ascending=False)
    print("\nper-sample mean KL:")
    print(by_sample.round(5).to_string())


def main() -> None:
    df = load_token_rows()
    probe_regret_concentration()
    probe_coverage_and_stability(df)
    probe_page_recall(df)


if __name__ == "__main__":
    main()
