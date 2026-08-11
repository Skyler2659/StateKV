# Refresh trigger fit summary

- source events: `results/temporal_cache_discovery/statekv_selective_refresh_labels_r2a_v3/partial_refresh_event_rows.parquet` (run `statekv_selective_refresh_labels_r2a_v3`)
- rows: 512 | samples: 4 | policies: attention, b2_uniform
- steps per sample-policy: 64 (median)
- **VERDICT: NO_FREEZE (UNDERPOWERED: plumbing run, freeze gate disabled)**

> UNDERPOWERED RUN: fewer than 10 samples — all numbers below are plumbing
> checks only, not evidence. Re-run on the full validation run before interpreting.

## 1. Label engineering

- primary label `tail_event` = refresh_benefit_lag4 >= 0.05 (rate 0.0195)
- secondary label `any_benefit` = refresh_benefit_lag4 > 0 (rate 0.4844)

| stat                   |        value |
|:-----------------------|-------------:|
| mean                   | -0.00279822  |
| median                 | -1.10885e-14 |
| p90                    |  0.00644108  |
| p99                    |  0.107688    |
| max                    |  0.364331    |
| min                    | -0.446291    |
| frac_gt_0              |  0.484375    |
| frac_ge_0.01           |  0.0859375   |
| frac_ge_0.05           |  0.0195312   |
| frac_ge_0.1            |  0.0136719   |
| frac_ge_0.25           |  0.00195312  |
| label_rate_any_benefit |  0.484375    |
| label_rate_tail_event  |  0.0195312   |
| lag16_mean             | -0.00279822  |
| lag16_median           | -1.10885e-14 |
| lag16_p90              |  0.00644108  |
| lag16_frac_gt_0        |  0.484375    |
| lag32_mean             | -0.00279822  |
| lag32_median           | -1.10885e-14 |
| lag32_p90              |  0.00644108  |
| lag32_frac_gt_0        |  0.484375    |

lag4/lag16 relation:
- lag_columns: refresh_benefit_lag4,refresh_benefit_lag16,refresh_benefit_lag32
- lag4_vs_lag16_max_abs_diff: 0.0
- lag4_lag16_identical: True
- NOTE: lag4 and lag16 benefit columns are identical here (12-step horizon vs 32-token recent window). Trigger fitting uses lag4 only; lag16 adds no information in this run.

## 2. Feature screening (policy=ALL, label=tail_event, top by |Spearman|)

| feature               |    spearman |   rank_auc |   precision_at_0.1 |   benefit_lift_at_0.1 |
|:----------------------|------------:|-----------:|-------------------:|----------------------:|
| score_tv_mean         |  0.040692   |   0.626113 |          0.02      |          -0.0160525   |
| stale_action_l1_lag4  | -0.0296438  |   0.523108 |          0         |          -0.0103297   |
| stale_action_l1_lag16 | -0.0296438  |   0.523108 |          0         |          -0.0103297   |
| churn_jaccard_mean    | -0.029473   |   0.515385 |          0         |          -0.0105966   |
| churn_jaccard_min     | -0.0280763  |   0.510729 |          0         |          -0.0105966   |
| churn_x_1minus_margin | -0.0269862  |   0.498785 |          0         |           0.00523006  |
| tv_x_churn            |  0.023397   |   0.650607 |          0.04      |          -0.0159913   |
| compressed_entropy    | -0.0110077  |   0.870319 |          0.0392157 |           0.000394918 |
| coverage_mass_mean    |  0.0109203  |   0.363745 |          0         |           0.00457539  |
| boundary_margin_mean  |  0.00498107 |   0.506175 |          0         |           0.00240077  |

Full per-policy tables in `refresh_trigger_feature_screen.csv`.

## 3. LOSO rule fitting (top 8 rules by held-out AUC)

| rule                                              |   auc_mean |   auc_min |   precision_mean |   recall_mean |   alert_rate_mean |   lift_positive_folds |   n_folds |   threshold_median |   threshold_min |   threshold_max |
|:--------------------------------------------------|-----------:|----------:|-----------------:|--------------:|------------------:|----------------------:|----------:|-------------------:|----------------:|----------------:|
| single:compressed_entropy                         |   0.878008 |  0.872951 |        0.100272  |      0.916667 |         0.162109  |                     2 |         4 |           1.01896  |        1.01145  |        1.03197  |
| rankprod:tv_x_churn*compressed_entropy            |   0.850041 |  0.738388 |        0.0330601 |      0.583333 |         0.388672  |                     1 |         4 |           0.194478 |        0.179532 |        0.558531 |
| rankprod:stale_action_l1_lag4*compressed_entropy  |   0.847237 |  0.807377 |        0.0857143 |      0.291667 |         0.0625    |                     3 |         4 |           0.53003  |        0.524821 |        0.54176  |
| rankprod:stale_action_l1_lag16*compressed_entropy |   0.847237 |  0.807377 |        0.0857143 |      0.291667 |         0.0625    |                     3 |         4 |           0.53003  |        0.524821 |        0.54176  |
| rankprod:churn_jaccard_mean*compressed_entropy    |   0.846554 |  0.806011 |        0.103896  |      0.375    |         0.0644531 |                     3 |         4 |           0.523707 |        0.519062 |        0.5353   |
| rankprod:churn_jaccard_min*compressed_entropy     |   0.838357 |  0.789617 |        0.103896  |      0.375    |         0.0644531 |                     3 |         4 |           0.523707 |        0.519062 |        0.5353   |
| rankprod:score_tv_mean*compressed_entropy         |   0.830607 |  0.703552 |        0.0320184 |      0.583333 |         0.357422  |                     1 |         4 |           0.226955 |        0.19429  |        0.569061 |
| and:tv_x_churn&compressed_entropy                 |   0.81483  |  0.663934 |        0.040625  |      0.583333 |         0.234375  |                     1 |         4 |           0.5607   |        0.286978 |        0.73592  |

Freeze gate: LOSO AUC >= 0.65 AND benefit lift positive in >= 4/4 folds.

## 3b. Fixed-interval baseline (mean benefit when alerting every k-th step)

| policy   |   k |   alert_frac |   mean_benefit_alerted |   mean_benefit_nonalerted |
|:---------|----:|-------------:|-----------------------:|--------------------------:|
| ALL      |   1 |     1        |            -0.00279822 |              nan          |
| ALL      |   2 |     0.5      |            -0.00263462 |               -0.00296181 |
| ALL      |   3 |     0.34375  |            -0.00161358 |               -0.00341874 |
| ALL      |   4 |     0.25     |            -0.00479219 |               -0.00213356 |
| ALL      |   6 |     0.171875 |            -0.00329401 |               -0.00269531 |
| ALL      |   8 |     0.125    |            -0.0103948  |               -0.00171298 |
| ALL      |  12 |     0.09375  |            -0.0030711  |               -0.00276999 |

Any frozen trigger must beat these alerted-mean-benefit numbers at comparable alert rates.
