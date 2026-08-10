# StateKV refresh arms — never vs every vs fixed_k16 (samples 101-105, 768/256)

Run: `/Users/wangsikai/l1-robust-kv-cache/results/temporal_cache_discovery/statekv_refresh_arms_qwen3_8b_768_256_v1`. Strict pure eviction, matched samples, same budget.

    policy       arm  samples  mean_kl_all  mean_kl_niah  mean_kl_gov  mean_niah_retrieval  mean_official_score
 attention     every       10     0.090231      0.023960     0.156502                  1.0             52.92029
 attention fixed_k16       10     0.272378      0.334738     0.210019                  1.0             53.28191
 attention     never       10     0.288872      0.345632     0.232113                  1.0             53.05335
b2_uniform     every       10     0.181826      0.024177     0.339474                  1.0             53.10136
b2_uniform fixed_k16       10     0.377217      0.351997     0.402437                  1.0             53.25370
b2_uniform     never       10     0.358755      0.335499     0.382012                  1.0             53.15260

Paired never-minus-every trajectory KL (negative = never better):
    policy          sample_id  is_niah  never_minus_every_kl  never_better
 attention     gov_report:101    False              0.225549         False
 attention     gov_report:102    False             -0.034747          True
 attention     gov_report:103    False              0.068752         False
 attention     gov_report:104    False             -0.008935          True
 attention     gov_report:105    False              0.127435         False
 attention synthetic_niah_101     True              0.291789         False
 attention synthetic_niah_102     True              0.363603         False
 attention synthetic_niah_103     True              0.286446         False
 attention synthetic_niah_104     True              0.294951         False
 attention synthetic_niah_105     True              0.371573         False
b2_uniform     gov_report:101    False              0.047638         False
b2_uniform     gov_report:102    False              0.013231         False
b2_uniform     gov_report:103    False              0.002418         False
b2_uniform     gov_report:104    False              0.057823         False
b2_uniform     gov_report:105    False              0.091579         False
b2_uniform synthetic_niah_101     True              0.238213         False
b2_uniform synthetic_niah_102     True              0.343090         False
b2_uniform synthetic_niah_103     True              0.312024         False
b2_uniform synthetic_niah_104     True              0.305600         False
b2_uniform synthetic_niah_105     True              0.357680         False

**Verdicts (predeclared): attention: NO_CLEAR_REFRESH_ADVANTAGE; b2_uniform: NO_CLEAR_REFRESH_ADVANTAGE**
