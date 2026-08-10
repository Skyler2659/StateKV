# StateKV 2B — propagation depth of candidate risk (Qwen3-8B, 768 ctx, budget 256/core 220)

Ladder run: `/Users/wangsikai/l1-robust-kv-cache/results/temporal_cache_discovery/statekv_ladder_qwen3_8b_2b_v1`. Panel candidates rolled out teacher-forced at horizons {1,2,4} on clones of the surviving attention-trajectory cache.

Per candidate x horizon means (step KL):
             candidate  horizon  rows  mean_step_kl  p95_step_kl  mean_cumulative_kl  p95_cumulative_kl
            b2_uniform        1   101      0.052852     0.426220            0.052852           0.426220
a2_temporal_volatility        1   101      0.052890     0.426196            0.052890           0.426196
             attention        1   101      0.052902     0.426075            0.052902           0.426075
                snapkv        1   101      0.053737     0.424248            0.053737           0.424248
               uniform        1   101      0.135999     0.631841            0.135999           0.631841
            b2_uniform        2   101      0.452284     0.733033            0.252568           0.440982
             attention        2   101      0.452328     0.732723            0.252615           0.443261
a2_temporal_volatility        2   101      0.452362     0.733048            0.252626           0.441250
                snapkv        2   101      0.640837     0.843676            0.347287           0.577683
               uniform        2   101      0.881905     2.571324            0.508952           2.281591
            b2_uniform        4   101      0.279469     0.255533            0.201988           0.327102
             attention        4   101      0.279594     0.255706            0.202044           0.326900
a2_temporal_volatility        4   101      0.279515     0.253170            0.202045           0.326677
                snapkv        4   101      0.280160     0.258533            0.249452           0.847444
               uniform        4   101      0.764290     1.511410            0.533431           5.521763

Cycles measured: 101; tied at horizon 1 (spread<0.001): 55.4%; separated at horizon 4: 38.6%; top-1 ranking agreement h1 vs h4: 62.4%.

Cliff signature: candidates whose step-KL explodes (>0.1) only at depth>=2 occur in 20.0% of cycles (mean 1.0 candidates/cycle).
(original predeclared rule outcome: DEEP_RISK; amended rule reports the cliff signature, which decides whether a deeper teacher can rank actions.)

Horizon-k oracle regret (teacher picks min at depth k):
horizon                      1       2       4
candidate                                     
a2_temporal_volatility  0.0006  0.0036  0.0006
attention               0.0006  0.0036  0.0007
b2_uniform              0.0005  0.0035  0.0006
snapkv                  0.0014  0.1921  0.0013
uniform                 0.0837  0.4331  0.4854

**2B verdict (predeclared): DEEP_RISK**
