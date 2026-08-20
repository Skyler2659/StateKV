# Dynamic-Horizon Oracle analysis

All four levels use the same Qwen3-8B qk_pool trajectories, six preregistered layers, eight KV heads, future-attention labels, and held-out sequences. Global/task/per-head choices are fitted only on the three development sequences. `NON_CAUSAL_TOKEN_TIME_ORACLE` uses future ranks and is an upper bound, not a deployable method. The gate uses paired sequence bootstrap on the mean over H={1,4,16,32}.
