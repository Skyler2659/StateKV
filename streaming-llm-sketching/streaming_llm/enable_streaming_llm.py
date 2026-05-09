from streaming_llm.kv_cache import PlainKVCache, StartRecentKVCache, L1RobustKVCache


def enable_streaming_llm(model, start_size, recent_size):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        # Keep a safe default for GPT2-like models.
        k_seq_dim = v_seq_dim = 2

    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache


def enable_plain_llm(model):
    return PlainKVCache()


def enable_l1_robust_llm(
    model,
    cache_size=512,
    num_sink_tokens=4,
    sketch_dim=1024,
    recompute_interval=32,
    seed=0,
    per_layer=True,
    use_reweight=False,
    recent_keep=0,
):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        k_seq_dim = v_seq_dim = 2

    return L1RobustKVCache(
        cache_size=cache_size,
        num_sink_tokens=num_sink_tokens,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        sketch_dim=sketch_dim,
        recompute_interval=recompute_interval,
        seed=seed,
        per_layer=per_layer,
        use_reweight=use_reweight,
        recent_keep=recent_keep,
    )
