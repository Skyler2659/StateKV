from streaming_llm.kv_cache import PlainKVCache


def enable_streaming_llm(model, start_size, recent_size):
    kv_cache = PlainKVCache()
    return kv_cache
