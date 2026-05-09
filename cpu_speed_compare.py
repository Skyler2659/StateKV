import time
import torch
from pathlib import Path
import importlib.util

ROOT = Path(r"c:\Users\Lenovo\Desktop\streaming-llm-main")
MAIN_KV = ROOT / "streaming-llm-main" / "streaming_llm" / "kv_cache.py"
PLAIN_KV = ROOT / "streaming-llm-plain" / "streaming_llm" / "kv_cache.py"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fake_pkv(num_layers=24, batch=1, heads=16, seq_len=2048, head_dim=64, dtype=torch.float32):
    # legacy格式: tuple[(k, v), ...], each: [B, H, S, D]
    layers = []
    for _ in range(num_layers):
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        layers.append((k, v))
    return tuple(layers)

def benchmark(cache_obj, rounds=200, num_coming=32):
    pkv = fake_pkv()
    t0 = time.perf_counter()
    for _ in range(rounds):
        pkv = cache_obj.evict_for_space(pkv, num_coming=num_coming)
    t1 = time.perf_counter()
    return (t1 - t0) / rounds * 1000  # ms/round

def main():
    main_mod = load_module("main_kv", MAIN_KV)
    plain_mod = load_module("plain_kv", PLAIN_KV)

    main_cache = main_mod.StartRecentKVCache(start_size=4, recent_size=1024, k_seq_dim=2, v_seq_dim=2)
    plain_cache = plain_mod.PlainKVCache()

    main_ms = benchmark(main_cache)
    plain_ms = benchmark(plain_cache)

    print("=== CPU KV-evict micro benchmark ===")
    print(f"main  (StartRecentKVCache): {main_ms:.4f} ms/round")
    print(f"plain (PlainKVCache)      : {plain_ms:.4f} ms/round")
    speedup = plain_ms / main_ms if main_ms > 0 else float("inf")
    print(f"speedup (plain/main)      : {speedup:.2f}x")

if __name__ == "__main__":
    main()