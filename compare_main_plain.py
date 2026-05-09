import importlib.util
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(r"c:\Users\Lenovo\Desktop\streaming-llm-main")
MAIN_KV = ROOT / "streaming-llm-main" / "streaming_llm" / "kv_cache.py"
PLAIN_KV = ROOT / "streaming-llm-plain" / "streaming_llm" / "kv_cache.py"

def load_module_from_file(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_kv_seq_len(pkv):
    # Newer transformers may return DynamicCache.
    if hasattr(pkv, "get_seq_length"):
        return int(pkv.get_seq_length())
    # Legacy format: tuple(layer), each layer=(k, v), shape [bs, heads, seq, dim]
    return pkv[0][0].shape[2]

def build_fake_legacy_pkv(
    num_layers=2,
    batch=1,
    num_heads=2,
    seq_len=80,
    head_dim=8,
):
    # Mimic legacy HuggingFace cache: tuple[(k, v), ...]
    layer_list = []
    for _ in range(num_layers):
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        layer_list.append((k, v))
    return tuple(layer_list)

def run_model_sanity(model, tok, text):
    x = tok(text, return_tensors="pt").input_ids
    out1 = model(input_ids=x, use_cache=True)
    pkv_before = out1.past_key_values
    len_before = get_kv_seq_len(pkv_before)

    # 再走一步，验证可继续推理
    y = out1.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out2 = model(input_ids=y, past_key_values=pkv_before, use_cache=True)

    return {
        "len_before": len_before,
        "logits1_shape": tuple(out1.logits.shape),
        "logits2_shape": tuple(out2.logits.shape),
        "cache_type": type(pkv_before).__name__,
    }

def run_evict_compare(main_cache, plain_cache, num_coming):
    legacy_pkv = build_fake_legacy_pkv(seq_len=80)

    main_before = get_kv_seq_len(legacy_pkv)
    main_after_pkv = main_cache.evict_for_space(legacy_pkv, num_coming=num_coming)
    main_after = get_kv_seq_len(main_after_pkv)

    plain_before = get_kv_seq_len(legacy_pkv)
    plain_after_pkv = plain_cache.evict_for_space(legacy_pkv, num_coming=num_coming)
    plain_after = get_kv_seq_len(plain_after_pkv)

    return {
        "main_before": main_before,
        "main_after": main_after,
        "plain_before": plain_before,
        "plain_after": plain_after,
    }

def main():
    torch.manual_seed(0)

    main_mod = load_module_from_file("main_kv_mod", MAIN_KV)
    plain_mod = load_module_from_file("plain_kv_mod", PLAIN_KV)

    StartRecentKVCache = main_mod.StartRecentKVCache
    PlainKVCache = plain_mod.PlainKVCache

    # 用小模型确保 CPU 可跑
    model_name = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu").eval()

    text = "Hello " * 40  # 拉长一点，方便触发 main 的截断
    num_coming = 8

    main_cache = StartRecentKVCache(start_size=4, recent_size=16, k_seq_dim=2, v_seq_dim=2)
    plain_cache = PlainKVCache()

    sanity_res = run_model_sanity(model, tok, text)
    evict_res = run_evict_compare(main_cache, plain_cache, num_coming)

    print("=== Cache Type (repo impl) ===")
    print("main :", type(main_cache).__name__)
    print("plain:", type(plain_cache).__name__)
    print()

    print("=== Model sanity (real forward on your transformers version) ===")
    print("past_key_values type:", sanity_res["cache_type"])
    print("logits shape:", sanity_res["logits1_shape"], "->", sanity_res["logits2_shape"])
    print()

    print("=== Eviction compare (legacy cache format expected by repo code) ===")
    print(f"main : {evict_res['main_before']} -> {evict_res['main_after']}")
    print(f"plain: {evict_res['plain_before']} -> {evict_res['plain_after']}")
    print()

    if (
        evict_res["main_after"] < evict_res["main_before"]
        and evict_res["plain_after"] == evict_res["plain_before"]
    ):
        print("PASS: main has eviction/truncation, plain is no-op.")
    else:
        print("CHECK: unexpected behavior, please inspect parameters and model cache layout.")

if __name__ == "__main__":
    main()