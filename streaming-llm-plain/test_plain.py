"""最小可行性测试：CPU + tiny model，验证 PlainKVCache 能跑通"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from streaming_llm.kv_cache import PlainKVCache

MODEL_NAME = "distilgpt2"
MAX_NEW_TOKENS = 20

print("Loading model on CPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu",
)
model.eval()

print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
print(f"Model type: {model.config.model_type}")

# Test 1: PlainKVCache 创建和调用
print("\n[Test 1] PlainKVCache instantiation...")
kv_cache = PlainKVCache()
assert kv_cache(None) is None, "Should return None for None input"
print("  OK")

# Test 2: 单 token 前向 + past_key_values
print("\n[Test 2] Single step forward + KV cache...")
text = "Hello, my name is"
input_ids = tokenizer(text, return_tensors="pt").input_ids

with torch.no_grad():
    # 第一次前向（prefill）
    out = model(input_ids, use_cache=True)
    pkv = out.past_key_values
    logits = out.logits[:, -1, :]
    next_token = logits.argmax(dim=-1)
    print(f"  Input: '{text}'")
    print(f"  Next token ID: {next_token.item()} -> '{tokenizer.decode(next_token.item())}'")
    print(f"  KV cache layers: {len(pkv)}")
    print("  OK")

# Test 3: PlainKVCache 透传（不淘汰）
print("\n[Test 3] PlainKVCache pass-through...")
pkv_after = kv_cache(pkv)
assert pkv_after is pkv, "Should return past_key_values unchanged"
print("  OK (returns past_key_values unchanged)")

# Test 4: 多步自回归生成
print("\n[Test 4] Multi-step autoregressive generation...")
generated = [next_token.item()]
current_pkv = pkv
for step in range(MAX_NEW_TOKENS):
    with torch.no_grad():
        out = model(next_token.unsqueeze(0), past_key_values=current_pkv, use_cache=True)
        current_pkv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
        # PlainKVCache 不做淘汰
        current_pkv = kv_cache(current_pkv)

output_text = tokenizer.decode(generated, skip_special_tokens=True)
print(f"  Generated: '{output_text}'")
print(f"  KV cache seq_len: {current_pkv[0][0].size(2)}")
print("  OK")

# Test 5: evict_for_space 和 evict_range 也是透传
print("\n[Test 5] evict_for_space / evict_range pass-through...")
assert kv_cache.evict_for_space(current_pkv, 10) is current_pkv
assert kv_cache.evict_range(current_pkv, 0, 5) is current_pkv
print("  OK")

print("\n=== All tests passed! PlainKVCache works on CPU. ===")
