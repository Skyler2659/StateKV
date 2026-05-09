import argparse
import importlib.util
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_module_from_file(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_kv_seq_dims(model_type: str) -> Tuple[int, int]:
    model_type = (model_type or "").lower()
    if "llama" in model_type or "gpt_neox" in model_type:
        return 2, 2
    if "mpt" in model_type:
        return 3, 2
    if "falcon" in model_type:
        return 1, 1
    # Default for GPT2-like cache layout: [B, H, S, D]
    return 2, 2


def get_seq_len_from_legacy_cache(past_key_values, k_seq_dim: int) -> int:
    return int(past_key_values[0][0].size(k_seq_dim))


def slice_tensor_by_dim(x: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    idx = [slice(None)] * x.dim()
    idx[dim] = slice(start, end)
    return x[tuple(idx)]


class RecentOnlyKVCache:
    """Sliding window baseline: keep only recent tokens, drop all prefix sinks."""

    def __init__(self, recent_size: int, k_seq_dim: int, v_seq_dim: int):
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

    def evict_for_space(self, past_key_values, num_coming: int):
        if past_key_values is None:
            return None
        seq_len = get_seq_len_from_legacy_cache(past_key_values, self.k_seq_dim)
        if seq_len + num_coming <= self.recent_size:
            return past_key_values
        keep_from = seq_len - self.recent_size + num_coming
        keep_from = max(0, keep_from)
        kept = []
        for k, v in past_key_values:
            new_k = slice_tensor_by_dim(k, self.k_seq_dim, keep_from, seq_len)
            new_v = slice_tensor_by_dim(v, self.v_seq_dim, keep_from, seq_len)
            kept.append((new_k, new_v))
        return tuple(kept)


def build_eval_text(repeat: int) -> str:
    base = (
        "Streaming language models process long conversations token by token. "
        "Attention sinks preserve stability while a bounded cache controls compute. "
        "This text is repeated to create a long stream for benchmarking. "
    )
    return base * repeat


@torch.no_grad()
def run_decode_eval(
    model,
    input_ids: torch.Tensor,
    kv_cache,
    k_seq_dim: int,
    label: str,
    max_steps: Optional[int] = None,
):
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    nlls = []
    step_times = []
    kv_lens = []

    total_steps = input_ids.size(1) - 1
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    for idx in range(total_steps):
        token = input_ids[:, idx : idx + 1]
        target = input_ids[:, idx + 1 : idx + 2].to(token.device).view(-1)

        if kv_cache is not None:
            past_key_values = kv_cache.evict_for_space(past_key_values, num_coming=1)

        t0 = time.perf_counter()
        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        logits = outputs.logits[:, -1, :].view(-1, model.config.vocab_size)
        nll = loss_fn(logits, target)
        nlls.append(nll)
        past_key_values = outputs.past_key_values

        seq_len = get_seq_len_from_legacy_cache(past_key_values, k_seq_dim)
        kv_lens.append(seq_len)

    mean_nll = torch.stack(nlls).mean().item()
    ppl = math.exp(mean_nll)
    total_time = sum(step_times)
    tok_per_s = total_steps / total_time if total_time > 0 else float("inf")

    return {
        "label": label,
        "steps": total_steps,
        "ppl": ppl,
        "total_s": total_time,
        "tok_per_s": tok_per_s,
        "avg_ms_per_token": (total_time / total_steps) * 1000.0,
        "max_kv_len": max(kv_lens) if kv_lens else 0,
        "final_kv_len": kv_lens[-1] if kv_lens else 0,
    }


def print_table(results):
    headers = [
        "mode",
        "steps",
        "ppl",
        "tok/s",
        "avg ms/tok",
        "max kv len",
        "final kv len",
    ]
    print("\n=== End-to-end decode benchmark ===")
    print(
        f"{headers[0]:<16} {headers[1]:>7} {headers[2]:>10} {headers[3]:>10} "
        f"{headers[4]:>12} {headers[5]:>12} {headers[6]:>13}"
    )
    for r in results:
        print(
            f"{r['label']:<16} {r['steps']:>7d} {r['ppl']:>10.4f} {r['tok_per_s']:>10.2f} "
            f"{r['avg_ms_per_token']:>12.3f} {r['max_kv_len']:>12d} {r['final_kv_len']:>13d}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text_repeat", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=256)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    main_kv_file = root / "streaming-llm-main" / "streaming_llm" / "kv_cache.py"
    main_kv_mod = load_module_from_file("main_kv_mod", main_kv_file)
    StartRecentKVCache = main_kv_mod.StartRecentKVCache

    print(f"Loading model: {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()

    # This benchmark requires legacy tuple cache layout used by repo code.
    # Newer transformers may return DynamicCache, which is not directly compatible.
    probe_ids = tokenizer("cache probe", return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad():
        probe_out = model(input_ids=probe_ids, use_cache=True)
    if not isinstance(probe_out.past_key_values, tuple):
        print("\nERROR: Your transformers runtime returns DynamicCache, but this repo's")
        print("StartRecentKVCache expects legacy tuple past_key_values.")
        print("To run this benchmark as intended, create a clean env and install:")
        print("  pip install transformers==4.33.0 torch datasets")
        print("Then rerun this script.")
        return

    k_seq_dim, v_seq_dim = infer_kv_seq_dims(model.config.model_type)
    streaming_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    window_cache = RecentOnlyKVCache(
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )

    text = build_eval_text(args.text_repeat)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)

    print(
        f"Tokenized length: {input_ids.size(1)} | max_steps: {args.max_steps} | "
        f"start_size: {args.start_size} | recent_size: {args.recent_size}"
    )

    # 1) plain no-op (unbounded cache growth)
    res_plain = run_decode_eval(
        model=model,
        input_ids=input_ids,
        kv_cache=None,
        k_seq_dim=k_seq_dim,
        label="plain_unbounded",
        max_steps=args.max_steps,
    )

    # 2) window-only baseline (recent only)
    res_window = run_decode_eval(
        model=model,
        input_ids=input_ids,
        kv_cache=window_cache,
        k_seq_dim=k_seq_dim,
        label="window_only",
        max_steps=args.max_steps,
    )

    # 3) streaming (start sinks + recent)
    res_stream = run_decode_eval(
        model=model,
        input_ids=input_ids,
        kv_cache=streaming_cache,
        k_seq_dim=k_seq_dim,
        label="streaming_sink",
        max_steps=args.max_steps,
    )

    results = [res_plain, res_window, res_stream]
    print_table(results)

    print("\n=== Key interpretation ===")
    print("- Speed: compare tok/s and avg ms/tok.")
    print("- Stability: compare ppl(window_only) vs ppl(streaming_sink).")
    print("- Cache budget: streaming/window should cap max kv len near start+recent/recent.")
    print("- plain_unbounded is expected to keep growing KV and slow down as steps increase.")


if __name__ == "__main__":
    main()
