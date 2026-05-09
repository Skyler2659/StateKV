import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_module_from_file(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def maybe_enable_pos_shift(model, sketching_root: Path, enabled: bool = True):
    if not enabled:
        print("pos_shift: disabled by flag")
        return
    model_type = (getattr(model.config, "model_type", "") or "").lower()
    try:
        if "llama" in model_type:
            mod = load_module_from_file(
                "sketch_pos_llama",
                sketching_root / "streaming_llm" / "pos_shift" / "modify_llama.py",
            )
            mod.enable_llama_pos_shift_attention(model)
            print("pos_shift: enabled for llama")
        elif "gpt_neox" in model_type:
            mod = load_module_from_file(
                "sketch_pos_gpt_neox",
                sketching_root / "streaming_llm" / "pos_shift" / "modify_gpt_neox.py",
            )
            mod.enable_gpt_neox_pos_shift_attention(model)
            print("pos_shift: enabled for gpt_neox")
        elif "falcon" in model_type:
            mod = load_module_from_file(
                "sketch_pos_falcon",
                sketching_root / "streaming_llm" / "pos_shift" / "modify_falcon.py",
            )
            mod.enable_falcon_pos_shift_attention(model)
            print("pos_shift: enabled for falcon")
        else:
            print(f"pos_shift: skipped (model_type={model_type})")
    except Exception as exc:
        print(f"pos_shift: failed to enable ({exc}); continuing without patch")


def infer_kv_seq_dims(model_type: str):
    model_type = (model_type or "").lower()
    if "llama" in model_type or "gpt_neox" in model_type:
        return 2, 2
    if "mpt" in model_type:
        return 3, 2
    if "falcon" in model_type:
        return 1, 1
    return 2, 2


def get_kv_seq_len(past_key_values, k_seq_dim: int):
    return int(past_key_values[0][0].size(k_seq_dim))


def _to_legacy_cache(past_key_values):
    """Convert any cache format to legacy tuple-of-tuples for eviction ops."""
    if past_key_values is None or isinstance(past_key_values, (list, tuple)):
        return past_key_values, None
    # 4.45-4.50: official to_legacy_cache() method
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache(), past_key_values
        except Exception:
            pass
    # 4.51+: .layers list of DynamicLayer objects with .keys/.values
    lyrs = getattr(past_key_values, "layers", None)
    if isinstance(lyrs, (list, tuple)) and len(lyrs) > 0:
        return tuple((lyr.keys, lyr.values) for lyr in lyrs), past_key_values
    # 4.45-4.50 fallback: .key_cache / .value_cache direct access
    for k_name, v_name in (("key_cache", "value_cache"),
                            ("_key_cache", "_value_cache")):
        kc = getattr(past_key_values, k_name, None)
        vc = getattr(past_key_values, v_name, None)
        if isinstance(kc, (list, tuple)) and isinstance(vc, (list, tuple)) and len(kc) == len(vc):
            return tuple((kc[i], vc[i]) for i in range(len(kc))), past_key_values
    return past_key_values, None


def _restore_cache_type(original_cache, legacy_cache):
    """Wrap evicted legacy tuple back into the original cache class."""
    if original_cache is None or isinstance(original_cache, (list, tuple)):
        return legacy_cache
    # 4.45-4.50: from_legacy_cache class method
    if hasattr(type(original_cache), "from_legacy_cache"):
        try:
            return type(original_cache).from_legacy_cache(legacy_cache)
        except Exception:
            pass
    # 4.51+: rebuild via .layers in-place, then call update() per layer
    if hasattr(original_cache, "layers"):
        for i, (k, v) in enumerate(legacy_cache):
            if i < len(original_cache.layers):
                original_cache.layers[i].keys = k
                original_cache.layers[i].values = v
            else:
                original_cache.update(k, v, i)
        return original_cache
    return legacy_cache


def slice_tensor_by_dim(x: torch.Tensor, dim: int, start: int, end: int):
    idx = [slice(None)] * x.dim()
    idx[dim] = slice(start, end)
    return x[tuple(idx)]


class SlidingWindowKVCache:
    """Recent-only baseline: no sink tokens, keep only latest cache_size tokens."""

    def __init__(self, cache_size: int, k_seq_dim: int, v_seq_dim: int):
        self.cache_size = int(cache_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = get_kv_seq_len(legacy_cache, self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        keep_from = seq_len - self.cache_size
        items = []
        for k, v in legacy_cache:
            items.append(
                (
                    slice_tensor_by_dim(k, self.k_seq_dim, keep_from, seq_len),
                    slice_tensor_by_dim(v, self.v_seq_dim, keep_from, seq_len),
                )
            )
        return _restore_cache_type(original_cache, tuple(items))

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = get_kv_seq_len(legacy_cache, self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        keep_from = max(0, seq_len - self.cache_size + num_coming)
        items = []
        for k, v in legacy_cache:
            items.append(
                (
                    slice_tensor_by_dim(k, self.k_seq_dim, keep_from, seq_len),
                    slice_tensor_by_dim(v, self.v_seq_dim, keep_from, seq_len),
                )
            )
        return _restore_cache_type(original_cache, tuple(items))


@torch.no_grad()
def run_decode_eval(
    model,
    input_ids,
    cache_obj,
    label: str,
    k_seq_dim: int,
    max_steps: int,
    progress_every: int = 100,
    eval_target_positions=None,
):
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    nlls = []
    step_times = []
    kv_lens = []

    total_steps = min(input_ids.size(1) - 1, max_steps)
    wall_start = time.perf_counter()
    print(f"[{label}] start: total_steps={total_steps}", flush=True)
    eval_set = set(eval_target_positions) if eval_target_positions is not None else None
    for idx in range(total_steps):
        token = input_ids[:, idx : idx + 1]
        target = input_ids[:, idx + 1 : idx + 2].to(token.device).view(-1)

        if cache_obj is not None:
            past_key_values = cache_obj.evict_for_space(past_key_values, num_coming=1)

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
        if eval_set is None or (idx + 1) in eval_set:
            nlls.append(nll)

        past_key_values = outputs.past_key_values
        if cache_obj is not None:
            past_key_values = cache_obj(past_key_values)

        kv_lens.append(get_kv_seq_len(past_key_values, k_seq_dim))

        step_id = idx + 1
        if progress_every > 0 and (step_id % progress_every == 0 or step_id == total_steps):
            elapsed = time.perf_counter() - wall_start
            inst_tok_s = step_id / elapsed if elapsed > 0 else float("inf")
            kv_now = kv_lens[-1] if kv_lens else 0
            print(
                f"[{label}] step={step_id}/{total_steps} kv={kv_now} "
                f"tok/s={inst_tok_s:.2f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if len(nlls) == 0:
        raise ValueError(
            "No target token selected for evaluation. Adjust max_steps or eval range."
        )
    mean_nll = torch.stack(nlls).mean().item()
    ppl = math.exp(mean_nll)
    total_s = sum(step_times)
    tok_per_s = total_steps / total_s if total_s > 0 else float("inf")

    return {
        "label": label,
        "steps": total_steps,
        "ppl": ppl,
        "tok_per_s": tok_per_s,
        "avg_ms_per_tok": (total_s / total_steps) * 1000.0,
        "max_kv_len": max(kv_lens) if kv_lens else 0,
        "final_kv_len": kv_lens[-1] if kv_lens else 0,
    }


def build_eval_text(repeat: int):
    unit = (
        "Streaming LLM benchmark text for cache strategy comparison. "
        "The sequence is repeated to trigger cache eviction behavior. "
    )
    return unit * repeat


def build_wikitext_eval_text(dataset_name, task, split, min_chars, sample_limit):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, task, split=split)
    chunks = []
    total_chars = 0
    for i, row in enumerate(ds):
        text = (row.get("text", "") or "").strip()
        if not text:
            continue
        chunks.append(text)
        total_chars += len(text)
        if total_chars >= min_chars or len(chunks) >= sample_limit:
            break
    if not chunks:
        raise ValueError("No usable text found in dataset split.")
    return "\n\n".join(chunks)


def build_needle_eval_input_ids(tokenizer, needle_pos, prefix_repeat=40):
    hay = (
        "The cat sat on the mat. "
        "The dog ran in the yard. "
        "The bird flew over the tree. "
    )
    needle = "The fox passed the dog and sat on the porch. "
    question = "\nWhere did the fox sit? The fox sat on the"
    answer = " porch."

    hay_tokens = tokenizer.encode(hay, add_special_tokens=False)
    prefix_tokens = hay_tokens * int(prefix_repeat)

    suffix_tokens = []
    target_suffix_len = int(max(0, needle_pos))
    while len(suffix_tokens) < target_suffix_len:
        suffix_tokens.extend(hay_tokens)
    suffix_tokens = suffix_tokens[:target_suffix_len]

    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

    full_tokens = (
        prefix_tokens + needle_tokens + suffix_tokens + question_tokens + answer_tokens
    )
    answer_start = (
        len(prefix_tokens)
        + len(needle_tokens)
        + len(suffix_tokens)
        + len(question_tokens)
    )
    answer_positions = list(range(answer_start, answer_start + len(answer_tokens)))
    # Ignore leading whitespace-only answer tokens to avoid inflating needle scores.
    while answer_positions:
        local_idx = answer_positions[0] - answer_start
        token_text = tokenizer.decode([answer_tokens[local_idx]])
        if token_text.strip() != "":
            break
        answer_positions.pop(0)
    return torch.tensor([full_tokens], dtype=torch.long), answer_positions


def print_table(results):
    print("\n=== Four-way comparison ===")
    print(
        f"{'mode':<16} {'steps':>7} {'ppl':>10} {'tok/s':>10} "
        f"{'avg ms/tok':>12} {'max kv len':>12} {'final kv len':>13}"
    )
    for r in results:
        print(
            f"{r['label']:<16} {r['steps']:>7d} {r['ppl']:>10.4f} {r['tok_per_s']:>10.2f} "
            f"{r['avg_ms_per_tok']:>12.3f} {r['max_kv_len']:>12d} {r['final_kv_len']:>13d}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text_repeat", type=int, default=120)
    parser.add_argument(
        "--text_source",
        type=str,
        default="wikitext",
        choices=["repeat", "wikitext", "needle"],
    )
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--wikitext_min_chars", type=int, default=12000)
    parser.add_argument("--wikitext_sample_limit", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--sketch_dim", type=int, default=1024)
    parser.add_argument("--recompute_interval", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l1_recent_keep", type=int, default=0)
    parser.add_argument("--mixed_recent_keep", type=int, default=64)
    parser.add_argument(
        "--comparison_mode",
        type=str,
        default="full",
        choices=["full", "three", "needle"],
        help=(
            "full: plain/sketching/main/sliding, "
            "three: recency/sink_l1_last/sink_recent_l1_last, "
            "needle: recency/main/sink_recent_l1_last"
        ),
    )
    parser.add_argument("--needle_pos", type=int, default=400)
    parser.add_argument("--needle_prefix_repeat", type=int, default=40)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--enable_pos_shift", action="store_true", default=True)
    parser.add_argument("--disable_pos_shift", action="store_false", dest="enable_pos_shift")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "streaming-llm-sketching"))
    sys.path.append(str(root / "streaming-llm-main"))
    sys.path.append(str(root / "streaming-llm-plain"))

    plain_mod = load_module_from_file(
        "plain_kv_mod",
        root / "streaming-llm-plain" / "streaming_llm" / "kv_cache.py",
    )
    main_mod = load_module_from_file(
        "main_kv_mod",
        root / "streaming-llm-main" / "streaming_llm" / "kv_cache.py",
    )
    sketch_mod = load_module_from_file(
        "sketch_kv_mod",
        root / "streaming-llm-sketching" / "streaming_llm" / "kv_cache.py",
    )

    print(f"Loading model: {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    torch.manual_seed(args.seed)
    maybe_enable_pos_shift(
        model,
        sketching_root=root / "streaming-llm-sketching",
        enabled=args.enable_pos_shift,
    )
    k_seq_dim, v_seq_dim = infer_kv_seq_dims(model.config.model_type)

    probe_ids = tokenizer("cache probe", return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad():
        probe_out = model(input_ids=probe_ids, use_cache=True)
    cache_format = "DynamicCache" if not isinstance(probe_out.past_key_values, tuple) else "legacy"
    print(f"KV cache format: {cache_format}")

    recent_size = max(1, args.cache_size - args.start_size)
    plain_cache = plain_mod.PlainKVCache()
    main_cache = main_mod.StartRecentKVCache(
        start_size=args.start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    sketch_cache = sketch_mod.L1RobustKVCache(
        cache_size=args.cache_size,
        num_sink_tokens=args.start_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        sketch_dim=args.sketch_dim,
        recompute_interval=args.recompute_interval,
        seed=args.seed,
        recent_keep=args.l1_recent_keep,
    )
    sketch_mixed_cache = sketch_mod.L1RobustKVCache(
        cache_size=args.cache_size,
        num_sink_tokens=args.start_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        sketch_dim=args.sketch_dim,
        recompute_interval=args.recompute_interval,
        seed=args.seed,
        recent_keep=args.mixed_recent_keep,
    )
    sliding_cache = SlidingWindowKVCache(
        cache_size=args.cache_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )

    eval_target_positions = None
    if args.text_source == "repeat":
        text = build_eval_text(args.text_repeat)
        print(f"text_source=repeat repeat={args.text_repeat}")
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    else:
        if args.text_source == "wikitext":
            text = build_wikitext_eval_text(
                dataset_name=args.dataset_name,
                task=args.task,
                split=args.split,
                min_chars=args.wikitext_min_chars,
                sample_limit=args.wikitext_sample_limit,
            )
            print(
                f"text_source=wikitext dataset={args.dataset_name}/{args.task} "
                f"split={args.split} min_chars={args.wikitext_min_chars}"
            )
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
        else:
            input_ids, eval_target_positions = build_needle_eval_input_ids(
                tokenizer,
                needle_pos=args.needle_pos,
                prefix_repeat=args.needle_prefix_repeat,
            )
            input_ids = input_ids.to(args.device)
            print(
                f"text_source=needle needle_pos={args.needle_pos} "
                f"prefix_repeat={args.needle_prefix_repeat}"
            )
            print(
                f"needle_eval_target_tokens={len(eval_target_positions)} "
                f"target_start={eval_target_positions[0] if eval_target_positions else 'NA'}"
            )
    print(
        f"Tokenized length: {input_ids.size(1)} | max_steps: {args.max_steps} "
        f"| cache_size: {args.cache_size} | start_size: {args.start_size}"
    )

    results = []
    if args.comparison_mode == "full":
        results.append(
            run_decode_eval(
                model,
                input_ids,
                plain_cache,
                label="plain",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                sketch_cache,
                label="sketching",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                main_cache,
                label="main",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                sliding_cache,
                label="sliding_window",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
    elif args.comparison_mode == "three":
        print(
            "comparison_mode=three: recency(sliding_window), "
            "sink_l1_last(sketching recent_keep=0), "
            "sink_recent_l1_last(sketching mixed_recent_keep)"
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                sliding_cache,
                label="recency_only",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                sketch_cache,
                label="sink_l1_last",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                sketch_mixed_cache,
                label="sink_recent_l1_last",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
    else:
        print(
            "comparison_mode=needle: recency(sliding_window), "
            "main(start+recent), l1_mixed(mixed recent+l1)"
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                sliding_cache,
                label="recency_only",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                main_cache,
                label="main",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )
        results.append(
            run_decode_eval(
                model,
                input_ids,
                sketch_mixed_cache,
                label="l1_mixed",
                k_seq_dim=k_seq_dim,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                eval_target_positions=eval_target_positions,
            )
        )

    print_table(results)
    print("\n=== How to read ===")
    if args.comparison_mode == "full":
        print("- plain: no eviction baseline; usually best ppl, growing KV.")
        print("- main: start+recent heuristic from original StreamingLLM.")
        print("- sliding_window: recent-only baseline without sink tokens.")
        print("- sketching: your L1-robust policy; target is lower ppl than window/main under budget.")
    elif args.comparison_mode == "three":
        print("- recency_only: pure recent window baseline.")
        print("- sink_l1_last: sink + l1-selected history + last token.")
        print(
            "- sink_recent_l1_last: sink + recent_keep + "
            "l1-selected older history + last token."
        )
    else:
        print("- recency_only: pure recent window baseline.")
        print("- main: start+recent baseline.")
        print("- l1_mixed: mixed strategy (recent+historical l1).")
        print("- In needle mode, ppl is computed only on answer tokens.")


if __name__ == "__main__":
    main()
