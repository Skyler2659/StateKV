import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from streaming_llm.l1_sketch import L1LeverageScoreEstimator


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--text_repeat", type=int, default=80)
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--sketch_dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).eval()
    text = (
        "L1 robust KV score analysis text. "
        "This repetition creates long token streams for ranking. "
    ) * args.text_repeat
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    out = model(input_ids=input_ids, use_cache=True)
    pkv = out.past_key_values

    if not isinstance(pkv, tuple):
        print("DynamicCache detected; this analysis script expects legacy tuple cache.")
        return

    k, v = pkv[0]
    v_rows = v[0].mean(dim=0)  # [S, D]
    estimator = L1LeverageScoreEstimator(
        sketch_dim=max(args.sketch_dim, v_rows.shape[1]),
        seed=args.seed,
    )
    scores = estimator.scores(v_rows, force_refit=True)
    keep_idx = torch.topk(scores, k=min(args.cache_size, scores.numel())).indices.sort().values

    top_values, top_idx = torch.topk(scores, k=min(20, scores.numel()))
    print("=== L1 score analysis ===")
    print(f"seq_len={scores.numel()} cache_size={args.cache_size}")
    print(f"selected={keep_idx.numel()} first10={keep_idx[:10].tolist()}")
    print("top score tokens:")
    for rank, (idx, val) in enumerate(zip(top_idx.tolist(), top_values.tolist()), start=1):
        print(f"{rank:2d}. token_idx={idx:5d} score={val:.6f}")


if __name__ == "__main__":
    main()
