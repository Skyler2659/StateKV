import torch
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.utils import parse_args, load
from streaming_llm.enable_streaming_llm import (
    enable_l1_robust_llm,
    enable_plain_llm,
    enable_streaming_llm,
)

device = "cuda"

args = parse_args()

data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

if getattr(args, "kv_strategy", "plain") == "plain":
    kv_cache = enable_plain_llm(model)
elif args.kv_strategy == "streaming":
    kv_cache = enable_streaming_llm(
        model,
        start_size=args.start_size,
        recent_size=args.recent_size,
    )
elif args.kv_strategy == "l1_robust":
    kv_cache = enable_l1_robust_llm(
        model,
        cache_size=args.cache_size,
        num_sink_tokens=args.start_size,
        sketch_dim=args.sketch_dim,
        recompute_interval=args.recompute_interval,
        seed=args.seed,
        per_layer=args.per_layer,
        use_reweight=args.use_reweight,
    )
else:
    raise ValueError(f"Unknown kv_strategy: {args.kv_strategy}")

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for text in data["text"][: args.num_samples]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            past_key_values = kv_cache.evict_for_space(past_key_values, 1)
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            past_key_values = kv_cache(past_key_values)
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"strategy={args.kv_strategy}\n")
    f.write(f"ppl={ppl.item()}\n")
