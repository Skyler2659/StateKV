"""Data-source helpers for PPL / needle benchmarks — no model or cache dependency."""
import torch


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
    for row in ds:
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


def build_long_text(split, sample_idx, target_words):
    """Standard long-document PPL source (wikitext-103, always available)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    words = []
    for i in range(len(ds)):
        text = (ds[i]["text"] or "").strip()
        if not text:
            continue
        words.extend(text.split())
        if len(words) >= target_words:
            break
    return " ".join(words[:target_words])


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

    full_tokens = prefix_tokens + needle_tokens + suffix_tokens + question_tokens + answer_tokens
    answer_start = len(prefix_tokens) + len(needle_tokens) + len(suffix_tokens) + len(question_tokens)
    answer_positions = list(range(answer_start, answer_start + len(answer_tokens)))
    # Ignore leading whitespace-only answer tokens to avoid inflating needle scores.
    while answer_positions:
        local_idx = answer_positions[0] - answer_start
        token_text = tokenizer.decode([answer_tokens[local_idx]])
        if token_text.strip() != "":
            break
        answer_positions.pop(0)
    return torch.tensor([full_tokens], dtype=torch.long), answer_positions
