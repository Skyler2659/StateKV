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


def build_needle_std_input_ids(tokenizer, needle_depth_pct=0.5, max_words=8000):
    """Standard Needle-in-a-Haystack: long natural text + OOV needle.

    Uses wikitext-103 as the haystack.  The needle is an out-of-vocabulary
    fact placed at *needle_depth_pct* through the text, and the question
    always appears at the very end.  This matches the setup used by most
    published KV-cache papers.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Build a long contiguous text from articles
    words = []
    for row in ds:
        t = (row.get("text", "") or "").strip()
        if not t:
            continue
        words.extend(t.split())
        if len(words) >= max_words + 2000:
            break
    haystack_words = words[:max_words]

    needle = "\n\nThe secret passcode is ZEBRA-8842. Remember this passcode.\n\n"
    question = "\nWhat is the secret passcode? The passcode is"
    answer = " ZEBRA-8842"

    haystack = " ".join(haystack_words)
    needle_pos_words = max(5, min(len(haystack_words) - 5,
                                  int(len(haystack_words) * needle_depth_pct)))
    prefix = " ".join(haystack_words[:needle_pos_words])
    suffix = " ".join(haystack_words[needle_pos_words:])

    full_text = prefix + needle + suffix + question + answer
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Locate answer positions by encoding the answer alone and scanning
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    question_tokens_bare = tokenizer.encode(question.strip(), add_special_tokens=False)
    needle_tokens_bare = tokenizer.encode(needle.strip(), add_special_tokens=False)

    # Find the answer span at the end of the tokenized sequence
    ans_start = len(full_tokens) - len(answer_tokens)
    # Verify match
    if full_tokens[ans_start:ans_start + len(answer_tokens)] != answer_tokens:
        # Fallback: search for answer token sequence
        for i in range(len(full_tokens) - len(answer_tokens), max(0, len(full_tokens) - 200), -1):
            if full_tokens[i:i + len(answer_tokens)] == answer_tokens:
                ans_start = i
                break

    answer_positions = list(range(ans_start, ans_start + len(answer_tokens)))
    # Strip leading whitespace-only tokens
    while answer_positions:
        local_idx = answer_positions[0] - ans_start
        token_text = tokenizer.decode([answer_tokens[local_idx]])
        if token_text.strip() != "":
            break
        answer_positions.pop(0)
    return torch.tensor([full_tokens], dtype=torch.long), answer_positions


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
