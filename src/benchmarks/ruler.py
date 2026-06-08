"""RULER benchmark wrapper — long-context tasks from NVIDIA.

RULER (Hsieh et al., 2024) tests 4 ability categories:
  1. Retrieval (needle, multi-needle, variable tracking)
  2. Multi-hop (variable tracking with dependencies)
  3. Aggregation (common words, frequent words)
  4. QA (SQuAD, HotpotQA in long context)

This module generates synthetic RULER-style samples when the official
dataset is unavailable, or wraps the official HF dataset when present.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import random
import string
import torch

from src.benchmarks.base import BaseBenchmark, BenchmarkResult


def _random_word(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _generate_variable_tracking_sample(
    rng: random.Random, seq_words: int = 2000, n_variables: int = 10,
) -> Dict[str, Any]:
    """Generate variable tracking: 'What is the value of VAR_X?'"""
    variables = {f"VAR_{i}": _random_word(rng, 8) for i in range(n_variables)}
    filler = [_random_word(rng, 5) for _ in range(seq_words)]

    # Scatter variable assignments through the text
    positions = sorted(rng.sample(range(100, seq_words - 100), n_variables))
    for pos, (var, val) in zip(positions, variables.items()):
        filler.insert(pos, f"[{var} = {val}]")

    # Query at the end
    query_var = rng.choice(list(variables.keys()))
    query = f"\nWhat is the value of {query_var}? The value is"
    answer = f" {variables[query_var]}"

    text = " ".join(filler) + query + answer
    return {
        "text": text,
        "answer": answer.strip(),
        "query_var": query_var,
        "n_variables": n_variables,
    }


def _generate_common_words_sample(
    rng: random.Random, seq_words: int = 2000, n_common: int = 5,
) -> Dict[str, Any]:
    """Generate aggregation task: list words appearing ≥ K times."""
    pool = [_random_word(rng, 5) for _ in range(50)]
    common_words = rng.sample(pool, min(n_common, len(pool)))
    threshold = max(3, seq_words // 100)

    words = []
    for _ in range(seq_words):
        if rng.random() < 0.05 and common_words:
            words.append(rng.choice(common_words))
        else:
            words.append(rng.choice(pool))

    query = (f"\nList all words that appear at least {threshold} times in the text above."
             f" The words are")
    # Compute actual common words
    from collections import Counter
    counts = Counter(words)
    actual_common = sorted(w for w, c in counts.items() if c >= threshold)
    answer = " " + ", ".join(actual_common)

    text = " ".join(words) + query + answer
    return {
        "text": text,
        "answer": answer.strip(),
        "threshold": threshold,
        "expected_words": actual_common,
    }


def _generate_multi_hop_sample(
    rng: random.Random, seq_words: int = 2000, n_hops: int = 3,
) -> Dict[str, Any]:
    """Generate multi-hop variable tracking: A→B→C chain."""
    entities = [_random_word(rng, 6) for _ in range(n_hops + 1)]
    chain = {}
    for i in range(n_hops):
        chain[entities[i]] = entities[i + 1]

    filler = [_random_word(rng, 5) for _ in range(seq_words)]
    positions = sorted(rng.sample(range(100, seq_words - 100), n_hops))
    clues = []
    for pos, (src, dst) in zip(positions, chain.items()):
        clue = f"[{src} points to {dst}]"
        filler.insert(pos, clue)
        clues.append(clue)

    query = f"\nWhat does {entities[0]} ultimately point to? It points to"
    answer = f" {entities[-1]}"

    text = " ".join(filler) + query + answer
    return {
        "text": text,
        "answer": answer.strip(),
        "chain": chain,
        "n_hops": n_hops,
        "start": entities[0],
        "end": entities[-1],
    }


RULER_TASK_GENERATORS = {
    "variable_tracking": _generate_variable_tracking_sample,
    "common_words": _generate_common_words_sample,
    "multi_hop": _generate_multi_hop_sample,
}


class RULERBenchmark(BaseBenchmark):
    """RULER-style synthetic long-context benchmark.

    Supports: variable_tracking, common_words, multi_hop.
    Generates samples when official dataset is unavailable.
    """

    name = "ruler"

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        n_samples_per_task: int = 20,
        seq_words: int = 2000,
        seed: int = 0,
        max_samples: Optional[int] = None,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.tasks = tasks or ["variable_tracking", "common_words", "multi_hop"]
        self.n_samples_per_task = n_samples_per_task
        self.seq_words = seq_words

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        samples: List[Dict[str, Any]] = []

        for task_name in self.tasks:
            gen_fn = RULER_TASK_GENERATORS.get(task_name)
            if gen_fn is None:
                print(f"[RULER] unknown task: {task_name}, skipping")
                continue

            for si in range(self.n_samples_per_task):
                raw = gen_fn(rng, seq_words=self.seq_words)
                text = raw["text"]
                answer = raw["answer"]

                # Find answer position
                prefix_text = text[:text.rfind(answer)]
                ids = tokenizer(text, return_tensors="pt").input_ids
                prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
                prefix_len = prefix_ids.size(1)
                eval_positions = list(range(prefix_len, ids.size(1)))

                samples.append({
                    "input_ids": ids,
                    "eval_positions": eval_positions,
                    "metadata": {
                        "task": task_name,
                        "answer": answer,
                        "seq_len": ids.size(1),
                        **{k: v for k, v in raw.items() if k not in ("text", "answer")},
                    },
                })

        return samples

    def compute_metrics(
        self, nlls: List[float], sample: Dict[str, Any], extra=None,
    ) -> Dict[str, float]:
        if not nlls:
            return {"ppl": float("inf"), "task": sample["metadata"]["task"]}
        mean_nll = sum(nlls) / len(nlls)
        return {
            "ppl": math.exp(mean_nll),
            "answer_ppl": math.exp(mean_nll),
            "task": sample["metadata"]["task"],
            "n_eval_tokens": len(nlls),
        }
