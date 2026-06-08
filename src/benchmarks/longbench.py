"""LongBench wrapper — standard long-context QA and summarization tasks.

LongBench (Bai et al., 2023) covers:
- Single-document QA (NarrativeQA, QuALITY, TriviaQA)
- Multi-document QA (HotpotQA, 2WikiMultihopQA, MuSiQue)
- Summarization (GovReport, QMSum, MultiNews)
- Few-shot learning (TREC, SAMSum)
- Code completion (LCC, RepoBench-P)
- Synthetic (PassageCount, PassageRetrieval)

This wrapper loads the HF dataset and converts samples to the common
benchmark interface.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import torch

from src.benchmarks.base import BaseBenchmark, BenchmarkResult


# LongBench task categories
TASK_CATEGORIES = {
    "single_doc_qa": ["narrativeqa", "quality", "triviaqa"],
    "multi_doc_qa": ["hotpotqa", "2wikimultihopqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "samsum"],
    "code": ["lcc", "repobench-p"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
}

ALL_TASKS = [t for tasks in TASK_CATEGORIES.values() for t in tasks]


def _truncate_to_max_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _build_qa_sample(
    row: Dict[str, Any], max_words: int = 4000,
) -> Dict[str, Any]:
    """Build a QA sample from a LongBench row."""
    context = row.get("context", "") or ""
    question = row.get("input", "") or row.get("question", "") or ""
    answers = row.get("answers", []) or []
    if isinstance(answers, str):
        answers = [answers]

    context = _truncate_to_max_words(context, max_words)
    text = f"{context}\n\nQuestion: {question}\nAnswer:"
    answer_text = f" {answers[0]}" if answers else " unknown"
    full_text = text + answer_text

    return {
        "text": full_text,
        "prefix_text": text,
        "answer": answer_text.strip(),
        "answers": answers,
        "task": row.get("_task", "unknown"),
    }


class LongBenchWrapper(BaseBenchmark):
    """Wraps LongBench HF dataset for KV cache eviction evaluation.

    Uses PPL on answer tokens as the primary metric (consistent with
    the rest of the framework).

    Args:
        tasks: list of LongBench task names (e.g. ["narrativeqa", "hotpotqa"])
        max_words: max context words per sample
        n_samples_per_task: max samples per task
        seed: random seed for sample selection
    """

    name = "longbench"

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        max_words: int = 4000,
        n_samples_per_task: int = 50,
        seed: int = 0,
        max_samples: Optional[int] = None,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.tasks = tasks or ["narrativeqa", "hotpotqa", "triviaqa"]
        self.max_words = max_words
        self.n_samples_per_task = n_samples_per_task

    def _load_task_data(self, task_name: str) -> List[Dict[str, Any]]:
        """Load a single LongBench task from HF datasets."""
        try:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", task_name, split="test",
                              trust_remote_code=True)
            rows = []
            for i, row in enumerate(ds):
                if i >= self.n_samples_per_task:
                    break
                row["_task"] = task_name
                rows.append(dict(row))
            return rows
        except Exception as e:
            print(f"[LongBench] failed to load {task_name}: {e}")
            return []

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for task_name in self.tasks:
            rows = self._load_task_data(task_name)
            if not rows:
                print(f"[LongBench] no data for task: {task_name}")
                continue

            for row in rows:
                raw = _build_qa_sample(row, max_words=self.max_words)
                text = raw["text"]
                prefix_text = raw["prefix_text"]

                ids = tokenizer(text, return_tensors="pt").input_ids
                prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
                prefix_len = prefix_ids.size(1)
                eval_positions = list(range(prefix_len, ids.size(1)))

                if not eval_positions:
                    continue

                samples.append({
                    "input_ids": ids,
                    "eval_positions": eval_positions,
                    "metadata": {
                        "task": task_name,
                        "answer": raw["answer"],
                        "answers": raw.get("answers", []),
                        "seq_len": ids.size(1),
                        "prefix_len": prefix_len,
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
