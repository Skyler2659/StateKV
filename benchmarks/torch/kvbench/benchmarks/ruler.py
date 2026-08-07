"""Official-dataset RULER adapter plus deterministic smoke generators."""
from __future__ import annotations

import random
import json
import string
from pathlib import Path
from typing import Any, Dict, List

from kvbench.benchmarks.base import BenchmarkAdapter, references_from_value
from kvbench.types import BenchmarkSample


RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


def _task_metadata(task: str) -> Dict[str, Any]:
    if task.startswith("niah"):
        return {"category": "retrieval", "official_max_new_tokens": 128}
    if task in {"vt", "variable_tracking"}:
        return {"category": "vt_aggregation", "official_max_new_tokens": 30}
    if task in {"cwe", "common_words", "common_words_extraction"}:
        return {"category": "vt_aggregation", "official_max_new_tokens": 120}
    if task in {"fwe", "freq_words", "freq_words_extraction"}:
        return {"category": "vt_aggregation", "official_max_new_tokens": 50}
    if task in {"qa", "qa_1", "qa_2"}:
        return {"category": "qa", "official_max_new_tokens": 32}
    raise ValueError("unsupported RULER task: %s" % task)


def _word(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _synthetic_niah(seed: int, count: int, context_length: int) -> List[BenchmarkSample]:
    samples = []
    for index in range(count):
        rng = random.Random(seed + index * 1009)
        key = "%s-%s" % (_word(rng), _word(rng, 5))
        value = "".join(rng.choice(string.digits) for _ in range(7))
        needle = "The special magic number for %s is %s." % (key, value)
        # Four simple words are roughly six tokenizer tokens; exact length is
        # truncated later by the runner and recorded as actual_input_length.
        repeats = max(16, int(context_length) // 6)
        filler = ["The sky is blue and grass is green."] * repeats
        insert = rng.randrange(len(filler))
        filler.insert(insert, needle)
        context = " ".join(filler)
        prompt = (
            context
            + "\n\nWhat is the special magic number for %s? " % key
            + "Answer with only the number."
        )
        samples.append(
            BenchmarkSample(
                sample_id="synthetic_niah_%d" % index,
                prompt=prompt,
                references=[value],
                task="niah_single_1",
                answer_text=value,
                full_text=prompt + " " + value,
                metadata={
                    "dataset_official": False,
                    **_task_metadata("niah_single_1"),
                    "evidence_texts": [needle],
                    "needle_depth": float(insert) / max(1, len(filler) - 1),
                },
            )
        )
    return samples


def _synthetic_vt(seed: int, count: int, context_length: int) -> List[BenchmarkSample]:
    samples = []
    for index in range(count):
        rng = random.Random(seed + index * 9173)
        assignments = [("VAR_%d" % i, _word(rng, 8)) for i in range(8)]
        filler = [_word(rng, 5) for _ in range(max(200, context_length // 2))]
        evidence = []
        for variable, value in assignments:
            statement = "[%s = %s]" % (variable, value)
            filler.insert(rng.randrange(len(filler)), statement)
            evidence.append(statement)
        query_variable, target = rng.choice(assignments)
        prompt = " ".join(filler) + "\nWhat is the value of %s?" % query_variable
        samples.append(
            BenchmarkSample(
                sample_id="synthetic_vt_%d" % index,
                prompt=prompt,
                references=[target],
                task="vt",
                answer_text=target,
                full_text=prompt + " " + target,
                metadata={
                    "dataset_official": False,
                    **_task_metadata("vt"),
                    "evidence_texts": [
                        statement for statement in evidence if query_variable in statement
                    ],
                },
            )
        )
    return samples


class RULERBenchmark(BenchmarkAdapter):
    def load(self) -> List[BenchmarkSample]:
        task_metadata = _task_metadata(self.cfg.task)
        if not self.cfg.dataset_name and not self.cfg.data_path:
            if self.cfg.require_official:
                raise RuntimeError(
                    "RULER paper runs require benchmark.dataset_name and an official dataset cache"
                )
            if self.cfg.task in {"vt", "variable_tracking"}:
                return _synthetic_vt(
                    self.seed, int(self.cfg.num_samples), int(self.cfg.context_length)
                )
            return _synthetic_niah(
                self.seed, int(self.cfg.num_samples), int(self.cfg.context_length)
            )

        dataset_config = self.cfg.dataset_config or self.cfg.task
        if self.cfg.data_path:
            path = Path(self.cfg.data_path)
            with open(path, "r", encoding="utf-8") as handle:
                dataset = [json.loads(line) for line in handle if line.strip()]
            source = str(path.resolve())
        else:
            from datasets import load_dataset

            split = self.cfg.split or dataset_config
            kwargs: Dict[str, Any] = {
                "path": self.cfg.dataset_name,
                "name": dataset_config,
                "split": split,
                "trust_remote_code": True,
            }
            if self.cfg.dataset_revision:
                kwargs["revision"] = self.cfg.dataset_revision
            dataset = load_dataset(**kwargs)
            source = str(self.cfg.dataset_name)
        samples: List[BenchmarkSample] = []
        for index, row_obj in enumerate(dataset):
            row = dict(row_obj)
            prompt = row.get("input") or row.get("prompt")
            if not prompt:
                context = row.get("context", "")
                question = row.get("question", "")
                answer_prefix = row.get("answer_prefix", "")
                prompt = "%s\n\n%s%s" % (context, question, answer_prefix)
            references = references_from_value(
                row.get("outputs", row.get("answers", row.get("answer")))
            )
            if not references:
                raise RuntimeError("official RULER row has no answer at index=%d" % index)
            evidence_texts = references_from_value(
                row.get("evidence", row.get("needle", row.get("clues")))
            )
            samples.append(
                BenchmarkSample(
                    sample_id=str(row.get("sample_id", row.get("index", "%s:%d" % (self.cfg.task, index)))),
                    prompt=str(prompt),
                    references=references,
                    task=self.cfg.task,
                    answer_text=references[0],
                    full_text=str(prompt) + " " + references[0],
                    metadata={
                        "dataset_official": True,
                        **task_metadata,
                        "official_dataset_index": index,
                        "dataset_name": self.cfg.dataset_name,
                        "dataset_config": dataset_config,
                        "dataset_revision": self.cfg.dataset_revision,
                        "evidence_texts": evidence_texts,
                        "raw_length": row.get("length"),
                        "source": source,
                    },
                )
            )
        selected = self.select(samples)
        if not selected:
            raise RuntimeError("official RULER dataset produced zero selected samples")
        return selected
