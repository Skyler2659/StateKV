"""Official SCBench same-context-different-query (SCDQ) adapter."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kvbench.benchmarks.base import BenchmarkAdapter, references_from_value
from kvbench.types import BenchmarkSample


# Ported from microsoft/MInference/scbench/eval_utils.py.  Keeping these values
# here makes the exact prompt/generation contract visible in resolved artifacts.
OFFICIAL_MAX_NEW_TOKENS = {
    "scbench_choice_eng": 40,
    "scbench_qa_eng": 40,
    "scbench_qa_chn": 40,
    "scbench_kv": 150,
    "scbench_kv_hard": 150,
    "scbench_mf": 5,
    "scbench_hashhop": 150,
    "scbench_prefix_suffix": 150,
    "scbench_kv_compressible": 150,
    "scbench_passkey": 15,
    "scbench_summary": 200,
    "scbench_vt": 30,
    "scbench_many_shot": 10,
}

OFFICIAL_SUPPORTED_TASKS = frozenset(OFFICIAL_MAX_NEW_TOKENS)
OFFICIAL_UNSUPPORTED_TASKS = {
    # RepoQA uses the official repository-level pass@1 scorer and additional
    # repo/function metadata; mixed tasks require separate subtask aggregates.
    "scbench_repoqa",
    "scbench_repoqa_and_kv",
    "scbench_summary_with_needles",
}

_SIMPLE_PREFIXES = {
    "scbench_passkey": (
        "There is an important info hidden inside a lot of irrelevant text.\n"
        "Find it and memorize it.\n"
        "I will quiz you about the important information.\n\n{context}"
    ),
    "scbench_kv": (
        "Extract the value corresponding to the specified key in the JSON object "
        "below.\n\n{context}"
    ),
    "scbench_kv_hard": (
        "Extract the value corresponding to the specified key in the JSON object "
        "below.\n\n{context}"
    ),
    "scbench_kv_compressible": (
        "Extract the value corresponding to the specified key in the following "
        "passage.\n\n{context}"
    ),
    "scbench_repoqa": (
        "Based on the function description and code context, please retrieve and "
        "repeat the exact described function from the code context in a code block "
        "wrapped by ```:\n\n{context}"
    ),
}


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _reference_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _official_ground_truth(task: str, turn: Dict[str, Any]) -> Any:
    answer = turn["answer"]
    if task == "scbench_choice_eng":
        options = list(turn["options"])
        if len(options) != 4 or answer not in options:
            raise ValueError("SCBench choice answer is not one of four options")
        return [answer, "ABCD"[options.index(answer)]]
    if task == "scbench_qa_eng":
        return [answer]
    return answer


def _official_prompts(
    task: str, context: str, turns: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """Port official ``create_scdq_prompt`` for its non-chat-template path."""
    if task == "scbench_choice_eng":
        prefix = "Read the book and answer the question.\n\n%s" % context
        queries = []
        for turn in turns:
            options = list(turn["options"])
            if len(options) != 4:
                raise ValueError("SCBench choice turn requires exactly four options")
            queries.append(
                "Question: %s\nA. %s\nB. %s\nC. %s\nD. %s\n\n"
                "The the correct answer is"
                % (turn["input"], options[0], options[1], options[2], options[3])
            )
        return prefix, queries
    if task == "scbench_qa_eng":
        prefix = (
            "Read the book and answer the question. Be very concise in your "
            "answer.\n\n%s" % context
        )
        return prefix, ["Question: %s\nAnswer:" % turn["input"] for turn in turns]
    if task == "scbench_qa_chn":
        prefix = "阅读以下书籍然后回答问题。\n\n%s" % context
        return prefix, ["问题：%s\n答案：" % turn["input"] for turn in turns]
    if task == "scbench_mf":
        prefix = "%s\n\n%s" % (turns[0]["input"], context)
        queries = []
        for turn in turns:
            match = re.findall(r"The .+ is", str(turn["input"]))
            if not match:
                raise ValueError("SCBench Math.Find input does not match official template")
            target = match[0].lower()[:-3]
            queries.append("What is %s?\n\n%s" % (target, turn["input"]))
        return prefix, queries
    template = _SIMPLE_PREFIXES.get(task, "{context}")
    return template.format(context=context), [str(turn["input"]) for turn in turns]


class SCBenchBenchmark(BenchmarkAdapter):
    """Load official HF rows or a byte-for-byte JSONL export of those rows.

    Official input uses ``context`` plus ``multi_turns``.  The older normalized
    ``shared_prefix``/``queries``/``answers`` schema remains available only for
    non-official smoke tests so it cannot silently enter a paper table.
    """

    def _load_rows(self) -> Tuple[Iterable[Dict[str, Any]], str, Any]:
        if self.cfg.data_path:
            path = Path(self.cfg.data_path)
            if not path.exists():
                raise FileNotFoundError("SCBench data_path does not exist: %s" % path)
            return _jsonl_rows(path), str(path), None
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("SCBench Hugging Face loading requires datasets") from exc
        dataset_name = self.cfg.dataset_name or "microsoft/SCBench"
        dataset_config = self.cfg.dataset_config or self.cfg.task
        kwargs: Dict[str, Any] = {
            "path": dataset_name,
            "name": dataset_config,
            "split": self.cfg.split or "test",
        }
        if self.cfg.dataset_revision:
            kwargs["revision"] = self.cfg.dataset_revision
        dataset = load_dataset(**kwargs)
        source = "%s/%s:%s" % (dataset_name, dataset_config, self.cfg.split or "test")
        return dataset, source, getattr(dataset, "_fingerprint", None)

    def _normalized_sample(
        self, row: Dict[str, Any], index: int, source: str
    ) -> BenchmarkSample:
        if self.cfg.require_official:
            raise RuntimeError(
                "official SCBench runs require raw context/multi_turns rows; the "
                "normalized schema is accepted only with require_official=false"
            )
        prefix = str(row["shared_prefix"])
        queries = [str(value) for value in row["queries"]]
        answers = references_from_value(row["answers"])
        if len(queries) != len(answers):
            raise ValueError("SCBench queries/answers length mismatch at row=%d" % index)
        return BenchmarkSample(
            sample_id=str(row.get("sample_id", "%s:%d" % (self.cfg.task, index))),
            prompt=prefix + "\n\n" + queries[0],
            references=answers,
            task=self.cfg.task,
            answer_text=answers[0],
            shared_prefix=prefix,
            queries=queries,
            metadata={
                "dataset_official": False,
                "query_count": len(queries),
                "scbench_mode": row.get("mode", self.cfg.task),
                "scbench_schema": "normalized_smoke",
                "scbench_prompt_implementation": "user_normalized",
                "source": source,
            },
        )

    def _official_sample(
        self,
        row: Dict[str, Any],
        index: int,
        source: str,
        dataset_fingerprint: Any,
    ) -> BenchmarkSample:
        task = self.cfg.task
        if task in OFFICIAL_UNSUPPORTED_TASKS:
            raise RuntimeError(
                "SCBench task=%s is intentionally blocked: its official scorer "
                "requires RepoQA pass@1 or a multi-subtask aggregate" % task
            )
        if task not in OFFICIAL_SUPPORTED_TASKS:
            raise RuntimeError(
                "SCBench task=%s has no audited official prompt/scorer port" % task
            )
        turns = list(row.get("multi_turns") or [])
        if not turns:
            raise ValueError("SCBench official row=%d has no multi_turns" % index)
        context = str(row["context"])
        if self.cfg.use_official_prompt:
            prefix, queries = _official_prompts(task, context, turns)
            prompt_implementation = "minference_create_scdq_prompt_python_port"
        else:
            if self.cfg.require_official:
                raise RuntimeError("official SCBench runs require use_official_prompt=true")
            prefix, queries = context, [str(turn["input"]) for turn in turns]
            prompt_implementation = "raw_context_nonofficial"
        query_labels = [_official_ground_truth(task, turn) for turn in turns]
        references = [_reference_text(turn["answer"]) for turn in turns]
        sample_id = str(row.get("id", index))
        return BenchmarkSample(
            sample_id="%s:%s" % (task, sample_id),
            prompt=prefix + "\n\n" + queries[0],
            references=references,
            task=task,
            answer_text=references[0],
            shared_prefix=prefix,
            queries=queries,
            metadata={
                "dataset_official": bool(row.get("dataset_official", True)),
                "official_dataset_index": index,
                "official_dataset_id": row.get("id", index),
                "dataset_fingerprint": dataset_fingerprint,
                "query_count": len(queries),
                "query_labels": query_labels,
                "query_tasks": [task] * len(queries),
                "scbench_mode": "same_context_different_query",
                "scbench_schema": "official_context_multi_turns",
                "scbench_prompt_implementation": prompt_implementation,
                "official_max_new_tokens": OFFICIAL_MAX_NEW_TOKENS[task],
                "source": source,
            },
        )

    def load(self) -> List[BenchmarkSample]:
        rows, source, fingerprint = self._load_rows()
        samples: List[BenchmarkSample] = []
        for index, row in enumerate(rows):
            if "context" in row and "multi_turns" in row:
                sample = self._official_sample(row, index, source, fingerprint)
            elif {"shared_prefix", "queries", "answers"}.issubset(row):
                sample = self._normalized_sample(row, index, source)
            else:
                raise ValueError(
                    "unrecognized SCBench schema at row=%d; expected official "
                    "context/multi_turns" % index
                )
            samples.append(sample)
        selected = self.select(samples)
        if not selected:
            raise RuntimeError("SCBench source produced zero selected samples")
        return selected
