"""Task loading with explicit, recorded same-category fallbacks."""
from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from kvbench.benchmarks.longbench import LongBenchBenchmark
from kvbench.benchmarks.ruler import _synthetic_niah
from statekv.config import DiscoveryConfig
from kvbench.types import BenchmarkSample


def _extend_retrieval_prompt(sample: BenchmarkSample) -> BenchmarkSample:
    suffix = (
        "\nState the requested number first. Then give a detailed account of how "
        "you located the relevant statement and distinguished it from the filler. "
        "Continue for at least 100 words without introducing a different number."
    )
    sample.prompt += suffix
    sample.full_text = sample.prompt + " " + (sample.answer_text or "")
    sample.metadata["discovery_generation_instruction"] = "detailed_retrieval_trace"
    return sample


def _fallback_reports(seed: int, count: int, paragraphs: int) -> List[BenchmarkSample]:
    topics = [
        ("regional transit", "reliability", "maintenance backlog"),
        ("watershed restoration", "water quality", "habitat monitoring"),
        ("public health preparedness", "response time", "training coverage"),
        ("school facilities", "energy use", "deferred repairs"),
        ("digital services", "accessibility", "legacy system migration"),
    ]
    samples = []
    for index in range(count):
        rng = random.Random(seed + index * 7919)
        agency, primary, secondary = topics[index % len(topics)]
        body = []
        for paragraph in range(paragraphs):
            year = 2018 + paragraph % 7
            amount = 40 + rng.randrange(160)
            body.append(
                (
                    "Section %d. In %d the %s program reviewed %d operating units. "
                    "The review tracked %s, %s, staffing constraints, procurement "
                    "delays, and regional variation. Reported observations were "
                    "checked against quarterly records, stakeholder interviews, "
                    "and follow-up inspections. The agency listed both completed "
                    "actions and items still awaiting verification."
                )
                % (paragraph + 1, year, agency, amount, primary, secondary)
            )
        prompt = (
            "You are given a report by a government agency. Write a one-page "
            "summary of the report, covering objectives, evidence, limitations, "
            "and follow-up actions.\n\nReport:\n"
            + "\n\n".join(body)
            + "\n\nSummary:"
        )
        samples.append(
            BenchmarkSample(
                sample_id="fallback_gov_report_%d" % index,
                prompt=prompt,
                references=["Deterministic synthetic report; no official reference used."],
                task="gov_report",
                metadata={
                    "category": "summarization",
                    "dataset_official": False,
                    "fallback": "deterministic_long_report",
                },
            )
        )
    return samples


_DISTRACTORS = [
    "The municipal archive completed its annual inventory of paper records.",
    "A coastal weather station reported ordinary seasonal temperature changes.",
    "The library extended weekend opening hours during the examination period.",
    "A transport survey counted bicycles at twelve intersections.",
    "The parks department replaced signs along three walking trails.",
    "A museum catalogued a collection of photographs from the nineteen sixties.",
    "The water utility inspected valves in several residential districts.",
    "A public workshop discussed tree planting and pavement maintenance.",
]


def _reasoning_samples(seed: int, count: int, distractors: int) -> List[BenchmarkSample]:
    samples = []
    for index in range(count):
        rng = random.Random(seed + index * 3571)
        a = 20 + rng.randrange(81)
        b = 2 + rng.randrange(9)
        c = 5 + rng.randrange(46)
        answer = a * b + c
        problem = (
            "A depot receives %d crates. Each crate contains %d instruments, "
            "and an additional shipment contains %d instruments. How many "
            "instruments are there in total?" % (a, b, c)
        )
        filler = [rng.choice(_DISTRACTORS) for _ in range(distractors)]
        insert = rng.randrange(max(1, distractors // 4), max(2, 3 * distractors // 4))
        filler.insert(insert, "[Problem] " + problem)
        prompt = (
            "Read the material and solve the embedded arithmetic problem.\n\n"
            + " ".join(filler)
            + "\n\nProblem: "
            + problem
            + "\nGive a careful step-by-step derivation, check the arithmetic in "
            "a second way, discuss which quantities are relevant, and state the "
            "final answer. Continue the explanation for at least 100 words.\nAnswer:"
        )
        samples.append(
            BenchmarkSample(
                sample_id="reasoning_long_generation_%d" % index,
                prompt=prompt,
                references=[str(answer)],
                task="reasoning_long_generation",
                metadata={
                    "category": "long_generation_reasoning",
                    "dataset_official": False,
                    "problem": problem,
                    "answer": str(answer),
                    "n_distractors": distractors,
                },
            )
        )
    return samples


def _longbench_config(task: str, settings: Dict[str, Any]) -> Any:
    return SimpleNamespace(
        task=task,
        dataset_name=settings.get("dataset_name"),
        dataset_config=settings.get("dataset_config"),
        split=settings.get("split", "test"),
        dataset_revision=settings.get("dataset_revision"),
        use_official_prompt=True,
        max_words=int(settings.get("max_words", 700)),
        require_official=False,
        num_samples=int(settings.get("num_samples", 1)),
        sample_strategy=str(settings.get("sample_strategy", "first")),
        sample_indices=settings.get("sample_indices"),
        data_path=None,
    )


def load_discovery_tasks(
    cfg: DiscoveryConfig,
) -> Tuple[List[BenchmarkSample], List[Dict[str, Any]]]:
    samples: List[BenchmarkSample] = []
    events: List[Dict[str, Any]] = []
    seed = int(cfg.runtime.seed)
    for task_name, raw_settings in cfg.tasks.items():
        settings: Dict[str, Any]
        if isinstance(raw_settings, int):
            settings = {"num_samples": int(raw_settings)}
        else:
            settings = dict(raw_settings or {})
        count = int(settings.get("num_samples", 1))
        if task_name == "ruler_niah":
            sample_offset = int(settings.get("sample_offset", 0))
            if sample_offset < 0:
                raise ValueError("ruler_niah sample_offset must be non-negative")
            loaded = _synthetic_niah(
                seed,
                count + sample_offset,
                int(settings.get("context_length", 384)),
            )[sample_offset : sample_offset + count]
            loaded = [_extend_retrieval_prompt(sample) for sample in loaded]
            events.append(
                {
                    "task": task_name,
                    "source": "repository_synthetic_ruler_niah",
                    "dataset_official": False,
                    "count": len(loaded),
                    "sample_offset": sample_offset,
                }
            )
        elif task_name in {"govreport_or_qmsum", "gov_report", "qmsum"}:
            preferred = str(settings.get("preferred", "gov_report"))
            try:
                loaded = LongBenchBenchmark(
                    _longbench_config(preferred, settings), seed
                ).load()[:count]
                events.append(
                    {
                        "task": task_name,
                        "source": "official_longbench_%s" % preferred,
                        "dataset_official": True,
                        "count": len(loaded),
                    }
                )
            except Exception as exc:
                loaded = _fallback_reports(
                    seed,
                    count,
                    int(settings.get("fallback_paragraphs", 18)),
                )
                for sample in loaded:
                    sample.metadata["fallback_reason"] = "%s: %s" % (
                        type(exc).__name__,
                        exc,
                    )
                events.append(
                    {
                        "task": task_name,
                        "source": "deterministic_long_report_fallback",
                        "dataset_official": False,
                        "count": len(loaded),
                        "fallback_reason": "%s: %s" % (type(exc).__name__, exc),
                    }
                )
        elif task_name in {
            "gsm8k_or_math_long_generation",
            "reasoning_long_generation",
        }:
            loaded = _reasoning_samples(
                seed,
                count,
                int(settings.get("n_distractors", 24)),
            )
            events.append(
                {
                    "task": task_name,
                    "source": "repository_template_reasoning_with_distractors",
                    "dataset_official": False,
                    "count": len(loaded),
                }
            )
        else:
            raise ValueError("unsupported discovery task: %s" % task_name)
        for sample in loaded:
            sample.metadata["requested_discovery_task"] = task_name
            samples.append(sample)
    return samples, events
