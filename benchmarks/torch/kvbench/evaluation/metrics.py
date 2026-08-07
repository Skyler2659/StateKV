"""Audited Python ports of public RULER, LongBench, and SCBench metrics."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


_SCBENCH_ROUGE_SCORER = None


def normalize_answer(text: str) -> str:
    lowered = (text or "").lower()
    lowered = "".join(char for char in lowered if char not in set(string.punctuation))
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    return " ".join(lowered.split())


def qa_f1(prediction: str, reference: str) -> float:
    left = normalize_answer(prediction).split()
    right = normalize_answer(reference).split()
    if not left or not right:
        return float(left == right)
    common = Counter(left) & Counter(right)
    count = sum(common.values())
    if count == 0:
        return 0.0
    precision = count / len(left)
    recall = count / len(right)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    left, right = (prediction or "").split(), (reference or "").split()
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, target in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if token == target
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(left), lcs / len(right)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _max_score(function, prediction: str, references: Iterable[str]) -> float:
    values = [function(prediction, str(reference)) for reference in references]
    return max(values) if values else 0.0


def _rouge_official(prediction: str, reference: str) -> float:
    try:
        from rouge import Rouge
    except ImportError as exc:
        raise RuntimeError(
            "official LongBench ROUGE requires the pinned rouge==1.0.1 dependency"
        ) from exc
    try:
        return float(
            Rouge().get_scores([prediction], [reference], avg=True)["rouge-l"]["f"]
        )
    except Exception:
        return 0.0


def _classification_score(
    prediction: str, reference: str, all_classes: Iterable[str]
) -> float:
    matches = [name for name in all_classes if str(name) in prediction]
    matches = [
        name
        for name in matches
        if not (str(name) in reference and str(name) != reference)
    ]
    return 1.0 / len(matches) if reference in matches else 0.0


def _count_score(prediction: str, reference: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(reference) for number in numbers) / len(numbers)


def _retrieval_score(prediction: str, reference: str) -> float:
    target = re.findall(r"Paragraph (\d+)", reference)
    if not target:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(number == target[0] for number in numbers) / len(numbers)


def _code_similarity(prediction: str, reference: str) -> float:
    try:
        from fuzzywuzzy import fuzz
    except ImportError as exc:
        raise RuntimeError(
            "official LongBench code similarity requires fuzzywuzzy==0.18.0"
        ) from exc
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return float(fuzz.ratio(candidate, reference)) / 100.0


def _as_label_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_zh_answer(text: str) -> str:
    chinese_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［"
        + chr(0xFF3C)
        + "］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』〖〗〔〕〘〙〚〛〜〝〞〟〰〾〿"
        "–—‘’‛“”„‟…‧﹏."
    )
    punctuation = set(string.punctuation + chinese_punctuation)
    return "".join(
        char
        for char in (text or "").lower()
        if char not in punctuation and not char.isspace()
    )


def _sequence_f1(prediction: Iterable[Any], reference: Iterable[Any]) -> float:
    left, right = list(prediction), list(reference)
    if not left or not right:
        return float(left == right)
    count = sum((Counter(left) & Counter(right)).values())
    if count == 0:
        return 0.0
    precision = count / len(left)
    recall = count / len(right)
    return 2.0 * precision * recall / (precision + recall)


def _scbench_choice_score(prediction: str, label: Any) -> float:
    labels = [str(item) for item in _as_label_list(label)]
    pred = prediction.strip()
    if not pred:
        return 0.0
    if pred[0] in "ABCD":
        return float(pred[0] in labels)
    if pred in labels:
        return 1.0
    cleaned = pred
    for char in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        cleaned = cleaned.replace(char, " ")
    cleaned = " ".join(cleaned.split())
    for prefix in ["answer is:", "answer:", "answer is", "option is"]:
        index = cleaned.find(prefix)
        if index >= 0:
            suffix = cleaned[index + len(prefix) :].lstrip()
            return float(any(suffix.startswith(item) for item in labels))
    for word in cleaned.split():
        if word in "ABCD":
            return float(word in labels)
    return 0.0


def _scbench_summary_score(prediction: str, label: Any) -> float:
    # The current official scorer uses evaluate.load("rouge") and rougeLsum.
    # Load lazily so retrieval-only cluster runs have no metric-download side effect.
    global _SCBENCH_ROUGE_SCORER
    try:
        import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "official SCBench summary scoring requires evaluate==0.4.3 and "
            "rouge-score==0.1.2"
        ) from exc
    if _SCBENCH_ROUGE_SCORER is None:
        _SCBENCH_ROUGE_SCORER = evaluate.load("rouge")
    reference = str(_as_label_list(label)[0])
    values = _SCBENCH_ROUGE_SCORER.compute(
        predictions=[prediction],
        references=[reference],
        use_aggregator=False,
    )
    return float(values["rougeLsum"][0])


def _scbench_score(task: str, prediction: str, label: Any) -> Dict[str, Any]:
    if task in {
        "scbench_kv",
        "scbench_kv_hard",
        "scbench_hashhop",
        "scbench_prefix_suffix",
        "scbench_kv_compressible",
    }:
        raw = float(str(label) in prediction)
        metric = "kv_retrieval"
    elif task == "scbench_passkey":
        target = _as_label_list(label)[0]
        match = re.search(r"\d+", prediction)
        raw = float(match is not None and match.group(0) == str(target))
        metric = "passkey_first_integer"
    elif task == "scbench_mf":
        target = _as_label_list(label)[0]
        match = re.search(r"\d+\.\d+|\d+", prediction)
        if not isinstance(target, (int, float)) or isinstance(target, bool):
            raise TypeError("official SCBench Math.Find label must be int or float")
        if match is None:
            raw = 0.0
        elif isinstance(target, int):
            raw = float(int(float(match.group(0))) == target)
        else:
            raw = float(float(match.group(0)) == target)
        metric = "math_find_first_number"
    elif task in {"scbench_qa_eng", "scbench_many_shot"}:
        labels = [str(item) for item in _as_label_list(label)]
        upper = prediction.strip().upper()
        raw = float(any(item.upper() in upper for item in labels))
        metric = "longdialogue_qa_eng"
    elif task == "scbench_choice_eng":
        raw = _scbench_choice_score(prediction, label)
        metric = "longbook_choice_eng"
    elif task == "scbench_qa_chn":
        normalized = list(_normalize_zh_answer(prediction))
        raw = max(
            (
                _sequence_f1(normalized, list(_normalize_zh_answer(str(item))))
                for item in _as_label_list(label)
            ),
            default=0.0,
        )
        metric = "longbook_qa_chn_character_f1"
    elif task == "scbench_vt":
        labels = [str(item) for item in _as_label_list(label)]
        raw = round(
            sum(item.lower() in prediction.lower() for item in labels)
            / max(1, len(labels)),
            2,
        )
        metric = "string_match_all"
    elif task == "scbench_summary":
        raw = _scbench_summary_score(prediction, label)
        metric = "rouge_lsum"
    elif task in {
        "scbench_repoqa",
        "scbench_repoqa_and_kv",
        "scbench_summary_with_needles",
    }:
        raise RuntimeError(
            "SCBench task=%s requires an official special/multi-task scorer and is "
            "not eligible for this scalar evaluator" % task
        )
    else:
        raise RuntimeError("SCBench task=%s has no audited official scorer" % task)
    return {
        "score": round(100.0 * raw, 4),
        "metric_name": metric,
        "metric_implementation": "scbench_official_python_port",
        "correct": None if task == "scbench_summary" else bool(raw >= 0.999999),
    }


def _ruler_family(task: str) -> str:
    if task.startswith("niah"):
        return "niah"
    if task in {"vt", "variable_tracking"}:
        return "variable_tracking"
    if task in {"qa", "qa_1", "qa_2"}:
        return "qa"
    if task in {"cwe", "common_words", "common_words_extraction"}:
        return "common_words_extraction"
    if task in {"fwe", "freq_words", "freq_words_extraction"}:
        return "freq_words_extraction"
    return task


def evaluate_prediction(
    benchmark: str,
    task: str,
    prediction: str,
    references: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    benchmark = benchmark.lower()
    task = task.lower()
    metadata = metadata or {}
    if benchmark == "ruler":
        lowered = prediction.lower()
        hits = [str(reference).lower() in lowered for reference in references]
        part = _ruler_family(task) == "qa"
        score = 100.0 * (float(any(hits)) if part else sum(hits) / max(1, len(hits)))
        return {
            "score": score,
            "metric_name": "string_match_part" if part else "string_match_all",
            "metric_implementation": "ruler_public_string_match",
            "correct": bool(any(hits) if part else hits and all(hits)),
        }

    if benchmark == "longbench":
        if task in {"gov_report", "qmsum", "multi_news", "samsum"}:
            raw = _max_score(_rouge_official, prediction, references)
            metric = "rouge_l"
        elif task == "trec":
            labels = [str(item) for item in metadata.get("all_classes") or []]
            raw = max(
                (_classification_score(prediction, reference, labels) for reference in references),
                default=0.0,
            )
            metric = "classification"
        elif task == "passage_count":
            raw = _max_score(_count_score, prediction, references)
            metric = "count"
        elif task == "passage_retrieval_en":
            raw = _max_score(_retrieval_score, prediction, references)
            metric = "retrieval"
        elif task in {"lcc", "repobench-p"}:
            raw = _max_score(_code_similarity, prediction, references)
            metric = "code_similarity"
        else:
            if task in {"triviaqa"}:
                prediction = prediction.lstrip("\n").split("\n")[0]
            raw = _max_score(qa_f1, prediction, references)
            metric = "qa_f1"
        return {
            "score": round(100.0 * raw, 4),
            "metric_name": metric,
            "metric_implementation": "longbench_official_python_port",
            "correct": None,
        }

    if benchmark == "scbench":
        if metadata.get("scbench_schema") == "official_context_multi_turns":
            query_task = str(metadata.get("query_task") or task).lower()
            if "query_label" not in metadata:
                raise RuntimeError("official SCBench evaluation requires query_label metadata")
            return _scbench_score(query_task, prediction, metadata["query_label"])
        raw = _max_score(qa_f1, prediction, references)
        return {
            "score": round(100.0 * raw, 4),
            "metric_name": "qa_f1",
            "metric_implementation": "scbench_normalized_smoke_qa_f1",
            "correct": bool(raw >= 0.999999),
        }
    return {"score": None, "metric_name": None, "metric_implementation": None, "correct": None}
