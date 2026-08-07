"""LongBench official HF dataset adapter for the 16 English tasks."""
from __future__ import annotations

from typing import Any, Dict, List

from kvbench.benchmarks.base import BenchmarkAdapter, references_from_value
from kvbench.types import BenchmarkSample


ENGLISH_TASKS = (
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
)

TASK_CATEGORIES = {
    "narrativeqa": "single_document_qa",
    "qasper": "single_document_qa",
    "multifieldqa_en": "single_document_qa",
    "hotpotqa": "multi_document_qa",
    "2wikimqa": "multi_document_qa",
    "musique": "multi_document_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "trec": "few_shot",
    "triviaqa": "few_shot",
    "samsum": "few_shot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}

PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\nStory: "
        "{context}\n\nNow, answer the question based on the story asconcisely "
        "as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nArticle: {context}\n\n Answer the question based on the "
        "above article as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no "
        "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide "
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer "
        "the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary "
        "of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of "
        "the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or "
        "instruction. Answer the query in one or more sentences.\n\nTranscript:\n"
        "{context}\n\nNow, answer the query based on the above meeting transcript "
        "in one or more sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the "
        "news.\n\nSummary:"
    ),
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may "
        "be duplicates. Please carefully read these paragraphs and determine how "
        "many unique paragraphs there are after removing duplicates. In other "
        "words, how many non-repeating paragraphs are there in total?\n\n{context}"
        "\n\nPlease enter the final count of unique paragraphs after removing "
        "duplicates. The output format should only contain the number, such as 1, "
        "2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\nThe "
        "following is an abstract.\n\n{input}\n\nPlease enter the number of the "
        "paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below.\n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}

NO_CHAT_TEMPLATE_TASKS = {
    "trec", "triviaqa", "samsum", "lcc", "repobench-p",
}


class LongBenchBenchmark(BenchmarkAdapter):
    def load(self) -> List[BenchmarkSample]:
        task = self.cfg.task
        if task not in ENGLISH_TASKS:
            raise ValueError("unsupported/non-English LongBench task: %s" % task)
        from datasets import load_dataset

        kwargs: Dict[str, Any] = {
            "path": self.cfg.dataset_name or "THUDM/LongBench",
            "name": self.cfg.dataset_config or task,
            "split": self.cfg.split or "test",
            "trust_remote_code": True,
        }
        if self.cfg.dataset_revision:
            kwargs["revision"] = self.cfg.dataset_revision
        dataset = load_dataset(**kwargs)
        samples: List[BenchmarkSample] = []
        if not self.cfg.use_official_prompt:
            raise RuntimeError(
                "LongBench paper runs require benchmark.use_official_prompt=true"
            )
        template = PROMPTS[task]
        for index, row_obj in enumerate(dataset):
            row = dict(row_obj)
            context = str(row.get("context", ""))
            question = str(row.get("input", row.get("question", "")))
            if self.cfg.max_words > 0:
                context = " ".join(context.split()[: int(self.cfg.max_words)])
            prompt = template.format(context=context, input=question)
            references = references_from_value(row.get("answers", row.get("answer")))
            if not references:
                raise RuntimeError("LongBench row has no reference at index=%d" % index)
            samples.append(
                BenchmarkSample(
                    sample_id="%s:%d" % (task, index),
                    prompt=prompt,
                    references=references,
                    task=task,
                    answer_text=references[0],
                    full_text=prompt + " " + references[0],
                    metadata={
                        "dataset_official": True,
                        "official_dataset_index": index,
                        "category": TASK_CATEGORIES[task],
                        "all_classes": row.get("all_classes"),
                        "raw_length": row.get("length"),
                        "dataset_revision": self.cfg.dataset_revision,
                        "dataset_name": self.cfg.dataset_name or "THUDM/LongBench",
                        "official_prompt": True,
                        "official_max_new_tokens": MAX_NEW_TOKENS[task],
                        "disable_chat_template": task in NO_CHAT_TEMPLATE_TASKS,
                        "require_official_metric": bool(self.cfg.require_official),
                    },
                )
            )
        selected = self.select(samples)
        if not selected:
            raise RuntimeError("LongBench dataset produced zero selected samples")
        return selected
