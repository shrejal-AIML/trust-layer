import re
import unicodedata
from statistics import mean

from src.generation.answer import CITE_RE

ABSTAIN_MARKERS = ("can't answer this", "can't verify this")
_STRIP_RE = re.compile(r"[^0-9a-z.]")


def normalize(text: str) -> str:
    """Lowercase and drop everything except letters, digits and '.', so hyphen styles and spacing don't matter."""
    return _STRIP_RE.sub("", unicodedata.normalize("NFKC", text).lower())


def contains_all(text: str, keywords: list[str]) -> bool:
    haystack = normalize(text)
    return all(normalize(k) in haystack for k in keywords)


def is_abstained(answer: str) -> bool:
    haystack = normalize(answer)
    return any(normalize(marker) in haystack for marker in ABSTAIN_MARKERS)

def context_hit(question: dict, passage_texts: list[str]) -> bool:
    """Strict when the question has an `evidence` field: one passage must contain all of it."""
    needles = question.get("evidence")
    if needles:
        return any(contains_all(text, needles) for text in passage_texts)
    return contains_all(" ".join(passage_texts), question["must_include"])


def score_question(question: dict, record: dict) -> dict:
    """Score one system output against one gold question."""
    answer = record["answer"]
    abstained = is_abstained(answer)
    answerable = question["answerable"]
    cited = list(dict.fromkeys(CITE_RE.findall(answer)))
    retrieved = set(record["passage_ids"])
    n_claims, n_supported = record["n_claims"], record["n_supported"]

    if answerable:
        correct = not abstained and contains_all(answer, question["must_include"])
        hit = context_hit(question, record["passage_texts"])
    else:
        correct, hit = abstained, None

    return {
        "id": question["id"],
        "answerable": answerable,
        "abstained": abstained,
        "correct": correct,
        "context_hit": hit,
        "cited_any": bool(cited),
        "citation_valid": sum(c in retrieved for c in cited) / len(cited) if cited else None,
        "faithfulness": n_supported / n_claims if n_claims else None,
        "unsupported_shown": n_claims - n_supported,
        "latency_s": record["latency_s"],
    }


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(mean(values), 3) if values else None


def aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"]]
    return {
        "accuracy": _mean(r["correct"] for r in rows),
        "answer_accuracy": _mean(r["correct"] for r in answerable),
        "abstain_recall": _mean(r["abstained"] for r in unanswerable),
        "false_abstain_rate": _mean(r["abstained"] for r in answerable),
        "context_hit_rate": _mean(r["context_hit"] for r in answerable),
        "faithfulness": _mean(r["faithfulness"] for r in answered),
        "unsupported_per_answer": _mean(r["unsupported_shown"] for r in answered),
        "citation_valid_rate": _mean(r["citation_valid"] for r in answered),
        "citation_coverage": _mean(r["cited_any"] for r in answered),
        "latency_mean_s": _mean(r["latency_s"] for r in rows),
    }