import re

from src.generation.answer import CITE_RE
from src.llm import chat, extract_json

PROMPT = """Split the answer into atomic claims.

Rules:
1. Each claim states exactly ONE fact and makes sense on its own (replace pronouns like "it" or "this" with the real subject).
2. Do not add, infer, or change any information.
3. For each claim, list the passage IDs cited for it in the answer (e.g. "doc1_chunk5"). Use [] if none.

Return ONLY a JSON array, no other text:
[{"claim": "...", "cited_ids": ["doc1_chunk5"]}]"""

_CITE_BRACKET_RE = re.compile(r"\s*\[doc\d+_chunk\d+\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_citations(text: str) -> str:
    return _CITE_BRACKET_RE.sub("", text).strip()


def _fallback(answer: str) -> list[dict]:
    """If the LLM output can't be parsed, treat each sentence as one claim."""
    claims = []
    for sentence in _SENTENCE_RE.split(answer.strip()):
        text = _strip_citations(sentence)
        if text:
            claims.append({"claim": text, "cited_ids": list(dict.fromkeys(CITE_RE.findall(sentence)))})
    return claims


def _clean(items) -> list[dict]:
    claims = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("claim"), str) and item["claim"].strip():
            ids = [i for i in item.get("cited_ids", []) if isinstance(i, str) and CITE_RE.fullmatch(i)]
            claims.append({"claim": item["claim"].strip(), "cited_ids": list(dict.fromkeys(ids))})
    return claims


def extract_claims(answer: str) -> list[dict]:
    raw = chat(
        [{"role": "system", "content": PROMPT},
         {"role": "user", "content": f"Answer:\n{answer}"}],
        max_tokens=2000,
    )
    try:
        claims = _clean(extract_json(raw))
    except ValueError:
        claims = []
    return claims or _fallback(answer)