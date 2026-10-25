import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[dict], BM25Okapi]:
    """Build the BM25 index once per process from the ingested chunks."""
    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines if line.strip()]
    return chunks, BM25Okapi([tokenize(c["text"]) for c in chunks])


def search(query: str, k: int) -> list[dict]:
    chunks, index = _load_index()
    scores = index.get_scores(tokenize(query))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**chunks[i], "score": round(float(scores[i]), 4)} for i in top if scores[i] > 0]