from sentence_transformers import CrossEncoder

from src.config import RERANK_MODEL, TOP_K

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL, max_length=512)
    return _model


def rerank(query: str, candidates: list[dict], k: int = TOP_K) -> list[dict]:
    """Score every (query, passage) pair jointly and keep the best k."""
    if not candidates:
        return []
    scores = get_reranker().predict([(query, c["text"]) for c in candidates], batch_size=16)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:k]
    return [
        {**c, "hybrid_score": c["score"], "score": round(float(s), 4)}
        for c, s in ranked
    ]