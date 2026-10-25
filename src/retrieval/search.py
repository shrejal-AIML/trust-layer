from src.config import CANDIDATES_K, TOP_K
from src.retrieval import dense, hybrid, reranker

MODES = ("dense", "hybrid", "rerank")


def search(query: str, mode: str = "rerank", k: int = TOP_K) -> dict:
    """Return {"passages": final top-k, "candidates": pre-rerank list (empty for dense)}."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "dense":
        return {"passages": dense.retrieve(query, k=k), "candidates": []}

    candidates = hybrid.retrieve(query, k=CANDIDATES_K)
    if mode == "hybrid":
        return {"passages": candidates[:k], "candidates": candidates}
    return {"passages": reranker.rerank(query, candidates, k=k), "candidates": candidates}