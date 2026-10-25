from src.config import CANDIDATES_K, POOL_K, RRF_K
from src.retrieval import bm25, dense


def rrf_fuse(dense_hits: list[dict], bm25_hits: list[dict],
             top_k: int, rrf_k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum of 1 / (rrf_k + rank) over both rankings."""
    fused: dict[str, dict] = {}
    for name, hits in (("dense", dense_hits), ("bm25", bm25_hits)):
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(
                hit["chunk_id"],
                {**hit, "score": 0.0, "dense_rank": None, "bm25_rank": None},
            )
            entry[f"{name}_rank"] = rank
            entry["score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:top_k]
    for hit in ranked:
        hit["score"] = round(hit["score"], 5)
    return ranked


def retrieve(query: str, k: int = CANDIDATES_K) -> list[dict]:
    return rrf_fuse(
        dense.retrieve(query, k=POOL_K),
        bm25.search(query, k=POOL_K),
        top_k=k,
    )