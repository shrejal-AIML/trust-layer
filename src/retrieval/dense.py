from src.config import QUERY_PREFIX, TOP_K
from src.ingestion.embed_store import get_embedder, get_collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    q_emb = get_embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    res = get_collection().query(
        query_embeddings=q_emb, n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        out.append({"chunk_id": cid, "text": doc, "score": round(1 - dist, 4), **meta})
    return out