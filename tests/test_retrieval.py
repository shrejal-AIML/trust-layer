from src.retrieval.bm25 import tokenize
from src.retrieval.hybrid import rrf_fuse


def _hit(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "score": 0.0, "text": chunk_id}


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("Multi-Head Attention, BLEU 28!") == ["multi", "head", "attention", "bleu", "28"]


def test_rrf_prefers_chunks_found_by_both_retrievers():
    dense_hits = [_hit("a"), _hit("b"), _hit("c")]
    bm25_hits = [_hit("c"), _hit("d")]
    fused = rrf_fuse(dense_hits, bm25_hits, top_k=4)
    assert fused[0]["chunk_id"] == "c"
    assert fused[0]["dense_rank"] == 3 and fused[0]["bm25_rank"] == 1
    assert len(fused) == 4


def test_rrf_respects_top_k():
    assert len(rrf_fuse([_hit("a"), _hit("b")], [_hit("c")], top_k=2)) == 2