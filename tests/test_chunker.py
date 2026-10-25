from src.ingestion.chunker import chunk_document


def test_chunk_sizes_and_ids():
    chunks = chunk_document(["word " * 1200], "doc1", "x.pdf")
    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "doc1_chunk0"
    assert all(c["n_tokens"] <= 500 for c in chunks)