from fastapi.testclient import TestClient

from src.api import main

client = TestClient(main.app)    # not used as a context manager, so no model warmup

FAKE_OUT = {
    "query_id": "abc123",
    "status": "verified",
    "final_answer": "Multi-head attention uses h heads [doc1_chunk0].",
    "final_score": {"confidence": 0.95},
    "loops": 0,
    "latency_s": 1.2,
    "kept_claims": [{
        "claim": "Multi-head attention uses h heads.", "verdict": "SUPPORTED", "method": "nli",
        "nli_entail": 0.95, "nli_contradict": 0.0, "evidence_ids": ["doc1_chunk0"],
        "cited_ids": ["doc1_chunk0"], "reason": "NLI entailment",
    }],
    "removed_claims": [],
    "history": [{"attempt": 0, "query": "q", "decision": "return", "faithfulness": 1.0, "confidence": 0.95}],
    "passages": [
        {"chunk_id": "doc1_chunk0", "source": "attention.pdf", "page": 1, "text": "evidence text"},
        {"chunk_id": "doc1_chunk9", "source": "attention.pdf", "page": 5, "text": "unused passage"},
    ],
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_claims_and_only_cited_sources(monkeypatch):
    monkeypatch.setattr(main, "run", lambda question, mode: FAKE_OUT)
    response = client.post("/ask", json={"question": "What is multi-head attention?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified" and body["confidence"] == 0.95
    assert body["kept_claims"][0]["confidence"] == 0.95
    assert [s["chunk_id"] for s in body["sources"]] == ["doc1_chunk0"]


def test_rejects_too_short_question():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_rejects_unknown_mode():
    response = client.post("/ask", json={"question": "What is attention?", "mode": "magic"})
    assert response.status_code == 422


def test_pipeline_failure_returns_502(monkeypatch):
    def boom(question, mode):
        raise RuntimeError("llm down")

    monkeypatch.setattr(main, "run", boom)
    assert client.post("/ask", json={"question": "What is attention?"}).status_code == 502