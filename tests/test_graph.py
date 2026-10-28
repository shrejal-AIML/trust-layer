from src.config import MAX_LOOPS
from src.graph import nodes
from src.graph.nodes import merge_passages, route
from src.graph.rag_graph import build_graph
from src.graph.state import initial_state
from src.verification.finalize import finalize_claims
from src.verification.scorer import score_answer


def _claim(text: str, verdict: str) -> dict:
    return {"claim": text, "cited_ids": ["doc1_chunk0"], "verdict": verdict, "method": "nli",
            "nli_entail": 0.95, "nli_contradict": 0.0, "evidence_ids": ["doc1_chunk0"], "reason": ""}


def _patch_graph(monkeypatch, claims: list[dict]) -> list[int]:
    """Replace every external call in the graph; returns a list that records verify calls."""
    passages = [{"chunk_id": "doc1_chunk0", "text": "t", "source": "a.pdf", "page": 1, "score": 1.0}]
    verify_calls: list[int] = []

    def fake_verify(answer, passages):
        verify_calls.append(1)
        return {"claims": claims, "score": score_answer(claims)}

    monkeypatch.setattr(nodes, "search", lambda q, mode, k: {"passages": passages, "candidates": []})
    monkeypatch.setattr(nodes, "generate_answer",
                        lambda q, p, avoid=None: {"answer": "draft", "cited_ids": [], "invalid_citations": []})
    monkeypatch.setattr(nodes, "rewrite_query", lambda q, unsupported, tried=None: "better query")
    monkeypatch.setattr(nodes, "verify_answer", fake_verify)
    return verify_calls


def test_merge_passages_dedupes_and_caps():
    old = [{"chunk_id": c} for c in "abcde"]
    new = [{"chunk_id": c} for c in "cfghi"]
    assert [p["chunk_id"] for p in merge_passages(old, new, limit=8)] == list("abcdefgh")


def test_route_loops_only_while_budget_remains():
    assert route({"score": {"decision": "revise"}, "loops": 0}) == "rewrite"
    assert route({"score": {"decision": "revise"}, "loops": MAX_LOOPS}) == "finalize"
    assert route({"score": {"decision": "return"}, "loops": 0}) == "finalize"
    assert route({"score": {"decision": "abstain"}, "loops": 0}) == "rewrite"
    assert route({"score": {"decision": "abstain"}, "loops": MAX_LOOPS}) == "finalize"

def test_finalize_keeps_everything_when_all_supported():
    out = finalize_claims("full answer", [_claim("a", "SUPPORTED"), _claim("b", "SUPPORTED")])
    assert out["status"] == "verified" and out["final_answer"] == "full answer"


def test_finalize_drops_unsupported_claims():
    out = finalize_claims("full answer", [_claim("good one", "SUPPORTED"), _claim("bad one", "CONTRADICTED")])
    assert out["status"] == "partial"
    assert out["final_answer"] == "good one [doc1_chunk0]."
    assert [c["claim"] for c in out["removed_claims"]] == ["bad one"]


def test_finalize_abstains_when_too_few_claims_survive():
    claims = [_claim("a", "SUPPORTED"), _claim("b", "CONTRADICTED"), _claim("c", "NOT_ENOUGH_INFO")]
    out = finalize_claims("full answer", claims)
    assert out["status"] == "abstained" and out["kept_claims"] == []


def test_finalize_abstains_when_model_abstained():
    out = finalize_claims("I can't answer this from the provided documents.", [])
    assert out["status"] == "abstained"
    assert out["final_answer"] == "I can't answer this from the provided documents."


def test_graph_loops_max_times_then_removes_bad_claims(monkeypatch):
    calls = _patch_graph(monkeypatch, [_claim("good", "SUPPORTED"), _claim("bad", "CONTRADICTED")])
    out = build_graph().invoke(initial_state("q"))
    assert len(calls) == MAX_LOOPS + 1
    assert out["loops"] == MAX_LOOPS and out["query"] == "better query"
    assert out["status"] == "partial"
    assert [c["claim"] for c in out["removed_claims"]] == ["bad"]


def test_graph_returns_immediately_when_all_supported(monkeypatch):
    calls = _patch_graph(monkeypatch, [_claim("good", "SUPPORTED")])
    out = build_graph().invoke(initial_state("q"))
    assert len(calls) == 1 and out["loops"] == 0 and out["status"] == "verified"