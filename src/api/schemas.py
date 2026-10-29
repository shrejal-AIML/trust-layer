from typing import Literal

from pydantic import BaseModel, Field

from src.verification.scorer import claim_confidence


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["dense", "hybrid", "rerank"] = "rerank"


class ClaimOut(BaseModel):
    claim: str
    verdict: str
    method: str
    confidence: float
    evidence_ids: list[str]
    reason: str


class SourceOut(BaseModel):
    chunk_id: str
    source: str
    page: int
    text: str


class AttemptOut(BaseModel):
    attempt: int
    query: str
    decision: str
    faithfulness: float
    confidence: float


class AskResponse(BaseModel):
    query_id: str
    status: str                      # verified | partial | abstained
    answer: str
    confidence: float
    loops: int
    latency_s: float
    kept_claims: list[ClaimOut]
    removed_claims: list[ClaimOut]
    sources: list[SourceOut]         # only passages that back a kept claim
    attempts: list[AttemptOut]


def _claim(result: dict) -> ClaimOut:
    return ClaimOut(
        claim=result["claim"],
        verdict=result["verdict"],
        method=result["method"],
        confidence=claim_confidence(result),
        evidence_ids=[i for i in result.get("evidence_ids", []) if i],
        reason=result.get("reason", ""),
    )


def build_response(out: dict) -> AskResponse:
    """Turn the graph's final state into the public API response."""
    kept = [_claim(c) for c in out["kept_claims"]]
    cited = list(dict.fromkeys(i for c in kept for i in c.evidence_ids))
    by_id = {p["chunk_id"]: p for p in out["passages"]}
    sources = [SourceOut(chunk_id=i, source=by_id[i]["source"], page=by_id[i]["page"],
                         text=by_id[i]["text"]) for i in cited if i in by_id]
    return AskResponse(
        query_id=out["query_id"],
        status=out["status"],
        answer=out["final_answer"],
        confidence=out["final_score"]["confidence"],
        loops=out["loops"],
        latency_s=out["latency_s"],
        kept_claims=kept,
        removed_claims=[_claim(c) for c in out["removed_claims"]],
        sources=sources,
        attempts=[AttemptOut(**h) for h in out["history"]],
    )