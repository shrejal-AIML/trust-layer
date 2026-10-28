import time
from dataclasses import dataclass

from src.config import TOP_K
from src.generation.answer import generate_answer
from src.graph.rag_graph import run as run_graph
from src.retrieval.search import search
from src.verification.pipeline import verify_answer
from src.verification.verifier import SUPPORTED


@dataclass(frozen=True)
class Config:
    name: str
    mode: str            # dense | hybrid | rerank
    verify_loop: bool    # True = full LangGraph pipeline


CONFIGS = {c.name: c for c in (
    Config("dense", "dense", False),
    Config("hybrid", "hybrid", False),
    Config("rerank", "rerank", False),
    Config("rerank_verify", "rerank", True),
)}


def _record(answer: str, passages: list[dict], n_claims: int, n_supported: int,
            latency_s: float, loops: int) -> dict:
    return {
        "answer": answer,
        "passage_ids": [p["chunk_id"] for p in passages],
        "passage_texts": [p["text"] for p in passages],
        "n_claims": n_claims,
        "n_supported": n_supported,
        "latency_s": round(latency_s, 2),
        "loops": loops,
    }


def _run_plain(question: str, mode: str) -> dict:
    """Retrieve + generate, no correction. The user would see this draft as is."""
    t0 = time.perf_counter()
    passages = search(question, mode=mode, k=TOP_K)["passages"]
    answer = generate_answer(question, passages)["answer"]
    latency = time.perf_counter() - t0

    # Measurement only (not counted in latency): how many claims in the draft are supported?
    claims = verify_answer(answer, passages)["claims"]
    n_supported = sum(c["verdict"] == SUPPORTED for c in claims)
    return _record(answer, passages, len(claims), n_supported, latency, loops=0)


def _run_verified(question: str, mode: str) -> dict:
    """Full pipeline. The user only sees claims that survived verification."""
    out = run_graph(question, mode=mode)
    shown = len(out["kept_claims"])
    return _record(out["final_answer"], out["passages"], shown, shown,
                   out["latency_s"], loops=out["loops"])


def run_question(config: Config, question: str) -> dict:
    if config.verify_loop:
        return _run_verified(question, config.mode)
    return _run_plain(question, config.mode)