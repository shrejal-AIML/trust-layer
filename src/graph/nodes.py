from src.config import MAX_CONTEXT_PASSAGES, MAX_LOOPS, TOP_K
from src.generation.answer import generate_answer
from src.generation.rewrite import rewrite_query
from src.graph.state import GraphState
from src.retrieval.search import search
from src.verification.finalize import finalize_claims
from src.verification.pipeline import verify_answer
from src.verification.verifier import SUPPORTED


def merge_passages(old: list[dict], new: list[dict], limit: int = MAX_CONTEXT_PASSAGES) -> list[dict]:
    """Keep earlier passages first, add new ones that are not duplicates, cap the total."""
    seen = {p["chunk_id"] for p in old}
    return (old + [p for p in new if p["chunk_id"] not in seen])[:limit]


def retrieve(state: GraphState) -> dict:
    result = search(state["query"], mode=state["mode"], k=TOP_K)
    return {"passages": merge_passages(state["passages"], result["passages"])}


def generate(state: GraphState) -> dict:
    avoid = [c["claim"] for c in state["claims"] if c["verdict"] != SUPPORTED]
    gen = generate_answer(state["question"], state["passages"], avoid=avoid)
    return {"answer": gen["answer"]}


def verify(state: GraphState) -> dict:
    result = verify_answer(state["answer"], state["passages"])
    score = result["score"]
    entry = {"attempt": state["loops"], "query": state["query"], "decision": score["decision"],
             "faithfulness": score["faithfulness"], "confidence": score["confidence"]}
    return {"claims": result["claims"], "score": score, "history": [entry]}


def rewrite(state: GraphState) -> dict:
    unsupported = [c["claim"] for c in state["claims"] if c["verdict"] != SUPPORTED]
    return {"query": rewrite_query(state["question"], unsupported), "loops": state["loops"] + 1}


def route(state: GraphState) -> str:
    """Loop only when the draft has unsupported claims AND loop budget remains."""
    if state["score"]["decision"] == "revise" and state["loops"] < MAX_LOOPS:
        return "rewrite"
    return "finalize"


def finalize(state: GraphState) -> dict:
    return finalize_claims(state["answer"], state["claims"])