import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    # inputs / working state
    question: str
    mode: str
    query: str                       # current search query (rewritten on retries)
    loops: int                       # rewrite attempts used so far
    passages: list[dict]
    answer: str
    claims: list[dict]               # verification results of the latest draft
    score: dict                      # score of the latest draft
    history: Annotated[list[dict], operator.add]   # one entry per attempt
    # outputs (set by finalize)
    status: str
    final_answer: str
    final_score: dict
    kept_claims: list[dict]
    removed_claims: list[dict]


def initial_state(question: str, mode: str = "rerank") -> GraphState:
    return {"question": question, "mode": mode, "query": question, "loops": 0,
            "passages": [], "claims": [], "history": []}