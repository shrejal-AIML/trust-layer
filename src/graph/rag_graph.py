import argparse
import time
import uuid
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.config import EMBED_MODEL, LLM_MODEL
from src.graph import nodes
from src.graph.state import GraphState, initial_state
from src.logging_utils import log_query
from src.retrieval.search import MODES
from src.verification.scorer import claim_confidence


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", nodes.route, {"rewrite": "rewrite", "finalize": "finalize"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("finalize", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def run(question: str, mode: str = "rerank") -> dict:
    query_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    out = get_graph().invoke(initial_state(question, mode))
    out["query_id"] = query_id
    out["latency_s"] = round(time.perf_counter() - t0, 2)

    log_query({
        "type": "graph_run", "query_id": query_id, "question": question, "mode": mode,
        "embed_model": EMBED_MODEL, "llm_model": LLM_MODEL,
        "status": out["status"], "loops": out["loops"],
        "final_answer": out["final_answer"], "final_score": out["final_score"],
        "kept_claims": out["kept_claims"], "removed_claims": out["removed_claims"],
        "history": out["history"], "passage_ids": [p["chunk_id"] for p in out["passages"]],
        "latency_s": out["latency_s"],
    })
    return out


def print_result(out: dict) -> None:
    score = out["final_score"]
    print(f"\n=== FINAL ANSWER [{out['status']}] ===\n{out['final_answer']}")
    print(f"\n=== CONFIDENCE === {score['confidence']} "
          f"(claims kept: {len(out['kept_claims'])}, removed: {len(out['removed_claims'])}, "
          f"loops used: {out['loops']})")

    print("\n=== EVIDENCE PER CLAIM ===")
    for i, c in enumerate(out["kept_claims"], start=1):
        print(f"{i}. {c['claim']}")
        print(f"   evidence={c['evidence_ids']} via {c['method']} confidence={claim_confidence(c)}")

    if out["removed_claims"]:
        print("\n=== REMOVED (could not verify) ===")
        for c in out["removed_claims"]:
            print(f"- [{c['verdict']}] {c['claim']}\n  reason: {c['reason']}")

    print("\n=== ATTEMPTS ===")
    for h in out["history"]:
        print(f"#{h['attempt']} query={h['query']!r} decision={h['decision']} "
              f"faithfulness={h['faithfulness']}")
    print(f"\nlatency: {out['latency_s']}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question through the verified RAG graph.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    args = parser.parse_args()
    print_result(run(" ".join(args.question), mode=args.mode))


if __name__ == "__main__":
    main()