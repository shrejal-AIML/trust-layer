import argparse
import time
import uuid

from src.config import EMBED_MODEL, LLM_MODEL, RERANK_MODEL, TOP_K
from src.generation.answer import generate_answer
from src.logging_utils import log_query
from src.retrieval.search import MODES, search


def _summary(hit: dict) -> dict:
    return {
        "chunk_id": hit["chunk_id"],
        "score": hit["score"],
        "dense_rank": hit.get("dense_rank"),
        "bm25_rank": hit.get("bm25_rank"),
    }


def ask(question: str, mode: str = "rerank", k: int = TOP_K) -> dict:
    query_id = uuid.uuid4().hex[:8]

    t0 = time.perf_counter()
    result = search(question, mode=mode, k=k)
    t1 = time.perf_counter()
    passages = result["passages"]
    gen = generate_answer(question, passages)
    t2 = time.perf_counter()

    record = {
        "query_id": query_id,
        "question": question,
        "pipeline": f"{mode}_v2",
        "retrieval_mode": mode,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL if mode == "rerank" else None,
        "llm_model": LLM_MODEL,
        "k": k,
        "candidates": [_summary(c) for c in result["candidates"]],
        "retrieved": [
            {**_summary(p), "source": p["source"], "page": p["page"], "text": p["text"]}
            for p in passages
        ],
        "answer": gen["answer"],
        "cited_ids": gen["cited_ids"],
        "invalid_citations": gen["invalid_citations"],
        "latency_retrieval_s": round(t1 - t0, 3),
        "latency_generation_s": round(t2 - t1, 3),
    }
    log_query(record)
    return record


def _rank(value) -> str:
    return f"#{value}" if value else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    args = parser.parse_args()

    r = ask(" ".join(args.question), mode=args.mode)
    print(f"\n=== ANSWER ({args.mode}) ===\n{r['answer']}")
    print("\n=== CITED ===", r["cited_ids"])
    if r["invalid_citations"]:
        print("!! INVALID CITATIONS:", r["invalid_citations"])
    print("\n=== RETRIEVED ===")
    for p in r["retrieved"]:
        print(f"[{p['chunk_id']}] score={p['score']} dense{_rank(p['dense_rank'])} "
              f"bm25{_rank(p['bm25_rank'])} ({p['source']}, p.{p['page']})")
    print(f"\nlatency: retrieval={r['latency_retrieval_s']}s generation={r['latency_generation_s']}s")


if __name__ == "__main__":
    main()