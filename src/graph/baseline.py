import sys
import time
import uuid
from src.config import LLM_MODEL, EMBED_MODEL, TOP_K
from src.retrieval.dense import retrieve
from src.generation.answer import generate_answer
from src.logging_utils import log_query


def ask(question: str, k: int = TOP_K) -> dict:
    query_id = uuid.uuid4().hex[:8]

    t0 = time.perf_counter()
    passages = retrieve(question, k=k)
    t1 = time.perf_counter()
    gen = generate_answer(question, passages)
    t2 = time.perf_counter()

    record = {
        "query_id": query_id,
        "question": question,
        "pipeline": "baseline_v1",
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "k": k,
        "retrieved": [{"chunk_id": p["chunk_id"], "score": p["score"],
                       "source": p["source"], "page": p["page"], "text": p["text"]}
                      for p in passages],
        "answer": gen["answer"],
        "cited_ids": gen["cited_ids"],
        "invalid_citations": gen["invalid_citations"],
        "latency_retrieval_s": round(t1 - t0, 3),
        "latency_generation_s": round(t2 - t1, 3),
    }
    log_query(record)
    return record


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    r = ask(q)
    print("\n=== ANSWER ===\n" + r["answer"])
    print("\n=== CITED ===", r["cited_ids"])
    if r["invalid_citations"]:
        print("!! INVALID CITATIONS:", r["invalid_citations"])
    print("\n=== RETRIEVED ===")
    for p in r["retrieved"]:
        print(f"[{p['chunk_id']}] score={p['score']} ({p['source']}, p.{p['page']})")