import re
from src.llm import chat

SYSTEM_PROMPT = """You answer questions using ONLY the numbered passages provided.

Rules:
1. Every factual sentence must end with the ID(s) of the passage(s) that support it, written exactly like [doc3_chunk12]. Use one bracket per ID, e.g. [doc1_chunk2] [doc3_chunk4].
2. Never cite an ID that is not in the provided passages.
3. Do not use outside knowledge. If the passages do not contain the answer, reply exactly: I can't answer this from the provided documents.
4. Be concise and precise."""

CITE_RE = re.compile(r"doc\d+_chunk\d+")


def build_context(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['chunk_id']}]\n{p['text']}" for p in passages)


def parse_citations(answer: str, passages: list[dict]) -> tuple[list[str], list[str]]:
    valid_ids = {p["chunk_id"] for p in passages}
    found = list(dict.fromkeys(CITE_RE.findall(answer)))   # unique, keep order
    cited = [c for c in found if c in valid_ids]
    invalid = [c for c in found if c not in valid_ids]
    return cited, invalid


def generate_answer(question: str, passages: list[dict]) -> dict:
    user_msg = f"Passages:\n\n{build_context(passages)}\n\nQuestion: {question}\n\nAnswer:"
    answer = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    cited, invalid = parse_citations(answer, passages)
    return {"answer": answer, "cited_ids": cited, "invalid_citations": invalid}