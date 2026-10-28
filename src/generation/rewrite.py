from src.llm import chat

PROMPT = """Write ONE short search query (under 15 words) that would retrieve passages answering the question.
Keep the key terms from the question. If unverified statements are given, use them as hints about what evidence is missing.
Queries already tried found nothing useful. Your query must be clearly different from all of them: use different key words, not just a reordering.
Research papers often say "generator", "backbone", "trained on", "implementation details", "we use" instead of the words in the question.
Return only the query, nothing else."""


def rewrite_query(question: str, unsupported: list[str], tried: list[str] | None = None) -> str:
    hints = "\n".join(f"- {claim}" for claim in unsupported[:5]) or "- (none)"
    already = "\n".join(f"- {q}" for q in (tried or [])) or "- (none)"
    raw = chat(
        [{"role": "system", "content": PROMPT},
         {"role": "user", "content": (f"Question: {question}\n\nUnverified statements:\n{hints}"
                                      f"\n\nQueries already tried:\n{already}")}],
        max_tokens=1000,
    )
    lines = [line.strip().strip('"') for line in raw.splitlines() if line.strip()]
    return lines[0] if lines else question