from src.llm import chat

PROMPT = """Write ONE short search query (under 15 words) that would retrieve passages answering the question.
Keep the key terms from the question. Use the unverified statements only as hints about what evidence is missing.
Return only the query, nothing else."""


def rewrite_query(question: str, unsupported: list[str]) -> str:
    hints = "\n".join(f"- {claim}" for claim in unsupported[:5]) or "- (none)"
    raw = chat(
        [{"role": "system", "content": PROMPT},
         {"role": "user", "content": f"Question: {question}\n\nUnverified statements:\n{hints}"}],
        max_tokens=1000,
    )
    lines = [line.strip().strip('"') for line in raw.splitlines() if line.strip()]
    return lines[0] if lines else question