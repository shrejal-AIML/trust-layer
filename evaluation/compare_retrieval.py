from src.retrieval.search import MODES, search

QUESTIONS = [
    "What is multi-head attention?",
    "What BLEU score did the Transformer get on WMT 2014 English-to-German?",
    "What are reflection tokens in Self-RAG?",
    "How does RAG-Token differ from RAG-Sequence?",
]


def main() -> None:
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        for mode in MODES:
            hits = search(question, mode=mode)["passages"]
            ids = ", ".join(f"{h['chunk_id']}(p{h['page']})" for h in hits)
            print(f"  {mode:<7} {ids}")


if __name__ == "__main__":
    main()