from src.graph.verified import print_report
from src.retrieval.search import search
from src.verification.pipeline import verify_answer

QUESTION = "What BLEU score did the Transformer get on WMT 2014 English-to-German?"
ANSWER = (
    "The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German task [doc1_chunk0]. "
    "It was trained for 3.5 days on eight GPUs for the English-to-French task [doc1_chunk0]. "
    "It also reached 35.0 BLEU on English-to-German, beating all earlier models by 10 points [doc1_chunk0]."
)


def main() -> None:
    passages = search(QUESTION, mode="rerank")["passages"]
    print_report(verify_answer(ANSWER, passages))


if __name__ == "__main__":
    main()