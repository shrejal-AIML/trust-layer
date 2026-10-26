import argparse

from src.graph.pipeline import ask
from src.logging_utils import log_query
from src.retrieval.search import MODES
from src.verification.pipeline import verify_answer


def print_report(verification: dict) -> None:
    print("\n=== CLAIMS ===")
    for i, c in enumerate(verification["claims"], start=1):
        print(f"{i}. [{c['verdict']}] via {c['method']} "
              f"(nli_entail={c['nli_entail']}, nli_contradict={c['nli_contradict']})")
        print(f"   claim:    {c['claim']}")
        print(f"   evidence: {c['evidence_ids']}  {c['reason']}")
    s = verification["score"]
    print(f"\n=== SCORE === decision={s['decision']} confidence={s['confidence']} "
          f"faithfulness={s['faithfulness']} "
          f"(supported={s['supported']}, contradicted={s['contradicted']}, "
          f"not_enough_info={s['not_enough_info']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question, then verify the answer.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    args = parser.parse_args()

    record = ask(" ".join(args.question), mode=args.mode)
    verification = verify_answer(record["answer"], record["retrieved"])
    log_query({"type": "verification", "query_id": record["query_id"], **verification})

    print(f"\n=== ANSWER ===\n{record['answer']}")
    print_report(verification)


if __name__ == "__main__":
    main()