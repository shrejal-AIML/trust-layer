from collections import Counter

from src.config import JUDGE_CONFIDENCE
from src.verification.verifier import CONTRADICTED, NOT_ENOUGH_INFO, SUPPORTED


def claim_confidence(result: dict) -> float:
    """0 for unsupported claims; NLI entailment prob, or a fixed value when the LLM judge approved."""
    if result["verdict"] != SUPPORTED:
        return 0.0
    return result["nli_entail"] if result["method"] == "nli" else JUDGE_CONFIDENCE


def decide(results: list[dict]) -> str:
    if not results:
        return "abstain"
    return "return" if all(r["verdict"] == SUPPORTED for r in results) else "revise"


def score_answer(results: list[dict]) -> dict:
    n = len(results)
    counts = Counter(r["verdict"] for r in results)
    return {
        "n_claims": n,
        "supported": counts[SUPPORTED],
        "contradicted": counts[CONTRADICTED],
        "not_enough_info": counts[NOT_ENOUGH_INFO],
        "faithfulness": round(counts[SUPPORTED] / n, 3) if n else 0.0,
        "confidence": round(sum(claim_confidence(r) for r in results) / n, 3) if n else 0.0,
        "decision": decide(results),
    }