from src.verification.claim_extractor import extract_claims
from src.verification.scorer import score_answer
from src.verification.verifier import verify_claim


def is_abstention(answer: str) -> bool:
    text = answer.lower().replace("\u2019", "'")
    return "can't answer this from the provided documents" in text


def verify_answer(answer: str, passages: list[dict]) -> dict:
    """answer -> atomic claims -> per-claim verdicts -> answer-level score and decision."""
    if is_abstention(answer):
        return {"claims": [], "score": score_answer([])}
    results = [verify_claim(c, passages) for c in extract_claims(answer)]
    return {"claims": results, "score": score_answer(results)}