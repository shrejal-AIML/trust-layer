from src.config import MIN_KEPT_FRACTION
from src.verification.scorer import score_answer
from src.verification.verifier import SUPPORTED

VERIFIED = "verified"
PARTIAL = "partial"
ABSTAINED = "abstained"

ABSTAIN_MESSAGE = "I can't verify this from the provided documents."


def _claim_to_sentence(result: dict) -> str:
    ids = result.get("evidence_ids") or result.get("cited_ids") or []
    cites = "".join(f"[{i}]" for i in ids)
    return f"{result['claim'].rstrip('.')} {cites}.".replace(" .", ".")


def finalize_claims(answer: str, claims: list[dict], min_kept: float = MIN_KEPT_FRACTION) -> dict:
    """Decide what the user finally sees: the full answer, only the verified claims, or an abstention."""
    kept = [c for c in claims if c["verdict"] == SUPPORTED]
    removed = [c for c in claims if c["verdict"] != SUPPORTED]

    if not claims:   # the model itself said it can't answer
        return {"status": ABSTAINED, "final_answer": answer or ABSTAIN_MESSAGE,
                "kept_claims": [], "removed_claims": [], "final_score": score_answer([])}

    if not removed:
        return {"status": VERIFIED, "final_answer": answer,
                "kept_claims": kept, "removed_claims": [], "final_score": score_answer(kept)}

    if len(kept) / len(claims) >= min_kept:
        return {"status": PARTIAL, "final_answer": " ".join(_claim_to_sentence(c) for c in kept),
                "kept_claims": kept, "removed_claims": removed, "final_score": score_answer(kept)}

    return {"status": ABSTAINED, "final_answer": ABSTAIN_MESSAGE,
            "kept_claims": [], "removed_claims": claims, "final_score": score_answer([])}