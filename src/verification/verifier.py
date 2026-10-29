import re

from src.config import (NLI_ENABLED, NLI_ENTAIL_THRESHOLD, NLI_MAX_LENGTH, NLI_MAX_WINDOWS,
                        NLI_MODEL, NLI_WINDOW_SENTENCES, NLI_WINDOW_STRIDE)
from src.generation.answer import build_context
from src.llm import chat, extract_json
from src.retrieval.bm25 import tokenize

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
VERDICTS = (SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO)

JUDGE_PROMPT = """You are a strict fact-checker. Decide whether the CLAIM is supported by the EVIDENCE passages, using ONLY the evidence (no outside knowledge).

- SUPPORTED: the evidence states or directly implies the whole claim.
- CONTRADICTED: the evidence states something incompatible with the claim.
- NOT_ENOUGH_INFO: the evidence does not settle the claim, or supports only part of it.

Return ONLY JSON:
{"verdict": "SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO", "evidence_ids": ["doc1_chunk5"], "reason": "one short sentence"}"""

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_nli = None


def make_windows(chunk_id: str, text: str,
                 size: int = NLI_WINDOW_SENTENCES, stride: int = NLI_WINDOW_STRIDE) -> list[tuple[str, str]]:
    """Split a passage into overlapping sentence windows so NLI never truncates evidence."""
    sentences = [s for s in _SENTENCE_RE.split(text.strip()) if s]
    windows = []
    for start in range(0, len(sentences), stride):
        windows.append((chunk_id, " ".join(sentences[start:start + size])))
        if start + size >= len(sentences):
            break
    return windows


def _get_nli():
    global _nli
    if _nli is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        labels = {i: name.lower() for i, name in model.config.id2label.items()}
        _nli = (tokenizer, model, labels)
    return _nli

def _top_windows(claim: str, windows: list[tuple[str, str]],
                 limit: int = NLI_MAX_WINDOWS) -> list[tuple[str, str]]:
    """Keep the windows sharing the most words with the claim; NLI cost no longer scales with evidence size."""
    claim_terms = set(tokenize(claim))
    ranked = sorted(windows, key=lambda w: len(claim_terms & set(tokenize(w[1]))), reverse=True)
    return ranked[:limit]


def nli_scores(claim: str, evidence: list[dict], batch_size: int = 16) -> dict:
    """Best entailment / contradiction probability over all evidence windows."""
    import torch

    tokenizer, model, labels = _get_nli()
    windows = _top_windows(claim, [w for p in evidence for w in make_windows(p["chunk_id"], p["text"])])
    best = {"entail": 0.0, "entail_id": None, "contradict": 0.0}
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        enc = tokenizer([w[1] for w in batch], [claim] * len(batch), truncation="only_first",
                                                max_length=NLI_MAX_LENGTH, padding=True, return_tensors="pt")
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        for (chunk_id, _), row in zip(batch, probs):
            p = {labels[j]: float(row[j]) for j in range(len(row))}
            if p["entailment"] > best["entail"]:
                best["entail"], best["entail_id"] = p["entailment"], chunk_id
            best["contradict"] = max(best["contradict"], p["contradiction"])
    return best


def llm_judge(claim: str, evidence: list[dict]) -> dict:
    raw = chat(
        [{"role": "system", "content": JUDGE_PROMPT},
         {"role": "user", "content": f"EVIDENCE:\n\n{build_context(evidence)}\n\nCLAIM: {claim}"}],
        max_tokens=2000,
    )
    valid_ids = {p["chunk_id"] for p in evidence}
    try:
        out = extract_json(raw)
        verdict = out.get("verdict") if out.get("verdict") in VERDICTS else NOT_ENOUGH_INFO
        ids = [i for i in out.get("evidence_ids", []) if i in valid_ids]
        return {"verdict": verdict, "evidence_ids": ids, "reason": str(out.get("reason", ""))}
    except (ValueError, AttributeError):
        return {"verdict": NOT_ENOUGH_INFO, "evidence_ids": [], "reason": "judge output unparseable"}


def verify_claim(claim: dict, passages: list[dict]) -> dict:
    """Check one claim against its cited passages (or all retrieved ones if it cites none)."""
    evidence = [p for p in passages if p["chunk_id"] in claim["cited_ids"]] or passages
    if not NLI_ENABLED:
        return {**claim, "nli_entail": 0.0, "nli_contradict": 0.0,
                "method": "llm_judge", **llm_judge(claim["claim"], evidence)}

    nli = nli_scores(claim["claim"], evidence)
    base = {**claim, "nli_entail": round(nli["entail"], 3), "nli_contradict": round(nli["contradict"], 3)}

    if nli["entail"] >= NLI_ENTAIL_THRESHOLD:
        return {**base, "verdict": SUPPORTED, "method": "nli",
                "evidence_ids": [nli["entail_id"]], "reason": "NLI entailment"}
    return {**base, "method": "llm_judge", **llm_judge(claim["claim"], evidence)}