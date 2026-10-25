import json
from bisect import bisect_right
from transformers import AutoTokenizer
from src.config import EMBED_MODEL, CHUNK_TOKENS, CHUNK_OVERLAP, REGISTRY_PATH

_tok = None


def _get_tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _tok.model_max_length = 10**9   # we chunk manually; silence length warning
    return _tok


def get_doc_id(filename: str) -> str:
    """Stable doc1, doc2, ... ids, persisted so they never shift."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        reg = {}
    if filename not in reg:
        reg[filename] = f"doc{len(reg) + 1}"
        REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg[filename]


def chunk_document(pages: list[str], doc_id: str, source: str,
                   size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    # Join pages, remembering where each page starts (for page numbers in citations)
    text, page_starts, page_nums = "", [], []
    for i, p in enumerate(pages):
        if not p:
            continue
        if text:
            text += "\n\n"
        page_starts.append(len(text))
        page_nums.append(i + 1)
        text += p
    if not text:
        return []

    enc = _get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    step = size - overlap

    chunks, idx = [], 0
    for start in range(0, n, step):
        end = min(start + size, n)
        c0, c1 = offsets[start][0], offsets[end - 1][1]
        chunk_text = text[c0:c1].strip()
        if chunk_text:
            page = page_nums[max(bisect_right(page_starts, c0) - 1, 0)]
            chunks.append({
                "chunk_id": f"{doc_id}_chunk{idx}",
                "doc_id": doc_id,
                "source": source,
                "page": page,
                "chunk_index": idx,
                "n_tokens": end - start,
                "text": chunk_text,
            })
            idx += 1
        if end == n:
            break
    return chunks