import re
import unicodedata
from collections import Counter


def _norm(line: str) -> str:
    # digits -> '#', so "Page 3" and "Page 4" count as the same line
    return re.sub(r"\d+", "#", line.strip().lower())


def _find_boilerplate(pages: list[str], min_ratio: float = 0.4) -> set[str]:
    """Short lines that repeat on many pages = headers/footers/page numbers."""
    if len(pages) < 4:
        return set()
    counts = Counter()
    for p in pages:
        seen = {_norm(l) for l in p.splitlines() if l.strip() and len(l.strip()) < 100}
        counts.update(seen)
    return {line for line, c in counts.items() if c / len(pages) >= min_ratio}


def _clean_page(text: str, boilerplate: set[str]) -> str:
    text = unicodedata.normalize("NFKC", text)          # fixes ligatures like ﬁ
    text = text.replace("\u00ad", "")                   # soft hyphen
    lines = [l for l in text.splitlines() if _norm(l) not in boilerplate]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)        # single newline -> space
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[str]) -> list[str]:
    boilerplate = _find_boilerplate(pages)
    return [_clean_page(p, boilerplate) for p in pages]