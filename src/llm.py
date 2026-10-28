import hashlib
import json
import time

from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_CACHE_PATH, LLM_MODEL, LLM_TIMEOUT_S

_client = None
_cache: dict[str, str] | None = None
stats = {"calls": 0, "cache_hits": 0, "seconds": 0.0}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                         timeout=LLM_TIMEOUT_S, max_retries=2)
    return _client


def _get_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        if LLM_CACHE_PATH.exists():
            for line in LLM_CACHE_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    _cache[row["key"]] = row["text"]
    return _cache


def _cache_key(messages: list[dict], temperature: float, max_tokens: int) -> str:
    payload = json.dumps([LLM_MODEL, messages, temperature, max_tokens],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 2000) -> str:
    """Chat completion. Replies at temperature 0 are cached on disk (keyed by model + prompt)."""
    cache = _get_cache() if temperature == 0 else None
    key = _cache_key(messages, temperature, max_tokens)
    if cache is not None and key in cache:
        stats["cache_hits"] += 1
        return cache[key]

    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    stats["calls"] += 1
    stats["seconds"] += time.perf_counter() - t0

    if cache is not None and text:
        cache[key] = text
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
    return text


def extract_json(text: str):
    """Parse the first JSON object or array in an LLM reply (ignores fences and chatter)."""
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return obj