import json

from openai import OpenAI
from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 2000) -> str:
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

def extract_json(text: str):
    """Parse the first JSON object or array in an LLM reply (ignores fences and chatter)."""
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return obj