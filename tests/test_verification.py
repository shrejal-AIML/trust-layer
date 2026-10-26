from src.llm import extract_json
from src.verification.scorer import score_answer
from src.verification.verifier import make_windows


def _result(verdict: str, method: str = "nli", entail: float = 0.95) -> dict:
    return {"verdict": verdict, "method": method, "nli_entail": entail}


def test_extract_json_ignores_fences_and_chatter():
    assert extract_json('Sure!\n```json\n[{"claim": "x"}]\n```') == [{"claim": "x"}]


def test_make_windows_overlap_and_cover_the_end():
    windows = make_windows("doc1_chunk0", "A. B. C. D. E. F. G.", size=3, stride=2)
    assert [w[1] for w in windows] == ["A. B. C.", "C. D. E.", "E. F. G."]


def test_all_supported_returns_answer():
    score = score_answer([_result("SUPPORTED"), _result("SUPPORTED", "llm_judge")])
    assert score["decision"] == "return" and score["faithfulness"] == 1.0


def test_unsupported_claim_triggers_revise():
    score = score_answer([_result("SUPPORTED"), _result("CONTRADICTED")])
    assert score["decision"] == "revise"
    assert score["faithfulness"] == 0.5 and score["contradicted"] == 1