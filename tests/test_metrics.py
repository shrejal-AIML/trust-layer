from evaluation.metrics import aggregate, context_hit, contains_all, is_abstained, score_question

def _record(answer: str, n_claims: int = 1, n_supported: int = 1) -> dict:
    return {"answer": answer, "passage_ids": ["doc1_chunk0"],
            "passage_texts": ["The Transformer got 28.4 BLEU."],
            "n_claims": n_claims, "n_supported": n_supported, "latency_s": 1.0}


def test_contains_all_ignores_case_hyphen_style_and_spacing():
    assert contains_all("RAG\u2011Token and rag sequence", ["rag-token", "RAG-Sequence"])
    assert not contains_all("only RAG-Token", ["rag-token", "rag-sequence"])


def test_is_abstained_handles_both_messages_and_curly_apostrophes():
    assert is_abstained("I can\u2019t answer this from the provided documents.")
    assert is_abstained("I can't verify this from the provided documents.")
    assert not is_abstained("The Transformer got 28.4 BLEU [doc1_chunk0].")


def test_score_question_checks_keywords_and_citation_validity():
    q = {"id": "q1", "answerable": True, "must_include": ["28.4"]}
    row = score_question(q, _record("28.4 BLEU [doc1_chunk0] [doc9_chunk9]"))
    assert row["correct"] and row["context_hit"]
    assert row["citation_valid"] == 0.5


def test_answering_an_unanswerable_question_is_wrong_and_abstaining_is_right():
    q = {"id": "q2", "answerable": False, "must_include": []}
    assert not score_question(q, _record("It is Olympus Mons."))["correct"]
    assert score_question(q, _record("I can't answer this from the provided documents.", 0, 0))["correct"]


def test_aggregate_splits_answerable_and_unanswerable():
    answerable = {"id": "a", "answerable": True, "must_include": ["28.4"]}
    unanswerable = {"id": "u", "answerable": False, "must_include": []}
    rows = [score_question(answerable, _record("28.4 [doc1_chunk0]")),
            score_question(unanswerable, _record("I can't answer this from the provided documents.", 0, 0))]
    metrics = aggregate(rows)
    assert metrics["answer_accuracy"] == 1.0 and metrics["abstain_recall"] == 1.0
    assert metrics["false_abstain_rate"] == 0.0

def test_context_hit_needs_one_passage_when_evidence_is_given():
    q = {"must_include": ["llama"], "evidence": ["base LM"]}
    assert not context_hit(q, ["Llama2-chat is a baseline.", "We use a base model."])
    assert context_hit(q, ["Llama2-chat is a baseline.", "Llama 2-7B as a base LM."])
    assert context_hit({"must_include": ["a", "b"]}, ["a here", "b there"])