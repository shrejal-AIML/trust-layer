import argparse
import json

import mlflow
from tqdm import tqdm

from evaluation.ablations import CONFIGS, Config, run_question
from evaluation.metrics import aggregate, contains_all, score_question
from src import llm
from src.config import CHUNKS_PATH, EMBED_MODEL, LLM_MODEL, RERANK_MODEL, ROOT

QUESTIONS_PATH = ROOT / "data" / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"
TABLE_COLUMNS = ("answer_accuracy", "abstain_recall", "false_abstain_rate", "context_hit_rate",
                 "faithfulness", "unsupported_per_answer", "citation_valid_rate", "latency_mean_s")


def load_questions(limit: int | None = None) -> list[dict]:
    lines = QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    return questions[:limit] if limit else questions


def check_gold() -> None:
    """Sanity check, no LLM: gold keywords must exist in the corpus, evidence inside a single chunk."""
    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in lines if line.strip()]
    corpus = " ".join(texts)
    bad = 0
    for q in load_questions():
        if not q["answerable"]:
            continue
        if not contains_all(corpus, q["must_include"]):
            bad += 1
            print(f"MISSING keyword: {q['id']} {q['must_include']}")
        if q.get("evidence"):
            n = sum(contains_all(t, q["evidence"]) for t in texts)
            print(f"{q['id']} evidence {q['evidence']} found in {n} chunk(s)")
            bad += n == 0
    print(f"gold check done: {bad} problem(s)")


def run_config(config: Config, questions: list[dict]) -> dict:
    before = dict(llm.stats)
    rows, errors = [], []
    for q in tqdm(questions, desc=config.name):
        try:
            record = run_question(config, q["question"])
        except Exception as exc:    # keep going; errors are reported, not scored
            errors.append({"id": q["id"], "error": repr(exc)})
            continue
        rows.append({**score_question(q, record), "question": q["question"],
                     "answer": record["answer"], "loops": record["loops"]})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{config.name}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    metrics = aggregate(rows)
    usage = {k: round(llm.stats[k] - before[k], 1) for k in llm.stats}
    print(f"  {config.name}: scored={len(rows)} errors={len(errors)} "
          f"llm_calls={usage['calls']} cache_hits={usage['cache_hits']} llm_seconds={usage['seconds']}")
    for e in errors:
        print(f"  ERROR {e['id']}: {e['error']}")

    with mlflow.start_run(run_name=config.name):
        mlflow.log_params({"mode": config.mode, "verify_loop": config.verify_loop,
                           "n_questions": len(rows), "n_errors": len(errors),
                           "llm_model": LLM_MODEL, "embed_model": EMBED_MODEL,
                           "rerank_model": RERANK_MODEL})
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.log_metrics({"llm_calls": usage["calls"], "llm_seconds": usage["seconds"]})
        mlflow.log_artifact(str(path))
    return metrics


def print_table(results: dict[str, dict]) -> None:
    header = f"{'config':<15}" + "".join(f"{c[:14]:>15}" for c in TABLE_COLUMNS)
    print("\n" + header + "\n" + "-" * len(header))
    for name, metrics in results.items():
        cells = "".join(f"{'-' if metrics[c] is None else metrics[c]:>15}" for c in TABLE_COLUMNS)
        print(f"{name:<15}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation / ablation study.")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--check-gold", action="store_true", help="verify gold keywords exist in the corpus")
    args = parser.parse_args()

    if args.check_gold:
        check_gold()
        return

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("trust-layer")

    questions = load_questions(args.limit)
    results = {name: run_config(CONFIGS[name], questions) for name in args.configs}
    print_table(results)


if __name__ == "__main__":
    main()