def print_report(verification: dict) -> None:
    print("\n=== CLAIMS ===")
    for i, c in enumerate(verification["claims"], start=1):
        print(f"{i}. [{c['verdict']}] via {c['method']} "
              f"(nli_entail={c['nli_entail']}, nli_contradict={c['nli_contradict']})")
        print(f"   claim:    {c['claim']}")
        print(f"   evidence: {c['evidence_ids']}  {c['reason']}")
    s = verification["score"]
    print(f"\n=== SCORE === decision={s['decision']} confidence={s['confidence']} "
          f"faithfulness={s['faithfulness']} "
          f"(supported={s['supported']}, contradicted={s['contradicted']}, "
          f"not_enough_info={s['not_enough_info']})")