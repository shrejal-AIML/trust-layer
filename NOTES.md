# Project notes (for the final README)

## Data quirks
- The Transformer paper reports 41.8 BLEU (EN-FR) in the abstract and results table,
  but 41.0 in one sentence of the body text. The gold set (q02) accepts either ("41.").

## Eval findings so far
- 15-question gold set: abstention recall 1.0, citation validity 1.0 in all configs.
- Gold set is too easy to separate dense / hybrid / rerank. Needs harder questions.
- Latency is dominated by DeBERTa NLI on CPU, not by the LLM (~1 s per Groq call).

## Open issues
- q15 false abstain ("Which base language model is Self-RAG built on?"): not yet diagnosed.