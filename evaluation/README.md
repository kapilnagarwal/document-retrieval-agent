# Real embedding retrieval evaluation

`retrieval_cases.json` contains eight synthetic documents and twenty labeled questions: fourteen with relevant sources and six intended to be unsupported. All content is public learning material. The runner creates temporary storage, uses real MiniLM embeddings, and never calls Gemini or reads private profile data.

```bash
python -m scripts.evaluate_retrieval
# Reuse a previously downloaded launcher model cache:
HF_HOME="$PWD/data/model_cache" HF_HUB_OFFLINE=1 python -m scripts.evaluate_retrieval
```

## Metrics

- `recall_at_k`: macro fraction of expected source filenames present among the top-k chunk results.
- `mrr_at_k`: reciprocal rank of the first relevant source in the returned source ranking.
- `false_support_rate`: fraction of negative questions accepted by the evidence gate.
- `positive_support_rate`: fraction of positive questions accepted by that gate; report this to expose over-abstention.
- `mean_seconds` and `p95_seconds`: warmed query retrieval plus evidence-check time, excluding model loading and indexing.

Default gates are recall >= 0.90, MRR >= 0.75, false support <= 0.10, and positive support >= 0.80. CLI arguments expose each threshold and `--minimum-coverage`. Reports are written to ignored `data/evaluation/report.json`; an unmet gate exits nonzero.

## Local baseline

On the synthetic corpus with MiniLM, top-k 5, and minimum keyword coverage 0.5: recall 1.0, MRR 1.0, false support 0.0, positive support 12/14 (approximately 0.857). Two relevant natural-language questions were conservatively rejected by the lexical gate. This is a small development benchmark, not a production accuracy claim. Latency is machine-dependent and is reported per run rather than fixed here.

The fixture emphasizes known constraints and some natural-language variation. Expand it with independently labeled paraphrases, adversarial negatives, and representative permitted documents before choosing a new model, chunk policy, or gate. Keep a held-out set when tuning. Retrieval ranking and a valid citation label do not measure whether a generated answer is factually supported.

The manually dispatched GitHub workflow runs the real benchmark. Normal CI avoids model downloads and validates the metric calculations and application behavior using deterministic doubles.
