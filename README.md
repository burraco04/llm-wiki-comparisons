# LLM Wiki comparisons

## Structured wiki benchmark (`structured_wiki_v2`)

`benchmark_llm_wiki.py` adapts the non-interactive method in
`tutorial_LLM_wiki_v3.ipynb`. Each source is processed through structured
extraction and, after the first source, reconciliation against existing entities
and concepts. Deterministic code creates Markdown source/entity/concept pages,
keeps per-source descriptions, applies validated ID merges, records proposed
contradictions for review, adds reciprocal links and saves a provenance graph.
Invalid extraction fails explicitly after one retry; it never silently falls
back to a source-only wiki. Open-ended semantic correctness still depends on
the generating model, even when JSON satisfies its schema.

The comparison runs five systems:

| System | Retrieval |
| --- | --- |
| `rag_semantic` | Top 3 dense matches over 200-word raw-text chunks |
| `wiki_routed` | LLM-selected source/entity/concept pages, keyword-ranked paragraphs, raw-text safety net and graph triples |
| `wiki_bm25` | BM25 over compiled-page and raw-source paragraphs, with graph triples |
| `wiki_hybrid` | BM25 + dense paragraph retrieval fused with reciprocal rank fusion (`k=60`), with graph triples |
| `llmwiki2` | External library in CROW mode; generative LLM plus `LiquidAI/d1-3B` |

As in the notebook, **all three wiki strategies perform LLM routing and search
term expansion** before retrieval; routing directly constrains only the routed
strategy. Wiki BM25 therefore uses expanded search terms, not just the original
question. The wiki stays frozen during evaluation: there are no analysis/demo
pages and no query write-back. Generative answers share the same prompt across
the first four systems. The context budget includes labels, separators and graph
text (12,000 characters). The library retains its own internal context policy.

Differences from the tutorial are limited to batch-oriented choices: typed page
paths prevent source/entity ID collisions; English stopwords come from
scikit-learn (no NLTK corpus download); source claims are retained on merged
pages; invalid references are rejected; whole-context budgets are enforced.
The tutorial's visualization, interactive examples, lint demonstrations and
write-back demonstrations are not part of this evaluation. Raw RAG chunks remain
fixed word windows, rather than the notebook's paragraph-aligned windows.

The implementation and CROW logic inside `llmwiki2` belong to its authors and
are not modified here. Jev is not used by the default benchmark configuration.

### Run on the GPU server

Rebuild the application image: the benchmark source and requirements are copied
into it, so reusing an old image would run the old implementation.

```bash
docker build -t llm-wiki-benchmark:cu129 .

# Small validation run (3 documents, up to 25 questions)
PORT=8000 sbatch --export=ALL -w faretra run_with_vllm.sh \
  Qwen/Qwen2.5-3B-Instruct -- python benchmark_llm_wiki.py

# Existing launcher: both generative models, 3/25 and 8/150 configurations
bash run_both_models.sh
```

Inspect the small run's generated wiki and extraction logs before submitting
the larger comparison. Models/endpoints still use the existing vLLM launchers.
The local automated tests below do not launch these GPU experiments.

New runs use their own versioned, timestamped directories:

```text
results/structured_wiki_v2/<model>/docs_<N>_questions_<M>/<UTC-run-id>/
  benchmark_manifest.json
  notebook_wiki_bundle/
    documents/
    compilation.json
    wiki/{sources,entities,concepts}/
    wiki/{index.md,knowledge_graph.json,log.json}
  notebook_ingestion.json
  notebook_answers.jsonl
  notebook_results.json
  llmwiki2_bundle/
  llmwiki2_answers.jsonl
  llmwiki2_calls.jsonl
  full_results.json
  summary.json
  summary.csv
```

Old result directories are preserved. JSONL answers are saved after each query;
these are interruption checkpoints, not automatic resume files. The manifest
records code/document hashes, package versions and retrieval settings. Summaries
report generative and classifier calls/tokens separately, as well as their totals.
Ingestion and indexing costs are recorded separately from per-query latency.

Results from the old runner should be treated as a different protocol: it kept
only the first line of model replies, omitted entity/concept pages and used
different retrieval settings. New scores cannot be attributed solely to adding
entity pages. Both models and all systems need fresh runs for a consistent
comparison. No empirical improvement is assumed by the code change itself.

### Offline regression tests

Install `requirements.txt` for the complete benchmark. The offline tests need
only numpy, openai, rank-bm25, rouge-score, jsonschema, nltk and scikit-learn;
they replace model inference and the embedding encoder with deterministic fakes.

```bash
python -m unittest -v test_benchmark_llm_wiki
```

Tests cover multiline responses, JSON validation/retries, merging and provenance,
graph endpoints/backlinks, routing fallbacks, the full context budget, all four
notebook strategies and classifier accounting. They verify orchestration, not
model accuracy, vLLM schema support or the external library's behavior.
