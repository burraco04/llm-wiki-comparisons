# LLM Wiki comparison benchmark

Direct calls disable implicit OpenAI SDK transport retries so each request attempt is visible. Ordinary invalid JSON allows two validation failures. Truncated JSON can use up to three generation attempts with increasing output budgets. Every request and recovery remains in the traces.

## Protocol: structured_wiki_v4

Each model/subset run evaluates six configurations on the same documents and questions:

| Configuration | Evidence selection |
| --- | --- |
| `rag_semantic` | Top 3 embedding matches over non-overlapping 200-word source chunks |
| `wiki_routed` | LLM page selection and query expansion, keyword-ranked paragraphs and graph relations |
| `wiki_bm25` | Same preparation; BM25 over wiki and original-source paragraphs |
| `wiki_hybrid` | Same preparation; BM25 and dense rankings fused through RRF, constant 60 |
| `llmwiki2_classic` | Independent note wiki with generative organizational and retrieval decisions |
| `llmwiki2_crow` | Independent note wiki with Decision Model decisions and generative fallback |

The structured wiki is built once and shared by its three retrieval strategies.
All three use LLM page selection and search terms; BM25 and hybrid rank the full
paragraph index. Direct contexts are limited to 12,000 characters, including
labels, separators and graph text. The note library retains its own context
policy. Neither wiki ingests generated answers during evaluation.

Classic and CROW build fresh bundles with the same source order, writer, prompt
overrides and installed library. The default classifier is `LiquidAI/d1-3B`.
The library is third-party work; its internal algorithms are not modified.
Its version is pinned to 0.8.0, matching the recorded v3 experiment.

## Recovery from truncated structured extraction

The v3 large Qwen2.5 run stopped on document 5 after two incomplete JSON
responses, both ending at 4,096 completion tokens. The traces also contain
whitespace padding and repeated relations, not a context-length or memory error.

Version v4 keeps the original extraction prompt, schema, source documents and
initial budgets. Only invalid responses trigger recovery:

- An invalid reply ending with `finish_reason=length` increases the output budget,
  normally 4,096 → 8,192 → 16,384 tokens for extraction. Retry growth is bounded
  by measured prompt usage and the served context limit, with 128 tokens reserved.
- Repair instructions request compact JSON without whitespace padding or repeated
  items. Non-truncated validation failures retain their output budget.
- If whole-document extraction still fails after truncation, three smaller JSON
  outputs request facts, nodes, then relations. Each reads the full original
  document; relations receive the actual extracted node IDs. Partial responses
  are discarded, never repaired into invented knowledge or source-only pages.
- Each call allows at most three generation attempts, or two ordinary validation
  failures. Failed schema-support negotiation can add a separately traced request.
  If the sectional recovery also fails, construction still fails explicitly.

The manifest records the policy, repair instruction and served context limit.
Requests record actual caps, validation outcomes and retry reasons. Ingestion
records distinguish `single` from `sectioned_after_truncation`. All recovery work
is included in construction time, calls and token usage. Historical v3 outputs
remain unchanged; new outputs go under `results/structured_wiki_v4`.

The served limit defaults to the container's `LLM_MAX_MODEL_LEN`. Use
`--llm-context-tokens` only to match the actual inference service configuration.
Changing recovery can change the generated wiki; use the same v4 image for all
conditions of a new comparative matrix and do not silently pool v3 and v4.

## Build and submit

**Rebuild the image before submitting:** it contains the benchmark, adapters,
metadata collector and prompts. Use the same rebuilt image for all four runs;
keep the source checkout unchanged while runs are queued or executing.

~~~bash
docker build -t llm-wiki-benchmark:structured-wiki-v4 .
VLLM_IMAGE=llm-wiki-benchmark:structured-wiki-v4 bash run_both_models.sh
~~~

All launchers default to `llm-wiki-benchmark:structured-wiki-v4`.
Before submitting, `run_both_models.sh` checks the actual image files against
the local benchmark, adapters, service script, metadata collector, Markdown
prompts and requirements. This check uses a temporary container without GPUs
or network access. A stale or missing image stops submission with a rebuild
instruction. Each `run_with_vllm.sh` job repeats the check before starting
services; both launchers use the checked immutable image ID. Updating a tag
cannot silently switch an already submitted job to a different image.

To check an existing image separately, without submitting a job:

~~~bash
python3 verify_benchmark_image.py llm-wiki-benchmark:structured-wiki-v4
~~~

The launcher submits four Slurm jobs: Qwen2.5-3B-Instruct and Qwen3.5-4B, each
with 3 documents/up to 25 questions and 8 documents/up to 150 questions. Each job
runs all six configurations and reserves two GPUs, one per inference service.
Classic and CROW run sequentially within that allocation.

For a small validation run:

~~~bash
VLLM_IMAGE=llm-wiki-benchmark:structured-wiki-v4 PORT=8000 \
  sbatch --export=ALL run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py --top-docs 3 --max-questions 25 \
  --wiki-modes classic crow
~~~

`--wiki-modes classic crow` is the default. `--wiki-modes crow` selects a CROW-only
diagnostic run, which cannot provide the paired comparison.
Pass `--dataset-revision <commit-SHA>` to the benchmark to pin the dataset;
selected inputs, dataset identity and fingerprints are recorded.

Both note-wiki constructions add work. Set the allocation time without editing
the launcher when needed:

~~~bash
RUN_TIME_LIMIT=12:00:00 VLLM_IMAGE=llm-wiki-benchmark:structured-wiki-v4 \
  bash run_both_models.sh
~~~

To rerun only the previously incomplete condition, on its original dataset revision:

~~~bash
VLLM_IMAGE=llm-wiki-benchmark:structured-wiki-v4 PORT=8002 \
  sbatch --export=ALL -w faretra run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py --top-docs 8 --max-questions 150 \
  --wiki-modes classic crow --dataset-revision 71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82
~~~

Rebuild the image first. This runs all six configurations in the requested
condition and keeps the historical run intact. A complete new matrix uses
`bash run_both_models.sh` instead.

The launchers capture actual image identity, service commands, hardware
inventory and package provenance when available. These are startup records,
not peak-memory or energy measurements.

## Troubleshooting an immediate exit

### Numeric GPU access works, but UUID lookup fails

Logs `exp-16605618` through `exp-16605621` stopped before Docker/model startup:
Slurm allocated GPUs `2,3`, index-to-UUID discovery succeeded, but the subsequent
`nvidia-smi -i GPU-190ed... --query-gpu=memory.free` failed with exit 255.
On the same host, querying index 2 returned the expected UUID, PCI `81:00.0` and
free memory, whereas querying that UUID reported an error for PCI `01:00.0`.
This establishes a UUID lookup/access problem; it does not establish that
the allocated GPU is physically defective or that CUDA works.

`start_vllm.sh` now retains numeric selectors when the normalized
`CUDA_VISIBLE_DEVICES` and `SLURM_JOB_GPUS` assignments agree. It checks each
allocated index's UUID and memory, passes exactly those indices to Docker, and
retains both UUIDs in logs/host allocation metadata. It refuses empty, duplicate
or conflicting assignments and never selects a different GPU as a workaround.
Numeric memory failures remain fatal. The UUID-only allocation path is retained.

This correction affects host scripts, so the existing verified v4 image can be
used without rebuilding it. Copy the updated host scripts to the server first.
Before queuing the full matrix, one scheduled startup-only check can verify
both real inference services and then print CLI help without running experiments:

~~~bash
VLLM_IMAGE=llm-wiki-benchmark:structured-wiki-v4 PORT=8000 \
  sbatch --export=ALL -w faretra run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py --help
~~~

After this reaches `ready` and `done`, submit `bash run_both_models.sh` normally.
If Docker or CUDA still reports an error, inspect that new service log: the
successful index query alone cannot establish CUDA health. Driver/device repair
must be coordinated with the server administrator, without resetting GPUs used
by other allocations. `gpu_diagnostic.sh` now queries only its allocated devices
and includes PCI identifiers; unrelated GPUs no longer determine its result.

### An older image rejects benchmark arguments

The November 11 logs (`exp-16419749` through `exp-16419752` and
`exp-16420762` through `exp-16420765`) show both model services becoming ready,
followed by `unrecognized arguments: --wiki-modes classic crow`. The image
`llm-wiki-benchmark:cu129` contains an older benchmark parser. The Dockerfile
copies the application into the image; only `results` is mounted from the
host. Copying updated Python files to the server therefore does not update
the code executed in the container. Rebuild using the command above and
submit with that image. Removing `--wiki-modes` would still run the old
benchmark and would not fix the version mismatch.

`>>> stopping vLLM` is printed by the launcher's exit handler on both success
and failure after image verification. It does not mean that the model services
or benchmark started.
The launcher now reports its phase and preserves the failing exit code;
`start_vllm.sh` also identifies the failing NVIDIA query and prints its output.
Previously, a failed `nvidia-smi` call could put its diagnostic on captured
stdout and terminate the script under `set -e`, leaving only the exit message.

`gpu_discovery` and `gpu_memory_query` failures occur before either model is
started. Check the allocation variables and GPU visibility inside the
same kind of allocation. This short diagnostic uses no models or containers:

~~~bash
mkdir -p logs
sbatch -w faretra gpu_diagnostic.sh
~~~

Submit this script directly so Slurm uses its Bash shebang. The previous
`--wrap` example was interpreted by `/bin/sh`, causing `redirection unexpected`
on the Bash here-string in `gpu-diagnostic-16420563.out`. The diagnostic now
checks every allocated GPU and preserves query failures in its final status.

The allocation variables may need to be compared with the indices and UUIDs
actually visible to `nvidia-smi`. A query on the login shell alone does not
check that allocation. Do not replace the allocated GPU list with arbitrary
devices on the host. An empty `squeue` result shows that the specified runs are
not currently queued; absence from `sacct` or `scontrol` alone does not explain
an old run's failure. Check the log timestamp and the earliest startup error.

Offline launcher checks with Python 3.8 or later, without GPUs or Docker:

~~~bash
python3 -m unittest -v test_startup_scripts test_verify_benchmark_image test_gpu_diagnostic
~~~

## Saved outputs

Every run has a separate versioned, timestamped directory:

~~~text
results/structured_wiki_v4/<model>/docs_<N>_questions_<M>/<UTC-run-id>/
  benchmark_manifest.json
  runtime_environment.json
  run_status.json
  dataset/
    documents.jsonl
    questions.jsonl
    statistics.json
    metadata.json
  prompts/
    direct_prompts.json
    schemas.json
    overrides/<operation>.md
    manifest.json
  structured_wiki_bundle/
    documents/
    compilation.json
    wiki/{sources,entities,concepts}/
    wiki/{index.md,knowledge_graph.json,log.json}
  structured_wiki/
    ingestion.jsonl
    ingestion/<document>.json
    retrieval_index.json
    raw_embeddings.npy
    wiki_embeddings.npy
    indexing.json
    model_calls.jsonl
    queries/<system>/<question>.json
    answers.jsonl
    results.json
  llmwiki2_classic/             # same files also under llmwiki2_crow/
    bundle/
    configuration.json
    model_calls.jsonl
    usage.jsonl
    ingestion.jsonl
    ingestion/<document>.json
    queries/<question>.json
    answers.jsonl
    results.json
    knowledge_before_queries.json
    knowledge_after_queries.json
  full_results.json
  summary.json
  summary.csv
  construction_summary.json
  summary_by_type.json
  summary_by_support_count.json
  paired_comparison.json
~~~

Query traces retain actual evidence, selected pages/notes, decisions and answer
requests. Model-call traces record request/response payloads and reported usage
with credentials removed. The thesis example `mhop_424` belongs to the larger
subset and can be inspected through these traces.

The adapter records installed library version/source identity and resolved
configuration, including thresholds, beam widths, state limits and generation
settings. Unsupported tracing APIs are reported explicitly. Defaults from
another installation are not substituted. Native library features are retained;
mode-specific features and other differences are recorded as possible confounds.

## Reading results and failures

Check `run_status.json` and component statuses before using summaries. Completed,
partial and failed components are distinguished. Failed construction prevents
evaluation of that incomplete wiki; subsequent independent components are
still attempted. Query errors retain their question, score zero and remain in
the averaging denominator. Checkpoints preserve work but do not auto-resume.

Summary scores are per-question EM, token F1 and ROUGE-L means. Generative and
classifier calls/tokens are separate and also totaled. Construction/indexing
costs are separate from query latency. Group tables describe question types and
support counts; the overall mean is not category balanced. The paired report
checks common conditions, completeness and configuration differences before
presenting classic–CROW deltas.

Latency includes retrieval, model waiting and some trace-writing overhead;
it is not pure inference time. Existing outputs are preserved. Compare v3 runs
with other v3 runs using compatible recorded settings.

## Offline validation

With the benchmark dependencies installed:

~~~bash
python -m unittest -v test_benchmark_llm_wiki test_benchmark_artifacts test_llmwiki_adapter
python3 -m unittest -v test_startup_scripts test_verify_benchmark_image test_gpu_diagnostic
for script in *.sh; do bash -n "$script" || exit; done
~~~

Tests use deterministic model/encoder fakes and HTTP mock transports. They
check orchestration, artifacts, error accounting and tracing, without running
GPU experiments or establishing accuracy. A local library integration check
does not establish the release/configuration used by the server.
