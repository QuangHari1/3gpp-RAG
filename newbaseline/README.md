# Telco-RAG new baseline

This directory is a self-contained, one-round RAG benchmark for TeleQnA. It
does not import or require `Telco-RAG_api/` at runtime.

The normal use case is simple: you already have the prepared `dataset/`
directory and the TeleQnA question file, and want to reproduce or compare an
experiment. **Do not download, chunk, or embed anything again** in that case.

## What is fixed in the supplied experiment

`config.toml` is the experiment configuration. Its current defaults are:

| Component | Value |
| --- | --- |
| Retrieval corpus | Paper-compatible Release-18 selection: 553 documents |
| Embeddings | `text-embedding-3-large`, 1,024 dimensions |
| Rephrase and answer model | `gpt-4o-mini` |
| Temperature | `0.0` |
| Semantic seed chunks | 8 |
| Citation expansion | Disabled (`citation_max_depth = 0`) |
| Tracking | MLflow local SQLite store and artifacts under `results/mlflow/` |

The precomputed corpus contains 252,329 embedded chunks. The full chunk source
contains 299,412 chunks so citation expansion can be enabled later without
rebuilding vectors.

## Required files

Run all local commands below **from `newbaseline/`**. The expected layout is:

```text
Telco-RAG/
├── newbaseline/
│   ├── config.toml
│   ├── pyproject.toml                    # UV dependency source of truth
│   ├── uv.lock                           # reproducible Python 3.11 lockfile
│   ├── resources/                         # router checkpoint and vocabulary
│   └── scripts/run_teleqna_benchmark.py
└── dataset/
    ├── teleqna/TeleQnA.json               # evaluation questions
    └── 3gpp/
        ├── Chunk/Rel-18/                  # ChunkSeries*.json
        ├── Embeddings/Rel-18/
        │   └── paper-baseline-gsma-rel18/ # manifest, .npy, metadata JSONL
        └── embedding_selections/
            └── paper-baseline-gsma-rel18.json
```

For an already embedded benchmark, `marked/Rel-18/` raw Markdown is not read
at runtime. Keep it only if you want to rebuild chunks or embeddings. Do not
move individual files out of the paths above: the manifest and configuration
refer to them.

## 1. Set the API key

For the default OpenAI configuration, create `newbaseline/.env`:

```env
OPENAI_API_KEY=sk-...
```

The key is used for query embeddings and the two LLM calls. It is never stored
in result files, manifests, or the Docker image.

To use a different compatible LLM, edit only `[llm]` and the two model names
under `[rag]` in `config.toml`. Keep the embedding model unchanged unless you
intend to regenerate the entire embedding corpus.

## 2. Install and verify locally

Install [uv](https://docs.astral.sh/uv/) and Python 3.11, then:

```bash
cd newbaseline
uv sync --all-groups
uv run scripts/run_teleqna_benchmark.py --help
```

`pyproject.toml` and `uv.lock` are the dependency source of truth. `uv sync`
creates the ignored local `.venv` automatically; do not activate it manually.
The lockfile already pins the CPU-only PyTorch index.

Run one paid smoke-test question before a larger experiment:

```bash
uv run \
  scripts/run_teleqna_benchmark.py \
  --limit 1 --workers 1 --progress-every 1 \
  --output results/teleqna/smoke-test.jsonl --no-compare
```

## 3. Run a benchmark

### Full dataset

Use a new output name for every distinct experiment. Results are appended one
question at a time, so rerunning the same command resumes safely.

```bash
uv run \
  scripts/run_teleqna_benchmark.py \
  --workers 4 --progress-every 10 \
  --output results/teleqna/repro-full.jsonl --no-compare
```

`--workers 4` processes four independent questions concurrently. Reduce it if
the provider rate-limits the account. The output's sibling
`repro-full.manifest.json` records every non-secret run parameter and the
TeleQnA SHA-256.

### Harder tail-200 comparison

This is the recommended quick experiment: reverse question order first, then
take the last 200 numeric questions. It compares the candidate with the
existing full baseline on exactly the shared scored questions.

```bash
uv run \
  scripts/run_teleqna_benchmark.py \
  --reverse --limit 200 --workers 4 --progress-every 10 \
  --output results/teleqna/my-change-tail200.jsonl \
  --compare-to results/teleqna/paper-baseline-gsma-rel18.jsonl
```

The terminal prints candidate accuracy, baseline accuracy, delta, improved and
regressed counts. The same data is saved next to the run as
`my-change-tail200.comparison.json`.

If the full baseline JSONL already exists and you omit `--output`, the runner
automatically uses `results/teleqna/paper-baseline-gsma-rel18-tail200.jsonl`
and compares it with `paper-baseline-gsma-rel18.jsonl`.

### Resume or restart

- Run the same command again to resume only unfinished question IDs.
- Add `--overwrite` to restart the specified `--output` JSONL from zero.
- Never use the same output filename for two different configurations.

Questions with no expected option are saved but marked `unscored`; they do not
contribute to accuracy or comparisons.

## Results and error analysis

Each benchmark JSONL row stores the answer, expected/predicted option,
correctness, router decision, semantic retrieval traces, and citation paths.
The full retrieved text is intentionally not repeated in every row.

Generate error-analysis tables and a readable summary:

```bash
uv run \
  scripts/analyze_teleqna_errors.py \
  --results results/teleqna/my-change-tail200.jsonl \
  --output-dir results/analysis/my-change-tail200
```

Read `results/analysis/my-change-tail200/summary.md` first. The directory also
contains CSV/JSON breakdowns for wrong answers, semantic scores, router series,
and citation paths.

MLflow is local by default. Every benchmark records its config, progress/final
metrics, result JSONL, manifest, comparison, and available analysis files under
`results/mlflow/`; no account or cloud upload is needed. Set
`[experiment_tracking].mode = "disabled"` in `config.toml` to turn this off.

To browse runs locally, keep this command running in a second terminal from
`newbaseline/`, then open <http://127.0.0.1:5000>:

```bash
uv run scripts/mlflow_ui.py
```

To import pre-existing JSONL checkpoints into the same UI once:

```bash
uv run \
python scripts/import_teleqna_results_to_mlflow.py
```

## Changing retrieval settings

For a clean ablation, edit `config.toml`, choose a fresh `--output` filename,
and rerun tail-200. The benchmark manifest captures these settings.

```toml
[rag]
retrieval_top_k = 8          # number of semantic seed chunks
citation_max_depth = 0       # 0 = no citation expansion
citation_total_chunks = 8    # total context budget, including seeds
```

To enable one-hop citation expansion while retaining eight seed chunks, use
for example:

```toml
citation_max_depth = 1
citation_total_chunks = 13
citation_chunks_per_heading = 2
```

Changing `rephrase_model`, `answer_model`, temperature, retrieval limits, or
citation settings does **not** require re-embedding. Changing `[embedding]`
does require new vectors; also set `router_backend = "semantic"` if the
embedding model is no longer the paper-compatible OpenAI model.

### Vocabulary ablation

`paper_legacy` reads the original paper DOCX and is the paper-equivalent
baseline. `release18_unambiguous` keeps the 569 copied paper term definitions,
but expands an acronym only when the provenance-rich Release-18 catalog has
exactly one meaning.

The default `release18_contextual` adds a semantic second pass only for a
question containing an ambiguous acronym such as `AMF` or `ARP`:

1. Retrieve seed chunks without expanding that acronym.
2. Compare each candidate meaning against those seeds and its source-series provenance.
3. Re-retrieve once with the winner only when its score and margin clear the
   `[vocabulary]` thresholds; otherwise abstain and retain the acronym.

This does not add an LLM call. The trace stores candidates, scores, confidence,
margin, and the selected meaning for later error analysis. To reproduce the
existing unambiguous ablation, change only:

```toml
[vocabulary]
mode = "release18_unambiguous"
```

`contextual_excluded_acronyms = ["3GPP"]` is intentional: `[3GPP Release N]`
is question metadata, not a term whose sense should affect retrieval.

## Run with Docker

The published image already contains this baseline, its router assets, the
prepared dataset, chunk files, embeddings, and previous offline artifacts. It
does not contain API keys, `Telco-RAG_api/`, or a development virtualenv.

```bash
docker pull quanghari/telco-rag-newbaseline:latest

docker run --rm -it \
  -e OPENAI_API_KEY \
  -v telco-rag-results:/workspace/newbaseline/results \
  quanghari/telco-rag-newbaseline:latest \
  newbaseline/scripts/run_teleqna_benchmark.py \
  --reverse --limit 200 --workers 4 --progress-every 10
```

The named volume preserves new results after the container exits. Do not
bind-mount an empty host `results/` directory because it hides the baseline
JSONLs bundled in the image. To use a different prepared dataset, mount it
read-only at `/workspace/dataset`:

```bash
docker run --rm -it \
  -e OPENAI_API_KEY \
  -v /absolute/path/to/dataset:/workspace/dataset:ro \
  -v telco-rag-results:/workspace/newbaseline/results \
  quanghari/telco-rag-newbaseline:latest \
  newbaseline/scripts/run_teleqna_benchmark.py --limit 1 --no-compare
```

## Rebuild the corpus (optional)

Only do this when prepared artifacts are absent or you deliberately changed
the embedding model/corpus. It downloads data and embedding requires paid API
calls, so it is not part of normal benchmark reproduction.

From the repository root:

```bash
uv run --project newbaseline \
  newbaseline/scripts/run_offline_pipeline.py --mode paper

uv run --project newbaseline \
  newbaseline/scripts/run_offline_pipeline.py --mode paper --embed --dry-run
```

Run the final command again with `--embed` (without `--dry-run`) only when you
intend to create or resume paid embeddings. The paper selection covers 549
Release-18 specifications plus the four Rel-14--Rel-17 summary documents.
