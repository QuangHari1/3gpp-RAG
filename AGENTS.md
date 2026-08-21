# Repository Guidelines

## Repository layout

This repository contains two independent RAG baselines.

- `Telco-RAG_api/` is the cloned and adapted Telco-oRAG paper baseline. Its Python code is in `src/`, `api/`, `scripts/`, `experiments/`, and `Telco-RAG_paper_version/`. `frontend/` is the original Next.js client retained for reference only.
- `newbaseline/` is the clean workspace for the new implementation. Put its source in `newbaseline/src/`, tests in `newbaseline/tests/`, and runnable helpers in `newbaseline/scripts/`.

Do not mix implementation code between the two baselines. It is fine to read `Telco-RAG_api/` for ideas, but the new baseline must own its dependencies, entry points, tests, and evaluation logic.

## Commands and validation

The paper baseline uses Python 3.11 and its existing benchmark is run from `Telco-RAG_api/`:

```bash
.venv/bin/python scripts/run_teleqna_benchmark.py --dataset datasets/TeleQnA.json --limit 1
```

Before changing that code, run `python3 -m compileall -q Telco-RAG_api` and `git diff --check`. Treat live provider execution, local benchmark execution, and static validation as separate results.

For `newbaseline/`, add focused tests and its dependency setup with the first implementation. Keep test data small and never commit API keys, downloaded 3GPP data, virtual environments, or generated benchmark results.

## Style

Use Python 4-space indentation, `snake_case` functions/modules, and `PascalCase` classes. Keep route handlers thin and make retrieval/evaluation behavior explicit and testable. Make changes narrowly scoped to one baseline unless shared documentation is being updated.
