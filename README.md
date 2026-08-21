# Telco-RAG baselines

This workspace keeps two independent implementations for comparison.

- `Telco-RAG_api/` contains the cloned and adapted Telco-oRAG paper baseline. It is retained as a reference for its routing, retrieval, validation, API, experiments, and legacy frontend ideas.
- `newbaseline/` is the workspace for the new baseline. Keep its implementation independent so comparison results remain meaningful.

## Layout

`Telco-RAG_api/` now contains all source code from the paper baseline:

- `src/`, `api/`, `scripts/`, `experiments/`, and `Telco-RAG_paper_version/` are its Python implementation.
- `frontend/` contains the original Next.js client, retained only as reference code.
- `datasets/`, `results/`, `.venv/`, and `output.json` are local runtime artifacts and are not tracked.

Start new work in `newbaseline/src/`. Add that baseline's dependency configuration, tests, scripts, and documentation inside `newbaseline/`; do not import paper-baseline modules directly as implementation dependencies.

## Paper-baseline validation

The paper baseline requires Python 3.11. Its existing benchmark can be run from its own directory:

```bash
cd Telco-RAG_api
.venv/bin/python scripts/run_teleqna_benchmark.py --dataset datasets/TeleQnA.json --limit 1
```

This command requires locally provisioned data and an `OPENAI_API_KEY` when generation is enabled.

## License

MIT. See [license](./license).
