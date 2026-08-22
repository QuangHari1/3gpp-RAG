FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/workspace/newbaseline/.venv/bin:$PATH"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY newbaseline/pyproject.toml newbaseline/uv.lock /workspace/newbaseline/
WORKDIR /workspace/newbaseline
RUN uv sync --frozen --no-dev
WORKDIR /workspace

# These are the owned implementation, router assets, raw corpus, chunks,
# embeddings, release summaries, and recorded local experiment artifacts.
COPY newbaseline /workspace/newbaseline
COPY dataset /workspace/dataset
COPY README.md AGENTS.md /workspace/

# Do not bake API keys into the image. Supply OPENAI_API_KEY (and HF_TOKEN only
# when downloading gated upstream data) at `docker run` time.
ENTRYPOINT ["python"]
CMD ["newbaseline/scripts/run_teleqna_benchmark.py", "--help"]
