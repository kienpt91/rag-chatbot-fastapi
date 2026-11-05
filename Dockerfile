# The service image: non-root, base image pinned by digest (python 3.13.17-slim-trixie, index pushed 2026-10-24),
# dependencies installed from requirements.txt with --require-hashes, nothing writable at run time.
FROM python@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 AS build
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check --no-deps --require-hashes -r /tmp/requirements.txt

FROM python@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
COPY src /app/src
COPY migrations /app/migrations
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_HOST=0.0.0.0 \
    RAG_MIGRATIONS_DIR=/app/migrations
USER 10001
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]
ENTRYPOINT ["python", "-m", "rag_chatbot"]
CMD ["serve"]
