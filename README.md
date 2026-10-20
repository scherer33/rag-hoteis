# Hotel RAG — Algarve

Retrieval-Augmented Generation over an Algarve hotel's operational procedures
(reception, F&B, kitchen, parking, maintenance), with traceable citations and
honest abstention.

> **Status:** under construction. The design and the implementation plan live in
> `docs/superpowers/`.

## Quickstart

```bash
cp .env.example .env     # fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
make build
make up
```

- UI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>
- Qdrant: <http://localhost:6333/dashboard>

## Development

```bash
make test    # pytest inside the dev container
make lint    # ruff inside the dev container
```

`uv run pytest` also works locally as a faster loop; the container is the
canonical environment.
