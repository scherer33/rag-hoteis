# RAG Hoteleiro do Algarve — Plano de Implementação

> **Para agentes implementadores:** SUB-SKILL OBRIGATÓRIA: usar `superpowers:subagent-driven-development` (recomendado) ou `superpowers:executing-plans` para implementar este plano tarefa a tarefa. Passos com checkbox (`- [ ]`) para tracking.

**Goal:** Construir um sistema RAG de portfólio em Python que responde a perguntas sobre SOPs de um hotel fictício do Algarve (PT-PT/EN) com citação rastreável, abstenção honesta e avaliação comparativa publicada.

**Architecture:** Ingestão de corpus markdown com frontmatter → chunking com offsets → embeddings locais (BGE-M3) → Qdrant; retrieval híbrido (vetorial + BM25, fusão RRF) com reranker cross-encoder; geração via LLM OpenAI-compatível (9router) com prompt grounded, citações `[n]` e marcador de abstenção; avaliação com golden dataset e métricas de retrieval (recall@k, MRR, nDCG) + RAGAS opcional; API FastAPI e UI Streamlit com citações.

**Tech Stack:** Python 3.12, `uv`, LangChain (loaders/splitters opcionais), `sentence-transformers` (BAAI/bge-m3), `qdrant-client` (modo in-memory nos testes), `rank-bm25`, `openai` SDK (base_url do 9router), FastAPI, Streamlit, pytest, ruff, Docker Compose, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-20-rag-hoteis-design.md`

## Global Constraints

- Todo o código, dados e docs vivem em `/home/phfs/Work/rag_hoteis`. Nada é escrito fora desta pasta.
- **Runtime Docker-first:** o stack corre com `docker compose up` (`qdrant`, `api`, `ui`); testes canónicos em `docker compose run --rm dev pytest`; `uv` local é loop rápido opcional. A mesma imagem serve CI e deploy (HF Spaces Docker SDK).
- **TDD obrigatório** (RED → GREEN → REFACTOR): nenhum código de produção sem teste que falha primeiro. Exceções só com aprovação do dono: ficheiros de configuração, corpus (dados), protótipos descartáveis.
- Python 3.12; gestor de dependências `uv`; lint/format `ruff`; testes `pytest`.
- Embeddings locais multilíngues: `BAAI/bge-m3` (fallback `nomic-ai/nomic-embed-text-v2-moe`).
- Reranker: `BAAI/bge-reranker-v2-m3`.
- LLM de geração: cliente OpenAI-compatível apontando para o 9router/omniroute do dono. Chave **apenas** via variável de ambiente `LLM_API_KEY`; nunca committada.
- Vector DB: Qdrant (`:memory:` nos testes unitários; Docker Compose em execução real).
- Corpus: 30–35 documentos fictícios plausíveis, 4 departamentos (`rececao`, `fnb`, `copa`, `estacionamento`) + manutenção, num único hotel; `property_id` fixo `algarve-1`.
- Golden dataset: ~30 perguntas PT-PT + 5 EN.
- Nenhum teste unitário pode depender de download de modelo, rede ou Docker: usar fakes/in-memory.
- Commits pequenos, mensagens convencionais (`feat:`, `test:`, `docs:`, `chore:`).

## Review Focus

Inputs/falhas que o spec implica e que um usuário razoável esperaria que funcionassem — cada um tem teste na tarefa dona:

1. **Pergunta em EN contra corpus PT** — retrieval e resposta devem funcionar (golden EN na Tarefa 11; teste de retriever com fake multilingual na Tarefa 9).
2. **Filtro de departamento** — pedir só `copa` não deve devolver chunks de `rececao` (Tarefa 6).
3. **Pergunta sem resposta na base** — deve abster, não inventar (Tarefa 12).
4. **Documento com duas versões (v1/v2)** — a ingestão por omissão indexa só a versão mais recente por `doc_id` (Tarefa 2).
5. **Pergunta com gíria/typo** ("o AC tá pingando") — o passo de query rewrite é aplicado antes do retrieval (Tarefa 9).

---

## Estrutura de Ficheiros

```
rag_hoteis/
├── pyproject.toml                  # deps, ruff, pytest config
├── uv.lock
├── .env.example                    # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, QDRANT_URL
├── .gitignore
├── .dockerignore
├── README.md
├── Dockerfile                      # multi-stage: base -> dev -> runtime
├── docker-compose.yml              # qdrant + api + ui + dev (+ profile gpu)
├── Makefile                        # up, test, lint, index, eval
├── config/llm.yaml                 # modelo/base_url (sem chave)
├── data/
│   ├── corpus/{rececao,fnb,copa,estacionamento,manutencao}/*.md
│   ├── golden/golden_set.jsonl
│   └── index.jsonl                 # gerado pela ingestão (gitignored)
├── src/hotel_rag/
│   ├── __init__.py
│   ├── config.py                   # Settings (pydantic-settings)
│   ├── models.py                   # Document, Chunk, RetrievedChunk, Citation, Answer
│   ├── corpus.py                   # ler frontmatter + filtrar versões
│   ├── chunking.py                 # chunk_document()
│   ├── embeddings.py               # Embedder protocol + SentenceTransformerEmbedder
│   ├── store.py                    # ChunkStore (Qdrant)
│   ├── bm25.py                     # BM25Index + reciprocal_rank_fusion()
│   ├── rerank.py                   # Reranker protocol + CrossEncoderReranker
│   ├── retrieval.py                # Retriever (vector|hybrid|hybrid_rerank)
│   ├── generation.py               # prompt, citações, abstenção
│   ├── pipeline.py                 # HotelRAG.ask()
│   └── eval/
│       ├── __init__.py
│       ├── golden.py               # GoldenItem + load_golden()
│       ├── metrics.py              # recall_at_k, reciprocal_rank, ndcg_at_k
│       └── run.py                  # runner CLI comparando configurações
├── api/main.py                     # FastAPI
├── ui/app.py                       # Streamlit
├── ui/logic.py                     # helpers testáveis (formatação/citações)
├── scripts/build_index.py          # ingestão end-to-end (CLI)
├── scripts/check_eval_threshold.py # gate de eval
└── tests/
    ├── unit/…
    └── integration/…               # marcados, excluídos por omissão
```

---

## Tarefas

### Task 1: Bootstrap do repositório + ambiente Docker

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.dockerignore`, `.env.example`, `README.md`, `config/llm.yaml`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `.github/workflows/ci.yml`, `src/hotel_rag/__init__.py`, `tests/unit/test_package.py`, `tests/unit/test_docker_assets.py`

**Interfaces:**
- Consumes: nada.
- Produces: pacote importável `hotel_rag`; `uv run pytest` e `uv run ruff check .` funcionais; stack Docker (`qdrant`, `api`, `ui`, `dev`) definido e validável com `docker compose config`; alvos `make up|test|lint|index|eval`.

> Exceção TDD aprovada: configuração/scaffold. Os testes abaixo provam que o pacote importa e que os artefactos Docker existem com a forma esperada; o resto é config.
>
> **Ruling:** no spec, `sentence-transformers` estava nas dependências base. Aqui passa a **extra `ml`**, e `ragas` a extra `eval`, para o ambiente de testes/CI não descarregar torch (~2GB). Custo se errado: um import a falhar em runtime sem o extra — mitigado pelo Dockerfile de runtime instalar `--extra ml`.

- [ ] **Step 1: Criar `pyproject.toml`**

Projeto `hotel-rag-algarve`, `requires-python = ">=3.12"`, layout `src`.
Dependências base: `pydantic>=2`, `pydantic-settings`, `python-frontmatter`, `pyyaml`, `qdrant-client`, `rank-bm25`, `openai`, `fastapi`, `uvicorn[standard]`, `streamlit`.
Extras: `ml = ["sentence-transformers>=3"]`, `eval = ["ragas>=0.2"]`.
Dev: `pytest`, `pytest-cov`, `ruff`.
Config ruff: `line-length = 100`, `target-version = "py312"`. Pytest: `testpaths = ["tests"]`, marker `integration`.

- [ ] **Step 2: Config auxiliares**

`.gitignore` (`.venv/`, `__pycache__/`, `.env`, `data/index.jsonl`, `results/`, `models/`, `*.pdf` gerados, `.superpowers/`), `.dockerignore` (`.git`, `.venv`, `models`, `data/index.jsonl`, `results`, `__pycache__`), `.env.example` (`LLM_BASE_URL=http://host.docker.internal:20128/v1`, `LLM_API_KEY=`, `LLM_MODEL=`, `QDRANT_URL=http://qdrant:6333`, `EMBED_MODEL=BAAI/bge-m3`, `HF_HOME=/models`), `config/llm.yaml` (placeholders sem segredo), `README.md` mínimo (nome + "ver docs/"), `.github/workflows/ci.yml` (checkout, `astral-sh/setup-uv`, `uv sync --all-extras`, `uv run ruff check .`, `uv run pytest -m "not integration"` — o gate Docker entra na Task 18).

- [ ] **Step 3: Escrever os testes**

`tests/unit/test_package.py`:
```python
import hotel_rag

def test_package_importable():
    assert hotel_rag.__version__ == "0.1.0"
```

`tests/unit/test_docker_assets.py`:
```python
from pathlib import Path

ROOT = Path(__file__).parents[2]

def test_docker_assets_exist():
    for name in ["Dockerfile", "docker-compose.yml", ".dockerignore", "Makefile"]:
        assert (ROOT / name).is_file(), name

def test_compose_defines_full_stack():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for service in ["qdrant:", "api:", "ui:", "dev:"]:
        assert service in text, service

def test_dockerfile_is_multistage_nonroot_and_caches_models():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.count("FROM ") >= 3
    assert "USER " in text
    assert "HF_HOME" in text
```

- [ ] **Step 4: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_package.py tests/unit/test_docker_assets.py -v`
Expected: FAIL — `hotel_rag` inexistente e ficheiros Docker ausentes.

- [ ] **Step 5: Criar `src/hotel_rag/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 6: Criar `Dockerfile` (multi-stage)**

Três estágios: `base` (python:3.12-slim, `uv`, `HF_HOME=/models`, `WORKDIR /app`, copiar `pyproject.toml`+`uv.lock`, `uv sync`), `dev` (`FROM base`, `uv sync --all-extras`, copiar fonte, `CMD ["pytest"]`), `runtime` (`FROM base`, `uv sync --no-dev --extra ml`, copiar `src/ api/ ui/ scripts/ config/`, criar utilizador não-root, `USER app`, `EXPOSE 8000 8501`, `HEALTHCHECK` a bater em `http://localhost:8000/health`, `CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]`).

- [ ] **Step 7: Criar `docker-compose.yml`**

Serviços: `qdrant` (imagem `qdrant/qdrant`, portas 6333/6334, volume `qdrant_data`, healthcheck), `api` (build target `runtime`, `env_file: .env`, depende de `qdrant` saudável, porta 8000), `ui` (mesma imagem, `streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0`, porta 8501, depende de `api`), `dev` (build target `dev`, volume `.:/app`, `models` em volume nomeado). Perfil opcional `gpu` com serviço `api-gpu` (mesma imagem, `deploy.resources.reservations.devices`).

- [ ] **Step 8: Criar `Makefile`**

Alvos: `build` (`docker compose build`), `up` (`docker compose up -d`), `down`, `test` (`docker compose run --rm dev pytest -m "not integration"`), `lint` (`docker compose run --rm dev ruff check .`), `index` (`docker compose run --rm dev python scripts/build_index.py`), `eval`, `shell`.

- [ ] **Step 9: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_package.py tests/unit/test_docker_assets.py -v && uv run ruff check .`
Expected: PASS, sem avisos. Se houver Docker instalado: `docker compose config -q` deve sair com código 0.

- [ ] **Step 10: `git init` + commit**

```bash
mise trust ~/Work/.mise.toml
git init
git add -A
git commit -m "chore: bootstrap project scaffold with docker stack"
```

---

### Task 2: Modelos de domínio

**Files:**
- Create: `src/hotel_rag/models.py`, `tests/unit/test_models.py`

**Interfaces:**
- Consumes: nada.
- Produces (usados por todas as tarefas seguintes):
  - `Dept` (StrEnum): `RECECAO="rececao"`, `FNB="fnb"`, `COPA="copa"`, `ESTACIONAMENTO="estacionamento"`, `MANUTENCAO="manutencao"`
  - `DocType` (StrEnum): `SOP="sop"`, `MANUAL="manual"`, `FICHA_TECNICA="ficha_tecnica"`, `TABELA="tabela"`, `REGISTO="registo"`, `CONTRATO="contrato"`
  - `DocMetadata(BaseModel)`: `doc_id: str`, `title: str`, `dept: Dept`, `doc_type: DocType`, `version: str`, `date: date`, `property_id: str = "algarve-1"`, `source_path: str`
  - `Document(BaseModel)`: `metadata: DocMetadata`, `text: str`
  - `Chunk(BaseModel)`: `chunk_id: str`, `doc_id: str`, `text: str`, `char_start: int`, `char_end: int`, `page: int | None = None`, `metadata: DocMetadata`
  - `RetrievedChunk(BaseModel)`: `chunk: Chunk`, `score: float`, `source: str`
  - `Citation(BaseModel)`: `doc_id: str`, `title: str`, `dept: Dept`, `page: int | None`, `snippet: str`, `char_start: int`, `char_end: int`
  - `Answer(BaseModel)`: `text: str`, `citations: list[Citation]`, `abstained: bool`, `confidence: float`

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_models.py
from datetime import date
import pytest
from pydantic import ValidationError
from hotel_rag.models import Dept, DocType, DocMetadata, Chunk

def test_dept_values_match_corpus_folders():
    assert [d.value for d in Dept] == ["rececao", "fnb", "copa", "estacionamento", "manutencao"]

def test_metadata_rejects_unknown_dept():
    with pytest.raises(ValidationError):
        DocMetadata(doc_id="x", title="t", dept="spa", doc_type="sop",
                    version="1.0", date=date(2026, 1, 1), source_path="a.md")

def test_chunk_keeps_offsets_consistent():
    meta = DocMetadata(doc_id="x", title="t", dept=Dept.COPA, doc_type=DocType.SOP,
                       version="1.0", date=date(2026, 1, 1), source_path="a.md")
    c = Chunk(chunk_id="x:0", doc_id="x", text="abc", char_start=0, char_end=3, metadata=meta)
    assert c.char_end - c.char_start == len(c.text)
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: FAIL — módulo `hotel_rag.models` inexistente.

- [ ] **Step 3: Implementar `models.py`**

Definir os enums e modelos acima com `pydantic.BaseModel` (o `Chunk` não valida offsets automaticamente; a consistência é garantida em `chunking.py` e testada aqui como contrato).

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/models.py tests/unit/test_models.py
git commit -m "feat: add domain models"
```

---

### Task 3: Carregamento de corpus (frontmatter + versões)

**Files:**
- Create: `src/hotel_rag/corpus.py`, `tests/unit/test_corpus.py`

**Interfaces:**
- Consumes: `Document`, `DocMetadata`, `Dept`, `DocType` (Task 2).
- Produces:
  - `load_document(path: Path) -> Document`
  - `load_corpus(root: Path, latest_only: bool = True) -> list[Document]`
  - Regra de versão: para o mesmo `doc_id`, `latest_only=True` mantém apenas a maior `version` (comparação por `packaging.version.Version` ou tupla de ints; documentar no docstring).

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_corpus.py
from pathlib import Path
from hotel_rag.corpus import load_document, load_corpus
from hotel_rag.models import Dept

FM = """---
doc_id: checkin
title: Check-in padrão
dept: rececao
doc_type: sop
version: "1.0"
date: 2026-01-10
---

# Check-in
Texto.
"""

def _write(tmp_path: Path, name: str, fm: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(fm, encoding="utf-8")
    return p

def test_load_document_reads_frontmatter(tmp_path):
    doc = load_document(_write(tmp_path, "a.md", FM))
    assert doc.metadata.dept is Dept.RECECAO
    assert doc.metadata.property_id == "algarve-1"
    assert "Check-in" in doc.text

def test_load_corpus_keeps_only_latest_version(tmp_path):
    old = FM.replace('version: "1.0"', 'version: "1.0"')
    new = FM.replace('version: "1.0"', 'version: "2.0"')
    _write(tmp_path, "v1.md", old); _write(tmp_path, "v2.md", new)
    docs = load_corpus(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata.version == "2.0"

def test_load_corpus_can_include_all_versions(tmp_path):
    _write(tmp_path, "v1.md", FM)
    _write(tmp_path, "v2.md", FM.replace('version: "1.0"', 'version: "2.0"'))
    assert len(load_corpus(tmp_path, latest_only=False)) == 2
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: hotel_rag.corpus`.

- [ ] **Step 3: Implementar `corpus.py`**

Usar `frontmatter.load()`; converter campos para `DocMetadata`; `load_corpus` faz `rglob("*.md")`, agrupa por `doc_id` e, se `latest_only`, fica com a maior versão (função `_version_key(v: str) -> tuple[int, ...]`).

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_corpus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/corpus.py tests/unit/test_corpus.py
git commit -m "feat: load corpus with frontmatter and version filtering"
```

---

### Task 4: Chunking com offsets

**Files:**
- Create: `src/hotel_rag/chunking.py`, `tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `Document`, `Chunk` (Task 2).
- Produces:
  - `chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]`
  - Garantias: cada `Chunk.text == doc.text[char_start:char_end]`; `chunk_id == f"{doc.doc_id}:{char_start}"`; chunks consecutivos sobrepõem-se em `overlap` caracteres (quando o texto é maior que `size`); nenhum chunk vazio; cortes preferem fim de parágrafo/frase dentro da janela.

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_chunking.py
from datetime import date
from hotel_rag.chunking import chunk_document
from hotel_rag.models import Document, DocMetadata, Dept, DocType

def _doc(text: str) -> Document:
    meta = DocMetadata(doc_id="d1", title="t", dept=Dept.SOP if False else Dept.RECECAO,
                       doc_type=DocType.SOP, version="1.0", date=date(2026, 1, 1),
                       source_path="a.md")
    return Document(metadata=meta, text=text)

def test_short_document_yields_single_chunk():
    chunks = chunk_document(_doc("abc"), size=100, overlap=10)
    assert len(chunks) == 1 and chunks[0].text == "abc"

def test_offsets_slice_back_to_original_text():
    text = "palavra " * 200
    chunks = chunk_document(_doc(text), size=100, overlap=20)
    assert all(text[c.char_start:c.char_end] == c.text for c in chunks)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)

def test_consecutive_chunks_overlap():
    text = "x" * 500
    chunks = chunk_document(_doc(text), size=100, overlap=20)
    assert chunks[1].char_start == chunks[0].char_end - 20

def test_no_empty_chunks():
    chunks = chunk_document(_doc("a\n\n" * 300), size=50, overlap=10)
    assert all(c.text.strip() for c in chunks)
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar `chunk_document`**

Janela deslizante de `size` caracteres com passo `size - overlap`; ajustar o fim para o último `\n\n` ou `. ` dentro da janela (se existir após metade da janela); `chunk_id` e offsets como acima.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/chunking.py tests/unit/test_chunking.py
git commit -m "feat: chunk documents preserving char offsets"
```

---

### Task 5: Corpus fictício do hotel (dados)

**Files:**
- Create: `data/corpus/rececao/*.md` (8), `data/corpus/fnb/*.md` (8), `data/corpus/copa/*.md` (9), `data/corpus/estacionamento/*.md` (6), `data/corpus/manutencao/*.md` (4), `tests/unit/test_corpus_quality.py`

**Interfaces:**
- Consumes: `load_corpus`, `Dept`, `DocType` (Tasks 2–3).
- Produces: 30–35 documentos válidos; pelo menos 1 par v1/v2 (`checkin`), 1 documento em EN (`evacuacao-en`/`fire-evacuation`), e documentos do tipo `TABELA` com tabelas markdown.

> Documentos são dados (exceção TDD aprovada). O teste abaixo valida a qualidade do dataset e é escrito primeiro.

**Lista mínima (ficheiros e `doc_id`):**

| Dept | Ficheiros |
|---|---|
| rececao | `checkin-v1.md` (v1.0, checkin), `checkin-v2.md` (v2.0), `checkout.md`, `reclamacoes.md`, `overbooking.md`, `handover-turno.md`, `vip.md`, `horarios-servicos.md` (TABELA) |
| fnb | `abertura-fecho.md`, `servico-mesa.md`, `reservas.md`, `bar-cocktails.md` (TABELA), `banquetes.md`, `alergenios.md` (TABELA), `queixas-fnb.md`, `stock-bebidas.md` |
| copa | `haccp-temperaturas.md` (TABELA), `plano-limpeza.md` (TABELA), `forno-combinado.md` (FICHA_TECNICA), `maquina-louca.md` (FICHA_TECNICA), `inventario-fifo.md`, `pragas.md`, `rastreabilidade.md`, `abertura-cozinha.md`, `residuos.md` |
| estacionamento | `valet.md`, `capacidade.md`, `rondas.md`, `objetos-perdidos.md`, `carregamento-ev.md`, `dano-viatura.md` |
| manutencao | `ac-erro-e3.md` (MANUAL), `piscina-quimica.md`, `caldeira-semanal.md`, `elevador-emergencia.md` |

Cada documento: frontmatter completo (`doc_id`, `title`, `dept`, `doc_type`, `version`, `date`) + 40–120 linhas de conteúdo plausível (procedimentos numerados, responsáveis, prazos). Conteúdo em PT-PT; `fire-evacuation.md` em EN dentro de `rececao`.

- [ ] **Step 1: Escrever o teste de qualidade do dataset**

```python
# tests/unit/test_corpus_quality.py
from pathlib import Path
from collections import Counter
from hotel_rag.corpus import load_corpus
from hotel_rag.models import Dept, DocType

CORPUS = Path(__file__).parents[2] / "data" / "corpus"

def test_corpus_has_expected_size_and_departments():
    docs = load_corpus(CORPUS)
    assert 30 <= len(docs) <= 35
    assert {d.metadata.dept for d in docs} == {
        Dept.RECECAO, Dept.FNB, Dept.COPA, Dept.ESTACIONAMENTO, Dept.MANUTENCAO}

def test_every_document_has_substance():
    for doc in load_corpus(CORPUS):
        assert len(doc.text.splitlines()) >= 30, doc.metadata.doc_id

def test_corpus_has_versioned_pair_table_and_english_doc():
    all_docs = load_corpus(CORPUS, latest_only=False)
    versions = Counter(d.metadata.doc_id for d in all_docs)
    assert max(versions.values()) == 2
    assert any(d.metadata.doc_type is DocType.TABELA for d in all_docs)
    assert any("evacuation" in d.text.lower() for d in all_docs)

def test_chunk_ids_unique_per_document():
    docs = load_corpus(CORPUS)
    assert len({d.metadata.doc_id for d in docs}) == len(docs)
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_corpus_quality.py -v`
Expected: FAIL — corpus vazio (assert de tamanho).

- [ ] **Step 3: Escrever os documentos**

Criar os ficheiros da tabela acima. Não copiar textos reais de manuais proprietários; escrever conteúdo original plausível.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_corpus_quality.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/corpus tests/unit/test_corpus_quality.py
git commit -m "data: add fictional Algarve hotel corpus"
```

---

### Task 6: Embedder (protocolo + implementações)

**Files:**
- Create: `src/hotel_rag/embeddings.py`, `tests/unit/test_embeddings.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `Embedder(Protocol)`: `embed_documents(texts: list[str]) -> list[list[float]]`; `embed_query(text: str) -> list[float]`
  - `FakeEmbedder` (em `tests/fakes.py`): determinístico, dimensão 8, útil para todos os testes.
  - `SentenceTransformerEmbedder(model_name: str = "BAAI/bge-m3")` com `dimension: int` (lazy load no primeiro uso).

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_embeddings.py
from tests.fakes import FakeEmbedder

def test_fake_embedder_is_deterministic():
    e = FakeEmbedder()
    assert e.embed_documents(["a"]) == e.embed_documents(["a"])
    assert e.embed_query("a") == e.embed_documents(["a"])[0]

def test_fake_embedder_vectors_have_fixed_dimension():
    e = FakeEmbedder()
    assert len(e.embed_query("texto")) == 8
    assert all(len(v) == 8 for v in e.embed_documents(["a", "b"]))
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: FAIL — `tests.fakes` inexistente.

- [ ] **Step 3: Implementar**

`src/hotel_rag/embeddings.py`: `Protocol` + `SentenceTransformerEmbedder` (import de `sentence_transformers` dentro de `__init__` para não carregar em testes). `tests/fakes.py`: `FakeEmbedder` gera vetor via hash estável (ex.: SHA-256 do texto → 8 floats normalizados).

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/embeddings.py tests/fakes.py tests/unit/test_embeddings.py
git commit -m "feat: add embedder protocol with fake and sentence-transformers impl"
```

---

### Task 7: Store Qdrant (in-memory nos testes)

**Files:**
- Create: `src/hotel_rag/store.py`, `tests/unit/test_store.py`

**Interfaces:**
- Consumes: `Chunk`, `RetrievedChunk`, `Dept` (Task 2).
- Produces:
  - `ChunkStore(url: str = ":memory:", collection: str = "hotel_docs", vector_size: int = 1024)`
  - `upsert(chunks: list[Chunk], vectors: list[list[float]]) -> None`
  - `search(vector: list[float], k: int = 10, dept: Dept | None = None) -> list[RetrievedChunk]` — devolve `source="vector"`, score de similaridade, ordenado desc.
  - `count() -> int`
  - Payload guarda `doc_id`, `dept`, `title`, `doc_type`, `version`, `text`, `char_start`, `char_end`.

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_store.py
from datetime import date
from hotel_rag.models import Chunk, DocMetadata, Dept, DocType
from hotel_rag.store import ChunkStore

def _chunk(i: int, dept: Dept) -> Chunk:
    meta = DocMetadata(doc_id=f"d{i}", title=f"t{i}", dept=dept, doc_type=DocType.SOP,
                       version="1.0", date=date(2026, 1, 1), source_path="a.md")
    return Chunk(chunk_id=f"d{i}:0", doc_id=f"d{i}", text=f"texto {i}",
                 char_start=0, char_end=8, metadata=meta)

def _vec(i: int) -> list[float]:
    v = [0.0] * 8
    v[i % 8] = 1.0
    return v

def test_upsert_and_count():
    s = ChunkStore(vector_size=8)
    s.upsert([_chunk(0, Dept.COPA)], [_vec(0)])
    assert s.count() == 1

def test_search_returns_most_similar_first():
    s = ChunkStore(vector_size=8)
    chunks = [_chunk(0, Dept.COPA), _chunk(1, Dept.RECECAO)]
    s.upsert(chunks, [_vec(0), _vec(1)])
    hits = s.search(_vec(0), k=2)
    assert hits[0].chunk.doc_id == "d0"
    assert hits[0].source == "vector"

def test_search_filters_by_department():
    s = ChunkStore(vector_size=8)
    s.upsert([_chunk(0, Dept.COPA), _chunk(1, Dept.RECECAO)], [_vec(0), _vec(1)])
    hits = s.search(_vec(1), k=5, dept=Dept.COPA)
    assert [h.chunk.metadata.dept for h in hits] == [Dept.COPA]
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_store.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar `ChunkStore`**

`qdrant_client.QdrantClient(location=url)` com `:memory:`; coleção `Distance.COSINE`; filtro `FieldCondition(key="dept", match=MatchValue(value=dept.value))`; reconstruir `Chunk` a partir do payload.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/store.py tests/unit/test_store.py
git commit -m "feat: add Qdrant chunk store with department filtering"
```

---

### Task 8: BM25 + fusão RRF

**Files:**
- Create: `src/hotel_rag/bm25.py`, `tests/unit/test_bm25.py`

**Interfaces:**
- Consumes: `Chunk`, `RetrievedChunk`, `Dept` (Task 2).
- Produces:
  - `BM25Index(chunks: list[Chunk])` com `search(query: str, k: int = 10, dept: Dept | None = None) -> list[RetrievedChunk]` (`source="bm25"`).
  - `reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]` — soma `1/(k+rank)` por `chunk_id`, devolve `source="hybrid"` com score fundido.

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_bm25.py
from datetime import date
from hotel_rag.bm25 import BM25Index, reciprocal_rank_fusion
from hotel_rag.models import Chunk, DocMetadata, Dept, DocType, RetrievedChunk

def _chunk(cid: str, text: str, dept=Dept.COPA) -> Chunk:
    meta = DocMetadata(doc_id=cid.split(":")[0], title=cid, dept=dept, doc_type=DocType.SOP,
                       version="1.0", date=date(2026, 1, 1), source_path="a.md")
    return Chunk(chunk_id=cid, doc_id=meta.doc_id, text=text, char_start=0,
                 char_end=len(text), metadata=meta)

def test_bm25_finds_exact_keyword():
    idx = BM25Index([_chunk("a:0", "temperatura da camara fria do peixe"),
                     _chunk("b:0", "procedimento de check-in do hospede")])
    hits = idx.search("camara fria", k=1)
    assert hits[0].chunk.chunk_id == "a:0"

def test_bm25_respects_department_filter():
    idx = BM25Index([_chunk("a:0", "camara fria", Dept.COPA),
                     _chunk("b:0", "camara fria", Dept.RECECAO)])
    assert [h.chunk.metadata.dept for h in idx.search("camara fria", k=5, dept=Dept.RECECAO)] == [Dept.RECECAO]

def test_rrf_merges_two_rankings():
    c1 = _chunk("a:0", "x"); c2 = _chunk("b:0", "y")
    h1 = [RetrievedChunk(chunk=c1, score=1.0, source="vector")]
    h2 = [RetrievedChunk(chunk=c2, score=1.0, source="bm25"),
          RetrievedChunk(chunk=c1, score=0.5, source="bm25")]
    fused = reciprocal_rank_fusion([h1, h2])
    assert [h.chunk.chunk_id for h in fused] == ["a:0", "b:0"]
    assert fused[0].source == "hybrid"
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_bm25.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar**

Tokenização: `re.findall(r"\w+", text.lower())` (suficiente para PT/EN). `rank_bm25.BM25Okapi`; filtrar por dept antes de pontuar; `k` resultados.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_bm25.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/bm25.py tests/unit/test_bm25.py
git commit -m "feat: add BM25 index and reciprocal rank fusion"
```

---

### Task 9: Reranker + Retriever (vector | hybrid | hybrid_rerank)

**Files:**
- Create: `src/hotel_rag/rerank.py`, `src/hotel_rag/retrieval.py`, `tests/unit/test_retrieval.py`

**Interfaces:**
- Consumes: `Embedder`, `ChunkStore`, `BM25Index`, `reciprocal_rank_fusion` (Tasks 6–8).
- Produces:
  - `Reranker(Protocol)`: `rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]`
  - `CrossEncoderReranker(model_name: str = "BAAI/bge-reranker-v2-m3")` (lazy import).
  - `Retriever(embedder, store, bm25, reranker=None, mode="hybrid_rerank", k_candidates=20)` com `retrieve(query: str, top_k: int = 5, dept: Dept | None = None, rewrite: Callable[[str], str] | None = None) -> list[RetrievedChunk]`.
  - `rewrite` é aplicado à query antes de qualquer retrieval quando fornecido (gancho do query rewriting; o LLM real é injetado na Task 13).

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_retrieval.py
from datetime import date
import pytest
from hotel_rag.bm25 import BM25Index
from hotel_rag.models import Chunk, DocMetadata, Dept, DocType, RetrievedChunk
from hotel_rag.retrieval import Retriever

def _chunk(cid, text):
    meta = DocMetadata(doc_id=cid.split(":")[0], title=cid, dept=Dept.COPA, doc_type=DocType.SOP,
                       version="1.0", date=date(2026, 1, 1), source_path="a.md")
    return Chunk(chunk_id=cid, doc_id=meta.doc_id, text=text, char_start=0,
                 char_end=len(text), metadata=meta)

class StubStore:
    def __init__(self, hits): self.hits = hits
    def search(self, vector, k=10, dept=None): return self.hits[:k]

class StubEmbedder:
    def embed_query(self, text): return [0.0] * 8
    def embed_documents(self, texts): return [[0.0] * 8 for _ in texts]

class ReverseReranker:
    def rerank(self, query, chunks, top_k):
        return list(reversed(chunks))[:top_k]

@pytest.mark.parametrize("mode", ["vector", "hybrid", "hybrid_rerank"])
def test_all_modes_return_top_k(mode):
    c = _chunk("a:0", "camara fria")
    r = Retriever(StubEmbedder(), StubStore([RetrievedChunk(chunk=c, score=1.0, source="vector")]),
                  BM25Index([c]), reranker=ReverseReranker(), mode=mode)
    assert len(r.retrieve("camara fria", top_k=1)) == 1

def test_rewrite_is_applied_before_retrieval():
    seen = {}
    class RecordingEmbedder(StubEmbedder):
        def embed_query(self, text): seen["q"] = text; return [0.0] * 8
    c = _chunk("a:0", "x")
    r = Retriever(RecordingEmbedder(), StubStore([RetrievedChunk(chunk=c, score=1.0, source="vector")]),
                  BM25Index([c]), mode="vector")
    r.retrieve("o AC ta pingando", rewrite=lambda q: "ar condicionado a pingar água")
    assert seen["q"] == "ar condicionado a pingar água"
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_retrieval.py -v`
Expected: FAIL — módulos inexistentes.

- [ ] **Step 3: Implementar**

`retrieval.py`: aplicar `rewrite`; `vector` → `store.search`; `hybrid` → RRF(vector, bm25); `hybrid_rerank` → RRF + `reranker.rerank`. `rerank.py`: protocolo + cross-encoder (`sentence_transformers.CrossEncoder`, lazy) ordenando por score desc.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_retrieval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/rerank.py src/hotel_rag/retrieval.py tests/unit/test_retrieval.py
git commit -m "feat: add hybrid retriever with optional reranking and query rewrite"
```

---

### Task 10: Métricas de retrieval

**Files:**
- Create: `src/hotel_rag/eval/__init__.py`, `src/hotel_rag/eval/metrics.py`, `tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: nada (funções puras).
- Produces:
  - `recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float`
  - `reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float`
  - `ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float`
  - Todas devolvem `0.0` quando `relevant_doc_ids` é vazio (documentado).

- [ ] **Step 1: Escrever testes (valores calculados à mão)**

```python
# tests/unit/test_metrics.py
import math
from hotel_rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k([], {"a"}, 5) == 0.0

def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0

def test_ndcg_at_k():
    # DCG = 1/log2(2) + 1/log2(4); IDCG com 2 relevantes = 1/log2(2) + 1/log2(3)
    expected = (1.0 + 1 / math.log2(4)) / (1.0 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == expected
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar**

Implementações diretas de IR clássico; deduplicar `retrieved_doc_ids` preservando a primeira ocorrência (documentar).

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/eval tests/unit/test_metrics.py
git commit -m "feat: add retrieval metrics (recall@k, MRR, nDCG)"
```

---

### Task 11: Golden dataset + loader

**Files:**
- Create: `src/hotel_rag/eval/golden.py`, `data/golden/golden_set.jsonl`, `tests/unit/test_golden.py`

**Interfaces:**
- Consumes: `Dept` (Task 2); `doc_id`s do corpus (Task 5).
- Produces:
  - `GoldenItem(BaseModel)`: `question: str`, `answer: str`, `relevant_doc_ids: list[str]`, `language: Literal["pt", "en"]`, `dept: Dept`
  - `load_golden(path: Path) -> list[GoldenItem]` (uma linha JSON por item).

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_golden.py
import json
from pathlib import Path
from hotel_rag.eval.golden import load_golden

def test_load_golden_parses_jsonl(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"question": "q", "answer": "a", "relevant_doc_ids": ["checkin"],
                             "language": "pt", "dept": "rececao"}) + "\n", encoding="utf-8")
    items = load_golden(p)
    assert items[0].relevant_doc_ids == ["checkin"]

def test_golden_dataset_is_complete_and_multilingual():
    items = load_golden(Path(__file__).parents[2] / "data" / "golden" / "golden_set.jsonl")
    assert len(items) >= 30
    assert sum(i.language == "en" for i in items) >= 5
    assert all(i.relevant_doc_ids for i in items)
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_golden.py -v`
Expected: FAIL — módulo/ficheiro inexistente.

- [ ] **Step 3: Implementar loader + escrever 30 perguntas**

Cobrir os 5 departamentos (≥4 por dept), 5 em EN, incluindo: 1 pergunta cuja resposta está só na v2 do checkin, 1 pergunta de tabela (temperaturas/alergénios), 1 pergunta de registo/manutenção. `relevant_doc_ids` usa os `doc_id` reais.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_golden.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/eval/golden.py data/golden tests/unit/test_golden.py
git commit -m "feat: add golden dataset loader and evaluation questions"
```

---

### Task 12: Geração com citações e abstenção

**Files:**
- Create: `src/hotel_rag/generation.py`, `tests/unit/test_generation.py`

**Interfaces:**
- Consumes: `RetrievedChunk`, `Citation`, `Answer` (Task 2).
- Produces:
  - `LLMClient(Protocol)`: `complete(prompt: str) -> str`
  - `ABSTAIN_MARKER = "ABSTAIN"`
  - `build_prompt(question: str, chunks: list[RetrievedChunk]) -> str` — contexto numerado `[1]..[n]` com `title`, `dept`, `doc_id`; instrução explícita: responder só com base no contexto, citar `[n]`, e responder exatamente `ABSTAIN` se o contexto não contiver a resposta.
  - `parse_citations(text: str, chunks: list[RetrievedChunk]) -> list[Citation]` — extrai `[n]`, deduplica, ignora índices inválidos; `snippet` = primeiros 200 chars do chunk.
  - `AnswerGenerator(llm: LLMClient, min_score: float = 0.3)` com `generate(question: str, chunks: list[RetrievedChunk]) -> Answer`: abstém se `not chunks`, se `chunks[0].score < min_score` ou se a resposta contém `ABSTAIN`; `confidence` = `max(0.0, chunks[0].score)` (ou `0.0` se abstém).

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_generation.py
from datetime import date
from hotel_rag.generation import AnswerGenerator, build_prompt, parse_citations
from hotel_rag.models import Chunk, DocMetadata, Dept, DocType, RetrievedChunk

def _hit(cid="a:0", text="Procedimento: desligar o AC.", score=0.9):
    meta = DocMetadata(doc_id=cid.split(":")[0], title="AC erro E3", dept=Dept.MANUTENCAO,
                       doc_type=DocType.MANUAL, version="1.0", date=date(2026, 1, 1),
                       source_path="a.md")
    c = Chunk(chunk_id=cid, doc_id=meta.doc_id, text=text, char_start=0,
              char_end=len(text), metadata=meta)
    return RetrievedChunk(chunk=c, score=score, source="hybrid_rerank")

class FakeLLM:
    def __init__(self, out): self.out = out
    def complete(self, prompt): return self.out

def test_prompt_contains_numbered_context_and_abstain_rule():
    p = build_prompt("o que fazer no erro E3?", [_hit()])
    assert "[1]" in p and "AC erro E3" in p and "ABSTAIN" in p

def test_citations_map_back_to_chunks():
    hits = [_hit()]
    cites = parse_citations("Desligar o AC [1].", hits)
    assert cites[0].doc_id == "a" and cites[0].dept is Dept.MANUTENCAO

def test_invalid_citation_index_is_ignored():
    assert parse_citations("Ver [9].", [_hit()]) == []

def test_generator_abstains_without_chunks():
    a = AnswerGenerator(FakeLLM("qualquer coisa")).generate("q", [])
    assert a.abstained and a.citations == [] and a.confidence == 0.0

def test_generator_abstains_on_marker():
    a = AnswerGenerator(FakeLLM("ABSTAIN")).generate("q", [_hit()])
    assert a.abstained and a.text == "ABSTAIN"

def test_generator_abstains_below_score_threshold():
    a = AnswerGenerator(FakeLLM("resposta [1]"), min_score=0.8).generate("q", [_hit(score=0.2)])
    assert a.abstained

def test_generator_returns_answer_with_citations():
    a = AnswerGenerator(FakeLLM("Desligar o AC [1].")).generate("q", [_hit()])
    assert not a.abstained and a.citations[0].title == "AC erro E3"
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_generation.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar**

`openai`-compatible client numa tarefa separada (Task 13) — aqui só o protocolo e a lógica pura.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_generation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/generation.py tests/unit/test_generation.py
git commit -m "feat: add grounded answer generation with citations and abstention"
```

---

### Task 13: Cliente LLM (9router) + pipeline end-to-end

**Files:**
- Create: `src/hotel_rag/config.py`, `src/hotel_rag/llm.py`, `src/hotel_rag/pipeline.py`, `tests/unit/test_pipeline.py`, `tests/integration/test_pipeline_inmemory.py`

**Interfaces:**
- Consumes: `Retriever`, `AnswerGenerator`, `LLMClient` (Tasks 9, 12).
- Produces:
  - `Settings` (pydantic-settings): `llm_base_url`, `llm_api_key`, `llm_model`, `qdrant_url`, `embed_model`, `abstain_min_score`.
  - `OpenAICompatClient(base_url: str, api_key: str, model: str, temperature: float = 0.0)` com `complete(prompt: str) -> str`.
  - `HotelRAG(retriever, generator, rewriter: Callable[[str], str] | None = None)` com `ask(question: str, top_k: int = 5, dept: Dept | None = None) -> Answer`.

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_pipeline.py
from datetime import date
from hotel_rag.models import Chunk, DocMetadata, Dept, DocType, RetrievedChunk
from hotel_rag.pipeline import HotelRAG
from hotel_rag.generation import AnswerGenerator

class StubRetriever:
    def __init__(self, hits): self.hits = hits
    def retrieve(self, query, top_k=5, dept=None, rewrite=None):
        if rewrite: query = rewrite(query)
        self.last_query = query
        return self.hits

class FakeLLM:
    def complete(self, prompt): return "Faça X [1]."

def _hit():
    meta = DocMetadata(doc_id="a", title="SOP", dept=Dept.RECECAO, doc_type=DocType.SOP,
                       version="1.0", date=date(2026, 1, 1), source_path="a.md")
    c = Chunk(chunk_id="a:0", doc_id="a", text="Faça X.", char_start=0, char_end=7, metadata=meta)
    return RetrievedChunk(chunk=c, score=0.9, source="hybrid")

def test_pipeline_returns_answer_with_citations():
    rag = HotelRAG(StubRetriever([_hit()]), AnswerGenerator(FakeLLM()))
    a = rag.ask("como fazer X?")
    assert not a.abstained and a.citations[0].doc_id == "a"

def test_pipeline_passes_rewriter_to_retriever():
    retriever = StubRetriever([_hit()])
    rag = HotelRAG(retriever, AnswerGenerator(FakeLLM()), rewriter=lambda q: q.upper())
    rag.ask("como fazer x?")
    assert retriever.last_query == "COMO FAZER X?"
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_pipeline.py -v`
Expected: FAIL — módulos inexistentes.

- [ ] **Step 3: Implementar**

`config.py` lê `.env`; `llm.py` usa `openai.OpenAI(base_url=..., api_key=...)` e `chat.completions.create(model=..., temperature=0, messages=[...])`, devolvendo `.choices[0].message.content`; `pipeline.py` orquestra como nos testes. Teste de integração (`@pytest.mark.integration`) monta corpus real + `FakeEmbedder` + `ChunkStore(:memory:)` + `BM25Index` + `FakeLLM` e verifica resposta com citação de um `doc_id` real do corpus.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_pipeline.py tests/integration/test_pipeline_inmemory.py -v -m "not integration"` e depois com `-m integration`
Expected: PASS em ambos.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/config.py src/hotel_rag/llm.py src/hotel_rag/pipeline.py tests/
git commit -m "feat: add OpenAI-compatible LLM client and end-to-end pipeline"
```

---

### Task 14: Script de ingestão (`build_index`)

**Files:**
- Create: `scripts/build_index.py`, `tests/unit/test_build_index.py`

**Interfaces:**
- Consumes: `load_corpus`, `chunk_document`, `SentenceTransformerEmbedder`, `ChunkStore`, `BM25Index`.
- Produces: `build_index(corpus_dir: Path, store: ChunkStore, embedder: Embedder) -> int` (devolve nº de chunks); CLI `python scripts/build_index.py --corpus data/corpus --qdrant-url http://localhost:6333`.

- [ ] **Step 1: Escrever teste**

```python
# tests/unit/test_build_index.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_index import build_index
from hotel_rag.store import ChunkStore
from tests.fakes import FakeEmbedder

def test_build_index_populates_store(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ndoc_id: a\ntitle: A\ndept: copa\ndoc_type: sop\nversion: '1.0'\n"
        "date: 2026-01-01\n---\n\nTexto de teste.\n", encoding="utf-8")
    store = ChunkStore(vector_size=8)
    assert build_index(tmp_path, store, FakeEmbedder()) >= 1
    assert store.count() >= 1
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_build_index.py -v`
Expected: FAIL — `scripts/build_index.py` inexistente.

- [ ] **Step 3: Implementar**

Carregar → chunkar com `size=800, overlap=120` → embeddar em lotes → `store.upsert`. CLI com `argparse` e retorno de contagem.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_build_index.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_index.py tests/unit/test_build_index.py
git commit -m "feat: add index build script"
```

---

### Task 15: Eval runner comparando configurações

**Files:**
- Create: `src/hotel_rag/eval/run.py`, `tests/unit/test_eval_run.py`, `results/.gitkeep`

**Interfaces:**
- Consumes: `load_golden`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `Retriever`.
- Produces:
  - `evaluate(retriever: Retriever, items: list[GoldenItem], k: int = 5) -> dict[str, float]` com chaves `recall@k`, `mrr`, `ndcg@k`.
  - `compare(configs: dict[str, Retriever], items: list[GoldenItem]) -> dict[str, dict[str, float]]`
  - `to_markdown(results: dict[str, dict[str, float]]) -> str` (tabela).
  - CLI `python -m hotel_rag.eval.run --configs vector,hybrid,hybrid_rerank --out results/latest.json` (escreve JSON + tabela Markdown).

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_eval_run.py
from hotel_rag.eval.run import evaluate, to_markdown
from hotel_rag.eval.golden import GoldenItem
from hotel_rag.models import Dept

class FixedRetriever:
    def __init__(self, doc_ids): self.doc_ids = doc_ids
    def retrieve(self, query, top_k=5, dept=None, rewrite=None):
        from tests.fakes import make_retrieved
        return [make_retrieved(d, 1.0) for d in self.doc_ids[:top_k]]

ITEMS = [GoldenItem(question="q", answer="a", relevant_doc_ids=["a"],
                    language="pt", dept=Dept.COPA)]

def test_evaluate_perfect_retrieval_scores_one():
    scores = evaluate(FixedRetriever(["a"]), ITEMS, k=5)
    assert scores["recall@k"] == 1.0 and scores["mrr"] == 1.0

def test_evaluate_miss_scores_zero():
    scores = evaluate(FixedRetriever(["z"]), ITEMS, k=5)
    assert scores["recall@k"] == 0.0

def test_to_markdown_contains_columns_and_rows():
    md = to_markdown({"vector": {"recall@k": 0.5, "mrr": 0.4, "ndcg@k": 0.45}})
    assert "| config | recall@k | mrr | ndcg@k |" in md and "vector" in md
```

> Adicionar `make_retrieved(doc_id, score)` em `tests/fakes.py` (constrói `RetrievedChunk` mínimo).

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_eval_run.py -v`
Expected: FAIL — `hotel_rag.eval.run` inexistente.

- [ ] **Step 3: Implementar**

`evaluate` percorre items, chama `retriever.retrieve(question, top_k=k)`, recolhe `doc_id` únicos e calcula médias das três métricas. CLI constrói os 3 retrievers reais (bge-m3 + Qdrant + BM25 + reranker) a partir do índice.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_eval_run.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotel_rag/eval/run.py tests/unit/test_eval_run.py tests/fakes.py results/.gitkeep
git commit -m "feat: add evaluation runner comparing retrieval configs"
```

---

### Task 16: API FastAPI

**Files:**
- Create: `api/__init__.py`, `api/main.py`, `tests/unit/test_api.py`

**Interfaces:**
- Consumes: `Answer`, `Dept`, `HotelRAG`.
- Produces:
  - `create_app(rag: HotelRAG) -> FastAPI`
  - `POST /answer` body `{question: str, top_k: int = 5, dept: str | null}` → `Answer` (JSON com `text`, `citations`, `abstained`, `confidence`)
  - `GET /health` → `{"status": "ok"}`

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_api.py
from fastapi.testclient import TestClient
from api.main import create_app
from hotel_rag.models import Answer, Citation, Dept

class StubRAG:
    def ask(self, question, top_k=5, dept=None):
        return Answer(text="Faça X [1].", confidence=0.9, abstained=False,
                      citations=[Citation(doc_id="a", title="SOP", dept=Dept.RECECAO,
                                          page=None, snippet="Faça X.", char_start=0, char_end=7)])

def test_health():
    c = TestClient(create_app(StubRAG()))
    assert c.get("/health").json() == {"status": "ok"}

def test_answer_returns_citations():
    c = TestClient(create_app(StubRAG()))
    r = c.post("/answer", json={"question": "como fazer X?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["doc_id"] == "a" and body["abstained"] is False

def test_answer_rejects_empty_question():
    c = TestClient(create_app(StubRAG()))
    assert c.post("/answer", json={"question": ""}).status_code == 422
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_api.py -v`
Expected: FAIL — `api.main` inexistente.

- [ ] **Step 3: Implementar**

`AskRequest(BaseModel)` com `question: str = Field(min_length=1)`; `create_app` injeta o `HotelRAG`; em `__main__` construir o pipeline real a partir de env.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api tests/unit/test_api.py
git commit -m "feat: add FastAPI answer endpoint"
```

---

### Task 17: UI Streamlit com citações

**Files:**
- Create: `ui/logic.py`, `ui/app.py`, `tests/unit/test_ui_logic.py`, `tests/unit/test_ui_app.py`

**Interfaces:**
- Consumes: `Answer`, `HotelRAG`.
- Produces:
  - `citation_label(c: Citation) -> str` — ex. `"[1] AC erro E3 · manutencao · pág. 3"` (omite página quando `None`).
  - `citation_markdown(c: Citation) -> str` — bloco com snippet em code fence.
  - `app: streamlit` executa com `st.chat_input`/`st.chat_message`, mostra `answer.text` e, por citação, `st.expander` com `citation_markdown`.

- [ ] **Step 1: Escrever testes**

```python
# tests/unit/test_ui_logic.py
from hotel_rag.models import Citation, Dept
from ui.logic import citation_label, citation_markdown

def test_label_includes_dept_and_page():
    c = Citation(doc_id="a", title="AC erro E3", dept=Dept.MANUTENCAO, page=3,
                 snippet="x", char_start=0, char_end=1)
    assert citation_label(c) == "[1] AC erro E3 · manutencao · pág. 3"

def test_label_omits_page_when_none():
    c = Citation(doc_id="a", title="SOP", dept=Dept.COPA, page=None,
                 snippet="x", char_start=0, char_end=1)
    assert "pág." not in citation_label(c)

def test_markdown_contains_snippet():
    c = Citation(doc_id="a", title="SOP", dept=Dept.COPA, page=None,
                 snippet="texto citado", char_start=0, char_end=12)
    assert "texto citado" in citation_markdown(c)
```

```python
# tests/unit/test_ui_app.py
import ui.app  # importa sem executar a UI de fato (Streamlit fora do runtime)

def test_ui_module_exposes_main():
    assert callable(ui.app.main)
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_ui_logic.py tests/unit/test_ui_app.py -v`
Expected: FAIL — módulos inexistentes.

- [ ] **Step 3: Implementar**

`ui/app.py` com `main()` e entrada `if __name__ == "__main__": main()`; pipeline construído via `st.cache_resource`. Citação clicável: `st.link_button` para o ficheiro do corpus servido estaticamente (`/app/static`) quando disponível; caso contrário, expander com o snippet (documentar limitação honestamente no README).

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest tests/unit/test_ui_logic.py tests/unit/test_ui_app.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui tests/unit/test_ui_logic.py tests/unit/test_ui_app.py
git commit -m "feat: add Streamlit UI with citation display"
```

---

### Task 18: Deploy, CI com gate de eval, README e ADRs

**Files:**
- Create: `scripts/check_eval_threshold.py`, `.github/workflows/ci.yml` (substituir stub), `README.md` (substituir), `docs/decisions/0001-stack.md`, `docs/decisions/0002-hybrid-retrieval.md`, `docs/decisions/0003-docker-runtime.md`, `tests/unit/test_check_eval_threshold.py`
- Dockerfile/compose já existem da Task 1; aqui só se estendem para deploy.

**Interfaces:**
- Consumes: tudo.
- Produces:
  - `check_threshold(results: dict, thresholds: dict[str, float]) -> list[str]` (lista de falhas; vazia = ok).
  - CI com dois jobs: `test` (`docker build --target dev` + `docker compose run --rm dev pytest -m "not integration"` + `ruff`) e `eval-gate` condicional (workflow_dispatch/PR com label `eval`) que roda `scripts/check_eval_threshold.py results/latest.json`.
  - Deploy: Hugging Face Spaces com Docker SDK, usando o estágio `runtime` do Dockerfile; `README.md` com o frontmatter do Space.

- [ ] **Step 1: Escrever teste do gate**

```python
# tests/unit/test_check_eval_threshold.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from check_eval_threshold import check_threshold

def test_passes_when_above():
    assert check_threshold({"hybrid_rerank": {"recall@k": 0.8}}, {"recall@k": 0.7}) == []

def test_fails_when_below():
    fails = check_threshold({"hybrid_rerank": {"recall@k": 0.5}}, {"recall@k": 0.7})
    assert len(fails) == 1 and "recall@k" in fails[0]
```

- [ ] **Step 2: Rodar para ver falhar**

Run: `uv run pytest tests/unit/test_check_eval_threshold.py -v`
Expected: FAIL — `scripts/check_eval_threshold.py` inexistente.

- [ ] **Step 3: Implementar script + configs**

`check_eval_threshold.py` lê JSON (arg 1), compara com `thresholds.json` (arg 2, default `config/eval_thresholds.json` com `{"recall@k": 0.7, "mrr": 0.6, "ndcg@k": 0.65}`), imprime falhas e `sys.exit(1)` se houver. Docker Compose/Dockerfile e workflow conforme Interfaces.

- [ ] **Step 4: Rodar para ver passar**

Run: `uv run pytest -m "not integration" && uv run ruff check .`
Expected: PASS, suite completa verde.

- [ ] **Step 5: README + ADRs**

README: o problema, o diagrama, quickstart (`uv sync`, `docker compose up -d qdrant`, `python scripts/build_index.py`, `streamlit run ui/app.py`), **tabela de eval** (preencher após correr), decisões, limitações honestas, licença. ADR 0001: stack e porquê. ADR 0002: hybrid + rerank e porquê (com números quando existirem).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: add docker, CI eval gate, README and ADRs"
```

---

## Self-Review (feito pelo autor do plano)

1. **Cobertura do spec:** todos os goals têm tarefa — pipeline (6–14), eval comparado (10–11, 15), citação (12, 17), abstenção (12), engenharia (1, 16, 18), demo deploy (18, documentado no README), documentação didática (18 + ADRs). Non-goals não geram tarefas por definição.
2. **Passos:** cada passo tem uma ação e um resultado verificável; sem "TBD" nem "adicionar validações".
3. **Consistência de tipos:** nomes e assinaturas definidos nos blocos *Interfaces* e repetidos igual nas tarefas seguintes (`RetrievedChunk.source`, `retrieve(..., rewrite=...)`, `Answer.abstained`).
4. **Review Focus:** os 5 casos têm teste na tarefa dona (2, 6, 9, 11, 12).
5. **Proporção:** o plano fixa decisões (assinaturas, ficheiros, valores) e deixa os corpos para o implementador; os testes são o contrato, não transcrições da implementação.
