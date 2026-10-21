# ADR 0003 — Runtime Docker-first

- **Estado:** aceite
- **Data:** 2026-10-20

## Contexto

O projeto tem de correr de forma idêntica na máquina de quem avalia, no CI e na
demo pública. As dependências de ML são pesadas e os pesos dos modelos não podem
ser re-descarregados em cada build.

## Decisão

Um único `Dockerfile` multi-stage:

1. **`base`** — `python:3.12-slim`, `uv`, dependências de produção (incluindo o
   extra `ml`) instaladas em `/opt/venv`.
2. **`dev`** — acrescenta `--all-extras` (pytest, ruff) e corre os testes,
   incluindo os de integração (usam fakes, sem descarregar modelos).
3. **`runtime`** — só produção, utilizador não-root `app`, healthcheck em
   `/health`.

Detalhes que evitam armadilhas comuns:

- **`/opt/venv` em vez de `/app/.venv`:** o serviço `dev` monta `.:/app`; se o
  venv estivesse dentro de `/app`, o bind mount apagava-o.
- **`HF_HOME=/models` + volume nomeado:** os pesos sobrevivem a `docker compose
  down` e não entram na imagem.
- **`--no-install-project` na stage base:** as dependências são cacheadas antes
  de copiar o código, invalidando a cache só quando `pyproject.toml`/`uv.lock`
  mudam.
- **Torch CPU por omissão:** `pyproject.toml` fixa `torch` ao índice
  `pytorch-cpu`, o que remove ~5 GB de wheels CUDA da imagem de deploy (HF
  Spaces é CPU-only). Localmente, o mesmo lock mantém CPU e produção iguais.
- **GPU é opt-in de verdade:** `docker compose --profile gpu up api-gpu` constrói
  o target `runtime-gpu`, que reinstala o wheel CUDA do PyPI (e exige
  `nvidia-container-toolkit` no host); a imagem por omissão não é afetada.
- **HF Spaces:** um único container corre a API em `localhost:8000` e o
  Streamlit em `0.0.0.0:8501` (`scripts/start_space.sh`), com Qdrant in-memory.

## Alternativas consideradas

- **Imagem só de runtime + `uv` no CI:** mais rápida, mas CI e produção deixavam
  de partilhar o mesmo ambiente.
- **`docker-compose` com `build:` por serviço:** duplicava camadas; usar a mesma
  imagem com `target` e `command` diferentes é mais simples.

## Consequências

- O primeiro build demora vários minutos (torch + export de camadas); builds
  seguintes usam cache. Com torch CPU a imagem de runtime fica em ~2,6 GB
  (seriam ~10 GB com as wheels CUDA).
- O CI constrói a imagem `dev` com cache `gha`, o que mantém o custo aceitável.
- A demo em HF Spaces arranca mais devagar (embedding do corpus no startup),
  aceitável para um Space gratuito.
