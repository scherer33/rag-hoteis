# ADR 0001 — Stack e limites do projeto

- **Estado:** aceite
- **Data:** 2026-10-20

## Contexto

O projeto é uma peça de portfólio para funções de engenharia de IA/RAG. Tem de
ser reproduzível por quem avalia, correr localmente sem depender de APIs pagas e
mostrar decisões de engenharia, não apenas um prompt.

## Decisão

| Camada | Escolha | Porquê |
|---|---|---|
| Linguagem | Python 3.12 | Ecossistema de ML e tipagem moderna (`StrEnum`, `X \| None`). |
| Gestão de dependências | `uv` | Resolve e instala em segundos; lockfile determinístico. |
| Vector store | Qdrant (`qdrant-client`) | Filtros por payload, modo in-memory para testes, imagem Docker oficial. |
| Embeddings | `BAAI/bge-m3` | Multilingue (PT/EN), 1024 dimensões, bom desempenho em retrieval denso. |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder multilingue, melhora precisão do top-k. |
| Retrieval léxico | `rank_bm25` (BM25Plus) | Complementa o denso em nomes, códigos e números (ex.: `E3`, `82 °C`). |
| LLM | Endpoint OpenAI-compatível (9router/omniroute) | Permite trocar de modelo sem alterar código; chave só em ambiente. |
| API | FastAPI | Contrato tipado, OpenAPI automático, `TestClient` nos testes. |
| UI | Streamlit | Demo rápida com citações expansíveis. |
| Testes | `pytest` + `ruff` | Loop rápido; testes canónicos correm no container `dev`. |

## Decisões de rejeição

- **LangChain para loaders/splitters:** os splitters standard não garantem
  offsets exatos no texto original, que é a base das citações rastreáveis.
  Um chunker próprio de ~40 linhas preserva `char_start`/`char_end` e prefere
  quebras de parágrafo/frase.
- **Base de dados relacional:** o corpus é imutável e pequeno (35 documentos);
  Qdrant cobre retrieval e filtros sem mais uma peça.
- **Fine-tuning:** o objetivo é retrieval aumentado com contexto verificável,
  não alterar o modelo base.

## Consequências

- O sistema depende de pesos locais (~4–5 GB para embeddings + reranker). O
  Docker isola isso num volume de cache (`HF_HOME=/models`).
- Sem GPU, a latência de geração é dominada pelo reranker; o modo `vector` fica
  disponível para comparação e para máquinas modestas.
- A demo pública usa Qdrant in-memory: o índice é reconstruído no arranque.
