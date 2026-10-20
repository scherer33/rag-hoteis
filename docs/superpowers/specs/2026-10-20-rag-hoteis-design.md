# RAG Hoteleiro do Algarve — Design Document

> **Data:** 2026-10-20
> **Autor:** conversa de planeamento com o dono do projeto
> **Estado:** rascunho para aprovação — nada é código até aprovar
> **Objetivo de leitura:** qualquer AI engineer senior consegue planear a execução a partir daqui, sem precisar de voltar à conversa.

---

## 1. Intent e o porquê

Projeto de **portfólio** de AI/RAG engineer, em **Python**, publicado publicamente no GitHub (gratuito). O domínio escolhido é um hotel fictício no **Algarve**, Portugal, que gera documentos de procedimentos internos (SOPs, manuais, fichas técnicas, tabelas, registos). O cenário hoteleiro é plausível e reconhecível — mas o que **diferencia** este portfólio não é o domínio; é a **qualidade do retrieval demostrávelmente avaliada**, a **citação rastreável** e a **rigor de engenharia de software** à volta (testes, CI, container, README profissional, decisões documentadas).

O mercado de 2026 confirma a direção: "RAG deixou de ser side quest" — as vagas de AI/RAG engineer avaliam retrieval-quality (chunking, hybrid search, reranking), avaliação (RAGAS/TruLens), robustez (abstenção, grounding, citação) e disciplina de SWE (~90% do trabalho de um AI engineer é engenharia de software, não LLM). Este projeto é desenhado para mostrar todas essas camadas.

## 2. Goals (o que queremos mostrar)

1. Pipeline RAG que responde a perguntas sobre SOPs de um hotel do Algarve em **PT-PT e EN**, citando o **documento e o trecho exatos** de onde a resposta veio.
2. **Avaliação comparativa real**: pelo menos três configurações de retrieval testadas contra um golden dataset (métricas RAGAS — faithfulness, context precision, context recall, answer relevance), tabela publicada no README, e a **decisão de configuração tomada com base em números**, não em opinião.
3. **Citação rastreável**: cada resposta aponta para a fonte (documento + trecho) e a UI abre o PDF na página certa.
4. **Abstenção honesta**: quando a base não cobre a pergunta, o sistema diz "não encontrei" (recusa), em vez de inventar.
5. **Engenharia de software rigorosa**: `uv`, `ruff`, `pytest`, Docker Compose, GitHub Actions CI, ADRs documentando decisões.
6. **Demo deployada** em free tier (Hugging Face Spaces), clicável em entrevistas.
7. **Documentação didática**: cada fase explicada em `docs/`, mostrando como um AI engineer pensa (o porquê de cada decisão).

## 3. Non-goals (o que fica fora de propósito)

- ❌ **Filiais / multi-property.** O campo `property_id` fica no schema dos documentos **apenas como gancho de conversa de entrevista** ("a arquitetura suporta multi-tenant; adicionaria um filtro por propriedade e repetiria o eval por filial"). Sem multi-tenant real.
- ❌ **Agentes com tool-use** ou que executam ações em sistemas externos (fazer reservas, etc). Foco no RAG.
- ❌ **Fine-tuning** de LLM — caro, lento, e não é o que as vagas pedem para este nível.
- ❌ **App mobile** — UI web responsiva basta.

## 4. Anti-goals (o que conta como falha, mesmo que funcione)

- O RAG funciona, mas a avaliação é "testei com 5 perguntas e parece bom". → **falha o portfólio**.
- UI funciona, mas sem citação rastreável, ou com citação quebrada (não abre na fonte certa). → **falha o portfólio**.
- O sistema **inventa resposta** quando a base não cobre. → **falha o portfólio**.
- README sem demo clicável e sem tabela de eval. → **portfólio fraco**.

## 5. Constraints

- **Pasta do projeto:** tudo em `/home/phfs/Work/rag_hoteis` — código, dados, docs, testes. Nada é escrito fora daqui.
- **Runtime: Docker-first.** Todo o stack (Qdrant, API, UI, ingestão) corre via Docker Compose. O Dockerfile é a fonte única de verdade do ambiente; o CI usa a mesma imagem. Testes correm no serviço `dev` (`docker compose run --rm dev pytest`) — `uv` local fica como loop rápido opcional.
- **Metodologia:** **TDD obrigatório** (RED → GREEN → REFACTOR): nenhum código de produção sem escrever primeiro um teste que falha. Exceções só com aprovação do dono: ficheiros de configuração, corpus (dados), protótipos descartáveis.
- **Linguagem:** Python (3.12).
- **Framework:** LangChain (componentes: document loaders, text splitters, embeddings, retrievers, prompt templates).
- **Vector DB:** Qdrant (mencionado nas vagas de 2026), em Docker Compose.
- **Embeddings:** **local** e **multilíngue** (PT-PT e EN). Escolha primária: `BAAI/bge-m3` via sentence-transformers. Fallback em CPU modesta: `nomic-ai/nomic-embed-text-v2-moe`.
- **Reranker:** local, cross-encoder `BAAI/bge-reranker-v2-m3` (gratuito, suficiente para demo).
- **LLM (geração de respostas):** via **9router/omniroute** do dono (cliente próprio, interface OpenAI-compatível). Configuração em `config/llm.yaml` (nome do modelo, base_url, timeout). **A chave NUNCA é committada** — lida de env var ou ficheiro local não versionado.
- **Corpus:** documentos fictícios plausíveis, autorais ou adaptados de manuais públicos reescritos. 4 departamentos — **receção, F&B (restaurante/bar), copa (cozinha), estacionamento** — num único hotel do Algarve.
- **Idioma dos dados de avaliação:** **PT-PT predominante, com 4-5 perguntas em EN** para demonstrar robustez multilingue.
- **Prazo:** 2–3 semanas.
- **Publicação:** GitHub público e gratuito.

## 6. Arquitetura (visão de conjunto)

```
                     ┌─────────────────────────────────────────────┐
                     │              INGESTÃO (fase 2)              │
                     │  corpus JSONL → loaders → chunker →         │
                     │  embeddings (local) → Qdrant (Docker)       │
                     └─────────────────────────────────────────────┘
                                          │
        query em PT/EN                    ▼
   ┌──────────────────┐     ┌─────────────────────────────────────────────┐
   │   UI (Streamlit) │     │               RETRIEVAL (fase 3)             │
   │   citação        │────▶│  query rewrite → hybrid search              │
   │   clicável       │     │  (BM25 + vetorial) → reranker → top-k        │
   └──────────────────┘     └─────────────────────────────────────────────┘
                                          │  contexto (chunks + fontes)
                                          ▼
                        ┌─────────────────────────────────────────────┐
                        │  GERAÇÃO (fase 5)                            │
                        │  LLM via 9router (OpenAI-compat)            │
                        │  prompt grounded + abstenção + citações      │
                        └─────────────────────────────────────────────┘
                                          │  resposta + sources
                                          ▼
                        ┌─────────────────────────────────────────────┐
                        │  AVALIAÇÃO (fase 4)                          │
                        │  golden dataset → RAGAS → tabela comparativa │
                        │  CI gate (falha PR se métricas caem)         │
                        └─────────────────────────────────────────────┘
```

**Fluxo do usuário (o que a pessoa vê):** digita uma pergunta ("qual o procedimento quando o hóspede pede check-out às 14h mas a saída é às 11h?") → o sistema devolve resposta em PT-PT/EN com as **fontes** listadas (departamento, documento, trecho) → clicar numa citação abre o PDF na página/trecho certo → se não houver resposta na base, o sistema recusa e sugere contactar o departamento.

## 7. Fases de execução (cada uma termina com algo que corre + commit pequeno)

| # | Fase | Saída visível | Commit-type |
|---|------|---------------|-------------|
| 0 | **Setup** | `mise trust`, instalar `uv`, `python 3.12`, `git init`, repo no GitHub, CI stub vazio | scaffold |
| 1 | **Corpus** | 25–35 documentos fictícios em `data/corpus/` (markdown/pdf), schema JSONL em `data/index.jsonl` com metadata: `dept`, `doc_type`, `date`, `version`, `property_id` | data |
| 2 | **Ingestão** | chunker + pipeline, embeddings locais, Qdrant a correr em Docker, index populado | feat |
| 3 | **Retrieval** | hybrid search (BM25 + vetorial), reranker, query rewrite — componente testável e comparável | feat |
| 4 | **Eval** | golden dataset (~30 perguntas com resposta esperada + docs esperados), RAGAS, **tabela comparativa** das 3 configurações | eval |
| 5 | **API** | FastAPI: `/answer` (com abstenção), `/sources` — testes | feat |
| 6 | **UI** | Streamlit com citação clicável → abre PDF na página certa | feat |
| 7 | **Polish** | tracing (opcional, se tempo), README com GIF+diagrama+tabela de eval, Docker Compose, deploy HF Spaces, ADRs em `docs/decisions/` | docs+deploy |

## 8. Estratégia de avaliação (o coração do portfólio)

- **Golden dataset**: ~30 perguntas PT-PT (com umas 4–5 EN), cada uma com: pergunta, resposta esperada (golden answer), e **conjunto de documentos/chunks que a resposta deveria usar** (golden context). Separado em `data/golden/` — não se mistura com o corpus de ingestão.
- **Configurações comparadas** (mínimo três):
  - A) baseline: chunking fixo + embedding-only retrieval
  - B) A + **hybrid search** (vetorial + BM25)
  - C) B + **reranker** cross-encoder
- **Métricas RAGAS**: faithfulness, answer relevancy, context precision, context recall (e recall@k/precision@k de retrieval puro).
- **Saída**: tabela comparativa no README, e a **configuração escolhida** justificada com os números.
- **CI gate**: no GitHub Actions, o eval corre com threshold mínimo; **PR que baixa as métricas falha**. (Documentado como ADR.)

## 9. Decisões já tomadas na conversa (tuyas)

| Tema | Decisão |
|------|---------|
| Escopo | Projeto completo, passo a passo, bem documentado, nível "AI engineer", GitHub público gratuito |
| Dados | Inventar/baixar (hotel fictício do Algarve) |
| Embeddings | Locais + multilíngue |
| LLM de geração | Via 9router/omniroute (OpenAI-compat), modelo a escolher depois na fase 5 |
| UI | Com citação clicável que abre o PDF na página |
| Domínio | 4 departamentos: receção, F&B, copa, estacionamento |
| Filiais | Descartadas (`property_id` só no schema) |
| Idioma do eval | PT-PT com casos EN |
| Runtime | Docker-first: Compose com `qdrant` + `api` + `ui` + `dev`; mesma imagem no CI e no deploy |
| Prazo | Curto, 2–3 semanas |

## 10. Minhas chamadas (decisões que o implementador pode tomar sozinho — marcadas como "minha escolha")

- **Embedding model:** `BAAI/bge-m3`; fallback `nomic-embed-text-v2-moe` se CPU modesta. *(minha escolha)*
- **Reranker:** `BAAI/bge-reranker-v2-m3`. *(minha escolha)*
- **BM25:** implementação nativa do LangChain (`langchain-community`). *(minha escolha)*
- **Vector DB:** Qdrant em Docker Compose — escolhido porque o mercado o cita. *(minha escolha)*
- **Eval framework:** RAGAS. *(minha escolha)*
- **UI:** Streamlit — mais barato que React para 2 semanas, ainda impressiona. *(minha escolha)*
- **CI:** GitHub Actions, pytest + eval gate no PR. *(minha escolha)*
- **Deploy:** Hugging Face Spaces com **Docker SDK** (usa o mesmo Dockerfile; demo com Qdrant embutido/in-memory e LLM via secrets). *(minha escolha)*
- **Docker:** multi-stage (`base` → `dev` → `runtime`), imagem `python:3.12-slim`, utilizador não-root, healthchecks, cache de pesos HF em volume (`HF_HOME=/models`), GPU como profile opcional (`docker compose --profile gpu`, requer nvidia-container-toolkit). *(minha escolha)*
- **Testes:** canónicos no container `dev`; `uv` local opcional para TDD rápido. *(minha escolha)*
- **Chunking size / top-k / temperatura / prompt exato / política de abstenção (threshold):** a definir durante a execução, documentado como ADR, decisão baseada em números do eval. *(minha escolha)*

## 11. Em aberto (tuyas — respondidas antes da execução)

1. **Modelo LLM concreto via 9router**: na fase 5, listo os modelos disponíveis no 9router/omniroute e o dono escolhe (`llama3.3-70b`, `qwen2.5-72b`, `mistral-large`, etc). Gravado em `config/llm.yaml`.
2. **Confirmar que BGE-M3 corre bem na máquina** (fase 0) ou trocar para `nomic-embed` — perguntar rápido na fase 0.
3. **(Documentado como resolvido)** Idioma do eval: PT-PT + 4–5 EN.

## 12. Fora de âmbito (armadilhas evitadas de propósito)

- ❌ Agente que faz reservas/executa ações.
- ❌ Fine-tuning.
- ❌ App mobile.
- ❌ Multi-filial real.

---

## O que falta ao implementador (resumo)

Construir as 8 fases da secção 7, por ordem, cada uma com commits pequenos, testes e documentação. As decisões "minha escolha" da secção 10 podem ser tomadas sem voltar ao dono. Os 2 itens "em aberto" da secção 11 exigem uma pergunta rápida ao dono (fase 0: embeddings; fase 5: modelo LLM).