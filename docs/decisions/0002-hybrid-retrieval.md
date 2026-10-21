# ADR 0002 — Retrieval híbrido com fusão RRF e reranking

- **Estado:** aceite
- **Data:** 2026-10-20

## Contexto

O corpus tem procedimentos em PT-PT e inglês, com códigos (`E3`), valores
(`82 °C`, `50 euros`) e nomes próprios. Retrieval puramente denso tende a
suavizar exatamente esse tipo de token; retrieval puramente lexical falha em
paráfrases e em perguntas colocadas noutra língua.

## Decisão

Três configurações comparáveis, selecionáveis em runtime:

1. **`vector`** — similaridade de cosseno sobre embeddings `bge-m3`.
2. **`hybrid`** — união de `vector` e BM25 com **Reciprocal Rank Fusion**:
   `score(d) = Σ 1 / (k + rank_i(d))`, com `k = 60`.
3. **`hybrid_rerank`** — os 20 candidatos fundidos passam por um cross-encoder
   (`bge-reranker-v2-m3`); os scores são normalizados com sigmoid para que o
   limiar de abstenção tenha o mesmo significado nas três configurações.

O BM25 usa `BM25Plus`, não `BM25Okapi`: em corpora pequenos o IDF do Okapi pode
colapsar para zero quando um termo aparece em metade dos documentos.

A abstenção é decidida antes e depois da geração:

- **antes:** sem chunks ou score do topo abaixo de `ABSTAIN_MIN_SCORE`;
- **depois:** o LLM responde `ABSTAIN` quando o contexto não contém a resposta.

## Porquê RRF e não soma de scores

Os scores densos e lexicais não são comparáveis (cosseno vs. BM25). O RRF usa
apenas a posição, pelo que não exige normalização nem calibração entre
retrievers — uma escolha robusta e fácil de defender em avaliação.

## Resultados medidos

Gerados por `make eval` sobre 40 perguntas (35 PT, 5 EN), `k = 5`, corpus real
do projeto (`results/latest.json`):

| Configuração | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| `vector` | 1.000 | 0.988 | 0.991 |
| `hybrid` | 1.000 | 0.983 | 0.988 |
| `hybrid_rerank` | 1.000 | 1.000 | 1.000 |

O recall está saturado: o corpus é pequeno e as perguntas derivam dos documentos,
por isso o documento certo entra quase sempre no top-5. O que diferencia as
configurações é a ordem: o reranker acerta o primeiro lugar em todas as
perguntas, e o híbrido sem reranker fica ligeiramente abaixo do vetor puro
(0.983 vs. 0.988), porque o BM25 promove termos comuns em perguntas
parafraseadas.

**Conclusão defensável:** neste corpus, o reranker é o que traz ganho mensurável;
o híbrido só se justifica com um benchmark mais difícil (nomes, códigos,
paráfrases coloquiais) ou com um corpus maior.

## Consequências

- `hybrid_rerank` é o modo por omissão na API e na UI.
- O reranker acrescenta latência; `vector` e `hybrid` continuam disponíveis para
  comparação e para máquinas sem recursos.
- O limiar de abstenção deve ser revisto quando o reranker ou o corpus mudarem.
