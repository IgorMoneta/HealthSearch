# 🩺 HealthSearch

Motor de busca médica híbrido desenvolvido em Python e Streamlit.

O HealthSearch combina **BM25** para recuperação lexical, **embeddings semânticos** para busca por significado e **Reciprocal Rank Fusion (RRF)** para gerar um ranking híbrido.

## Funcionalidades

- Corpus médico pré-carregado com 6 documentos.
- Tokenização e limpeza de texto em português.
- Normalização para minúsculas.
- Remoção de caracteres especiais e stopwords.
- BM25 com `k₁` e `b` ajustáveis.
- Busca semântica com Sentence Transformers.
- Simulação vetorial como fallback documentado.
- Similaridade de cosseno.
- Fusão híbrida com RRF.
- Peso `α` ajustável entre BM25 e busca semântica.
- Interface em abas.
- Matriz comparativa dos rankings.

## Fluxo

```text
Corpus → Pré-processamento → BM25 ──────┐
                         ↘ Embeddings ───┤→ RRF → Ranking
```

## Fórmula RRF

```text
Score_RRF(D) = α × 1/(60 + Rank_BM25)
             + (1 - α) × 1/(60 + Rank_Semântico)
```

## Instalação

```bash
pip install streamlit pandas rank-bm25 sentence-transformers
```

## Execução

```bash
streamlit run healthsearch_app.py
```

## Estrutura

```text
HealthSearch/
├── healthsearch_app.py
└── README.md
```

## Tecnologias

- Python
- Streamlit
- Pandas
- rank-bm25
- Sentence Transformers

## Evoluções possíveis

- Cross-Encoder para re-ranking dos Top-3.
- Importação de documentos externos.
- Persistência dos embeddings.
- Corpus médico maior.
- Métricas de avaliação como Precision@K e NDCG.
