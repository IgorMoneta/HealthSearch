import math
import re
import unicodedata

import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS = True
except ImportError:
    SENTENCE_TRANSFORMERS = False

DOCUMENTOS = [
    {"id": 1, "titulo": "Protocolo Emergência ECG", "texto": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."},
    {"id": 2, "titulo": "Guia de Farmacologia Cardíaca", "texto": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."},
    {"id": 3, "titulo": "Diretriz de Hipertensão Arterial", "texto": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."},
    {"id": 4, "titulo": "Manual de AVC Isquêmico", "texto": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."},
    {"id": 5, "titulo": "Protocolo de Reanimação RCR", "texto": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."},
    {"id": 6, "titulo": "Procedimentos de UTI Geral", "texto": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."}
]

STOPWORDS = {"a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "e", "em", "no", "na", "nos", "nas", "para", "por", "com", "que", "ao", "aos", "se", "até", "como"}

SINONIMOS = {
    "dor": {"dor", "precordial", "toracica", "toracico"},
    "coracao": {"coracao", "cardiaca", "cardiaco", "coronariana", "miocardio"},
    "infarto": {"infarto", "miocardio"},
    "pressao": {"pressao", "hipertensao"},
    "avc": {"avc", "cerebral", "isquemico"},
    "parada": {"parada", "cardiorrespiratoria", "rcr"},
    "eletrocardiograma": {"eletrocardiograma", "ecg"},
    "monitoramento": {"monitoramento", "monitorizacao", "telemetria"}
}


def limpar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9\s-]", " ", texto)
    tokens = re.findall(r"\b[a-z0-9-]+\b", texto)
    return [token for token in tokens if token not in STOPWORDS]


def calcular_bm25(bm25, query, k1, b):
    bm25.k1 = k1
    bm25.b = b
    scores = bm25.get_scores(query)
    return sorted(
        [{"ID": d["id"], "Título": d["titulo"], "Score": float(s)} for d, s in zip(DOCUMENTOS, scores)],
        key=lambda x: x["Score"], reverse=True
    )


def vetor_simulado(tokens):
    return [sum(1 for token in tokens if token in grupo) for grupo in SINONIMOS.values()]


def similaridade_cosseno(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


@st.cache_resource
def carregar_modelo():
    if not SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


def buscar_semantico(query, modelo):
    if modelo is not None:
        textos = [d["texto"] for d in DOCUMENTOS]
        vetores = modelo.encode(textos, normalize_embeddings=True)
        vetor_query = modelo.encode([query], normalize_embeddings=True)[0]
        scores = [float(sum(a * b for a, b in zip(v, vetor_query))) for v in vetores]
    else:
        query_vetor = vetor_simulado(limpar_texto(query))
        scores = [similaridade_cosseno(query_vetor, vetor_simulado(limpar_texto(d["texto"]))) for d in DOCUMENTOS]

    return sorted(
        [{"ID": d["id"], "Título": d["titulo"], "Score": float(s)} for d, s in zip(DOCUMENTOS, scores)],
        key=lambda x: x["Score"], reverse=True
    )


def ranking(resultados):
    return {item["ID"]: pos for pos, item in enumerate(resultados, 1)}


def calcular_rrf(ranking_bm25, ranking_semantico, alpha):
    k_rrf = 60
    resultados = []
    for doc_id in ranking_bm25:
        rb = ranking_bm25[doc_id]
        rs = ranking_semantico[doc_id]
        score = alpha / (k_rrf + rb) + (1 - alpha) / (k_rrf + rs)
        titulo = next(d["titulo"] for d in DOCUMENTOS if d["id"] == doc_id)
        resultados.append({"ID": doc_id, "Título": titulo, "Rank BM25": rb, "Rank Semântico": rs, "Score RRF": score})
    return sorted(resultados, key=lambda x: x["Score RRF"], reverse=True)


st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")
st.title("🩺 HealthSearch")
st.write("Motor de busca médica híbrido com BM25, busca semântica e Reciprocal Rank Fusion.")

st.sidebar.header("Parâmetros")
k1 = st.sidebar.slider("k₁ - Saturação de frequência", 0.0, 3.0, 1.2, 0.1)
b = st.sidebar.slider("b - Normalização por comprimento", 0.0, 1.0, 0.75, 0.05)
alpha = st.sidebar.slider("α - Peso BM25 no RRF", 0.0, 1.0, 0.5, 0.05)

query = st.text_input("Consulta médica", placeholder="Ex.: dor no peito e infarto")

tokens_documentos = [limpar_texto(d["texto"]) for d in DOCUMENTOS]
bm25 = BM25Okapi(tokens_documentos)
modelo = carregar_modelo()

abas = st.tabs(["Léxico", "Semântico", "Híbrido RRF", "Matriz Comparativa"])

if query:
    tokens_query = limpar_texto(query)
    resultados_bm25 = calcular_bm25(bm25, tokens_query, k1, b)
    resultados_semanticos = buscar_semantico(query, modelo)
    ranking_bm25 = ranking(resultados_bm25)
    ranking_semantico = ranking(resultados_semanticos)
    resultados_rrf = calcular_rrf(ranking_bm25, ranking_semantico, alpha)
    ranking_rrf = ranking(resultados_rrf)

    with abas[0]:
        st.subheader("Motor Léxico - BM25")
        st.write("Tokens da consulta:", tokens_query)
        df = pd.DataFrame(resultados_bm25).rename(columns={"Score": "Score BM25"})
        st.dataframe(df, use_container_width=True, hide_index=True)

    with abas[1]:
        st.subheader("Motor Semântico")
        if modelo is not None:
            st.info("Embeddings gerados com Sentence Transformers.")
        else:
            st.warning("Modelo Sentence Transformers indisponível. Usando simulação vetorial documentada no código.")
        df = pd.DataFrame(resultados_semanticos).rename(columns={"Score": "Similaridade de Cosseno"})
        st.dataframe(df, use_container_width=True, hide_index=True)

    with abas[2]:
        st.subheader("Reciprocal Rank Fusion")
        df = pd.DataFrame(resultados_rrf)
        df["Score RRF"] = df["Score RRF"].round(6)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success(f"Documento mais relevante: {resultados_rrf[0]['Título']}")

    with abas[3]:
        st.subheader("Matriz Comparativa")
        comparacao = []
        for d in DOCUMENTOS:
            comparacao.append({
                "ID": d["id"],
                "Título": d["titulo"],
                "Rank BM25": ranking_bm25[d["id"]],
                "Rank Semântico": ranking_semantico[d["id"]],
                "Rank RRF": ranking_rrf[d["id"]]
            })
        st.dataframe(pd.DataFrame(comparacao), use_container_width=True, hide_index=True)
else:
    st.info("Digite uma consulta para iniciar a busca.")

st.divider()
st.subheader("Corpus Médico")
st.dataframe(pd.DataFrame(DOCUMENTOS), use_container_width=True, hide_index=True)
