# app_eda_pdfs_ngram_explicativo.py
# ------------------------------------------------------------------
# Análise exploratória completa de PDFs (tokenização WordPiece,
# reconstrução via offset, n‑grams, gráficos interpretativos,
# modelagem de tópicos e resumo via Groq).
# ------------------------------------------------------------------

# === IMPORTAÇÕES ==================================================
import re, string, json, requests, os
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import streamlit as st
import PyPDF2, nltk
from nltk.corpus import stopwords
from langdetect import detect
from transformers import BertTokenizerFast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats

# === CONFIG =======================================================
MODEL_NAME = "bert-base-uncased"
N_TOPICS   = 5
TOP_N      = 20   # top n‑grams a exibir
nltk.download('stopwords')

# === FUNÇÕES UTILITÁRIAS ==========================================
def extrair_texto_pdf(uploaded_file):
    txt = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        txt += page.extract_text() + " "
    return txt

def detectar_idioma(texto):
    try:
        return detect(texto)
    except:                      # fallback
        return 'en'

def limpar_texto(txt):
    txt = txt.lower()
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(f"[{re.escape(string.punctuation)}]", "", txt)
    return txt.strip()

# ---------- visual ------------
def barra_frequencia(freq, titulo, legenda):
    fig, ax = plt.subplots(figsize=(10,6))
    pares = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    sns.barplot(x=[p[1] for p in pares], y=[p[0] for p in pares], ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Frequência")
    st.pyplot(fig)
    st.caption(legenda)

def nuvem(frequencia, titulo, legenda):
    wc = WordCloud(width=1600, height=800, background_color="white").generate_from_frequencies(frequencia)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo)
    st.pyplot(fig)
    st.caption(legenda)

def histograma(tamanhos, titulo, legenda):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(tamanhos, bins=20, kde=True, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Tamanho do token")
    st.pyplot(fig)
    st.caption(legenda)

# ---------- estatísticas -------
def stats_tamanhos(nome, tamanhos):
    moda_val = stats.mode(tamanhos, keepdims=False).mode if len(tamanhos) else np.nan
    if isinstance(moda_val, (np.ndarray, list)): moda_val = moda_val[0] if len(moda_val)>0 else np.nan
    return dict(
        Nome              = nome,
        Total             = len(tamanhos),
        Média             = np.mean(tamanhos)  if len(tamanhos) else np.nan,
        Mediana           = np.median(tamanhos)if len(tamanhos) else np.nan,
        Moda              = moda_val,
        Mínimo            = np.min(tamanhos)   if len(tamanhos) else np.nan,
        Máximo            = np.max(tamanhos)   if len(tamanhos) else np.nan,
        Desvio_Padrão     = np.std(tamanhos)   if len(tamanhos) else np.nan
    )

# ---------- token recon --------
def tokens_reconstruidos(texto, tokenizer):
    enc = tokenizer(texto, return_offsets_mapping=True, add_special_tokens=True)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offs = enc["offset_mapping"]
    palavra, palavras = "", []
    for tok, (s, e) in zip(toks, offs):
        if tok in ("[CLS]", "[SEP]"): continue
        if tok.startswith("##"):
            palavra += tok[2:]
        else:
            if palavra: palavras.append(palavra)
            palavra = tok
    if palavra: palavras.append(palavra)
    return palavras

# ---------- n‑grams ------------
def contar_ngrams(tokens, n):
    return Counter([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

# ---------- tópicos ------------
def topicos_globais(textos, n_topics):
    vect = CountVectorizer(stop_words="english")
    X = vect.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(X)
    words = vect.get_feature_names_out()
    return [[words[i] for i in comp.argsort()[-10:][::-1]] for comp in lda.components_]

# ---------- dendrograma --------
def dendrograma(textos, top_n=50):
    vect = TfidfVectorizer(stop_words="english", max_features=top_n)
    X = vect.fit_transform(textos).toarray().T
    words = vect.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(16,8))
    dendrogram(linkage(X, method="ward"), labels=words, leaf_rotation=90, leaf_font_size=11, ax=ax)
    ax.set_title("Dendrograma de Palavras‑Chave")
    st.pyplot(fig)
    st.caption("Interpretação: linhas horizontais representam agrupamentos; \nquanto mais alto o ponto de união, menos semelhantes os grupos.")

# ---------- resumo Groq -------
def resumo_groq(texto):
    try:
        key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.warning("⚠️ Defina GROQ_API_KEY em secrets para gerar resumos.")
        return ""
    payload = {
        "model":"meta-llama/llama-4-scout-17b-16e-instruct",
        "messages":[{"role":"user","content":f"Resuma de forma objetiva:\n\n{texto}"}],
        "temperature":0.3,"max_tokens":1024}
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization":f"Bearer {key}",
                               "Content-Type":"application/json"},
                      data=json.dumps(payload))
    return r.json()["choices"][0]["message"]["content"] if r.ok else "Erro na API Groq"

# === APP STREAMLIT ===============================================
st.set_page_config(page_title="📄 EDA PDFs + n‑grams + Groq", layout="wide")
st.title("📄 EDA Completa de PDFs (WordPiece, n‑grams, Tópicos e Resumo Groq)")

upl = st.sidebar.file_uploader("📂 Faça upload de um ou mais PDFs", type="pdf", accept_multiple_files=True)
if not upl: st.stop()

names = [u.name for u in upl]
pdf_sel = st.sidebar.selectbox("Escolha o PDF para análise:", names)
texts  = [extrair_texto_pdf(f) for f in upl]

# === PROCESSA PDF SELECIONADO ====================================
idx   = names.index(pdf_sel)
texto = texts[idx]
idioma = detectar_idioma(texto)
lang  = "portuguese" if idioma.startswith("pt") else "english"
stop  = set(stopwords.words(lang))
texto_cln = limpar_texto(texto)

tok   = BertTokenizerFast.from_pretrained(MODEL_NAME)
tokens_wp = tok.tokenize(texto_cln)
tokens_rec= tokens_reconstruidos(texto_cln, tok)

# === HEADERS ======================================================
st.header(f"📚 PDF: `{pdf_sel}` — Idioma detectado: **{idioma.upper()}**")
c1,c2=st.columns(2)
c1.metric("Tokens WordPiece", len(tokens_wp))
c2.metric("Tokens Reconstruídos", len(tokens_rec))

# === RESUMO VIA GROQ =============================================
st.subheader("📜 Resumo Inteligente (Groq)")
with st.spinner("Gerando resumo..."):
    st.write(resumo_groq(texto_cln))

st.divider()

# === FREQUÊNCIAS & NUVENS ========================================
def mostra_freq(tokens, titulo_base):
    freq_all = Counter(tokens)
    freq_fil = Counter([t for t in tokens if t not in stop])
    barra_frequencia(freq_all,  f"Top {titulo_base} (com stopwords)",
                     "Barras: cada linha é uma palavra, comprimento = contagem.")
    barra_frequencia(freq_fil,  f"Top {titulo_base} (sem stopwords)",
                     "Mesma interpretação, removendo stopwords.")
    nuvem(freq_all,  f"Nuvem {titulo_base} (com stopwords)",
          "Palavras maiores = mais frequentes. Boa para visão geral.")
    nuvem(freq_fil,  f"Nuvem {titulo_base} (sem stopwords)",
          "Remove palavras muito comuns, realçando termos chave.")

st.subheader("📈 WordPiece")
mostra_freq(tokens_wp, "WordPiece")

st.subheader("📈 Reconstruídos")
mostra_freq(tokens_rec, "Reconstruídos")

# === N‑GRAMS ======================================================
def mostra_ngrams(tokens, label):
    for n in (2,3):
        freq = contar_ngrams(tokens, n)
        if not freq: continue
        barra_frequencia(freq, f"Top {n}-grams ({label})",
                         f"Cada barra é um {n}-gram (sequência de {n} palavras).")
        nuvem(freq, f"Nuvem {n}-grams ({label})",
              f"Tamanhos maiores indicam {n}-grams mais frequentes.")

st.subheader("📈 n‑grams WordPiece")
mostra_ngrams(tokens_wp, "WordPiece")

st.subheader("📈 n‑grams Reconstruídos")
mostra_ngrams(tokens_rec, "Reconstruídos")

# === DISTRIBUIÇÕES ===============================================
st.subheader("🔵 Distribuições de Tamanhos")
tam_wp = histograma([len(t) for t in tokens_wp],
                    "Distribuição WordPiece",
                    "Eixo X: tamanho do token; Altura: frequência.")
tam_wp_ns = histograma([len(t) for t in tokens_wp if t not in stop],
                    "Distribuição WordPiece (sem stopwords)",
                    "Comparar com gráfico anterior para ver impacto de stopwords.")
tam_rc = histograma([len(t) for t in tokens_rec],
                    "Distribuição Reconstruídos",
                    "Tokens já reconstruídos tendem a ser maiores.")
tam_rc_ns= histograma([len(t) for t in tokens_rec if t not in stop],
                    "Distribuição Reconstruídos (sem stopwords)",
                    "Remove termos muito comuns.")

# === TABELA ESTATÍSTICA ==========================================
st.subheader("📊 Estatísticas Resumidas")
df_stats = pd.DataFrame([
    stats_tamanhos("WordPiece (com stopwords)", [len(t) for t in tokens_wp]),
    stats_tamanhos("WordPiece (sem stopwords)", [len(t) for t in tokens_wp if t not in stop]),
    stats_tamanhos("Reconstruídos (com stopwords)", [len(t) for t in tokens_rec]),
    stats_tamanhos("Reconstruídos (sem stopwords)", [len(t) for t in tokens_rec if t not in stop]),
])
st.dataframe(df_stats, use_container_width=True)
st.caption("Como ler: cada linha resume distribuição de tamanhos — média, mediana, etc.")

st.divider()

# === TÓPICOS & DENDROGRAMA =======================================
st.subheader("🧠 Modelagem de Tópicos Global (LDA)")
for i,topico in enumerate(topicos_globais([limpar_texto(t) for t in texts], N_TOPICS), 1):
    st.write(f"**Tópico {i}:** {', '.join(topico)}")
st.caption("Cada tópico é um conjunto de palavras que ocorrem frequentemente juntas.")

st.subheader("🌳 Dendrograma de Palavras‑Chave")
dendrograma([limpar_texto(t) for t in texts])
