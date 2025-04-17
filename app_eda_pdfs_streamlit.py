# app_eda_pdfs_ngram_metrica.py
# ---------------------------------------------------------------
# EDA de PDFs com WordPiece, reconstrução, n‑grams, métricas
# (sentenças, médias, Top‑10/Down‑10, PoS), LDA, dendrograma
# e resumo Groq – com explicações automáticas nos gráficos.
# ---------------------------------------------------------------

# === IMPORTS ====================================================
import re, string, json, requests, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt; import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import streamlit as st
import PyPDF2, nltk, spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect
from transformers import BertTokenizerFast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats

# --- baixar recursos NLTK --------------------------------------
nltk.download("punkt")
nltk.download("stopwords")

# tagger padrão + novo pacote segmentado por idioma
nltk.download("averaged_perceptron_tagger")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

# sentence abbrev tables (punkt_tab)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# === CONFIG =====================================================
MODEL_NAME = "bert-base-uncased"
N_TOPICS   = 5
TOP_N      = 20

# --- carrega spaCy PT se disponível -----------------------------
try:
    nlp_pt = spacy.load("pt_core_news_sm")
except Exception:
    nlp_pt = None

# ================================================================
# ------------------ FUNÇÕES UTILITÁRIAS -------------------------
def extrair_texto_pdf(file):
    txt = ""
    for p in PyPDF2.PdfReader(file).pages:
        txt += p.extract_text() + " "
    return txt

def detectar_idioma(txt):
    try:
        return detect(txt)
    except:
        return "en"

def limpar_texto(t):
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return re.sub(f"[{re.escape(string.punctuation)}]", "", t).strip()

# ------------------ Métricas de sentenças/PoS -------------------
def sentencas(txt, lang):
    if lang.startswith("pt") and nlp_pt:
        return [s.text.strip() for s in nlp_pt(txt).sents]
    return sent_tokenize(txt)

def tokens_palavras(sent, lang):
    if lang.startswith("pt") and nlp_pt:
        return [t.text for t in nlp_pt(sent)]
    return word_tokenize(sent)

def pos_counts(tokens, lang):
    nouns=verbs=preps=0
    if lang.startswith("pt") and nlp_pt:
        for t in nlp_pt(" ".join(tokens)):
            if t.pos_=="NOUN": nouns+=1
            if t.pos_=="VERB": verbs+=1
            if t.pos_=="ADP":  preps+=1
    else:
        for _,tg in nltk.pos_tag(tokens):
            if tg.startswith("NN"): nouns+=1
            if tg.startswith("VB"): verbs+=1
            if tg=="IN":           preps+=1
    return nouns,verbs,preps

def gerar_metricas(txt_cln, tokens, lang):
    sents = sentencas(txt_cln, lang)
    num_sent = len(sents)
    mean_sent = np.mean([len(tokens_palavras(s,lang)) for s in sents]) if num_sent else 0
    num_tok = len(tokens); mean_tok = num_tok/num_sent if num_sent else 0
    freq = Counter(tokens)
    top10  = ", ".join([w for w,_ in freq.most_common(10)])
    low10  = ", ".join([w for w,_ in freq.most_common()[-10:]]) if len(freq)>=10 else ", ".join(freq)
    n,v,p = pos_counts(tokens, lang)
    return dict(Nº_Sentenças=num_sent, Média_Sent_Len=round(mean_sent,2),
                Nº_Tokens=num_tok, Média_Tokens_por_Sent=round(mean_tok,2),
                Top10=top10, Down10=low10,
                Substantivos=n, Verbos=v, Preposições=p)

# ------------------ WordPiece reconstrutor ----------------------
def tokens_reconstruidos(txt, tokenizer):
    enc = tokenizer(txt, return_offsets_mapping=True, add_special_tokens=True)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offs = enc["offset_mapping"]
    cur, out = "", []
    for t,(s,e) in zip(toks,offs):
        if t in ("[CLS]","[SEP]"): continue
        if t.startswith("##"):
            cur += t[2:]
        else:
            if cur: out.append(cur)
            cur = t
    if cur: out.append(cur)
    return out

# ------------------ n‑grams & tópicos ---------------------------
def ngram_freq(toks,n):
    return Counter([" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)])

def topicos_globais(textos, n_topics):
    vec = CountVectorizer(stop_words="english"); X = vec.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(X)
    words = vec.get_feature_names_out()
    return [[words[i] for i in comp.argsort()[-10:][::-1]] for comp in lda.components_]

# ------------------ Visual helpers ------------------------------
def barra(freq,title,cap):
    fig,ax=plt.subplots(figsize=(10,6))
    pares=sorted(freq.items(), key=lambda x:x[1], reverse=True)[:TOP_N]
    sns.barplot(x=[p[1] for p in pares], y=[p[0] for p in pares], ax=ax)
    ax.set_title(title); ax.set_xlabel("Frequência"); st.pyplot(fig); st.caption(cap)

def nuvem(freq,title,cap):
    wc=WordCloud(width=1600,height=800,background_color="white").generate_from_frequencies(freq)
    fig,ax=plt.subplots(figsize=(10,6)); ax.imshow(wc); ax.axis("off"); ax.set_title(title)
    st.pyplot(fig); st.caption(cap)

def histo(vals,title,cap):
    fig,ax=plt.subplots(figsize=(10,6))
    sns.histplot(vals,bins=20,kde=True,ax=ax)
    ax.set_title(title); ax.set_xlabel("Tamanho"); st.pyplot(fig); st.caption(cap)

def dendro(texts,top_n=50):
    vec=TfidfVectorizer(stop_words="english",max_features=top_n)
    X=vec.fit_transform(texts).toarray().T; words=vec.get_feature_names_out()
    fig,ax=plt.subplots(figsize=(16,8))
    dendrogram(linkage(X,method="ward"),labels=words,leaf_rotation=90,leaf_font_size=11,ax=ax)
    ax.set_title("Dendrograma de Palavras‑Chave"); st.pyplot(fig)
    st.caption("Linhas horizontais = união de clusters; altura = dissimilaridade.")

# ------------------ Resumo Groq ---------------------------------
def resumo_groq(txt):
    try: key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.warning("⚠️ Defina GROQ_API_KEY em secrets."); return ""
    r=requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        data=json.dumps({"model":"meta-llama/llama-4-scout-17b-16e-instruct",
                         "messages":[{"role":"user","content":f"Resuma:\n\n{txt}"}],
                         "temperature":0.3,"max_tokens":1024}))
    return r.json()["choices"][0]["message"]["content"] if r.ok else "Erro Groq"

# === STREAMLIT APP ==============================================
st.set_page_config(page_title="EDA PDFs + métricas", layout="wide")
st.title("📄 EDA de PDFs – métricas, n‑grams, PoS e Groq")

upl = st.sidebar.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)
if not upl: st.stop()

names=[u.name for u in upl]
pdf_sel=st.sidebar.selectbox("Escolha o PDF:", names)
idx=names.index(pdf_sel); texto_raw=extrair_texto_pdf(upl[idx])

lang=detectar_idioma(texto_raw)
stop=set(stopwords.words("portuguese" if lang.startswith("pt") else "english"))
tok=BertTokenizerFast.from_pretrained(MODEL_NAME)
texto=limpar_texto(texto_raw)
tokens_wp=tok.tokenize(texto)
tokens_rec=tokens_reconstruidos(texto,tok)

# Sidebar – métricas rápidas
metrics=gerar_metricas(texto,tokens_rec,lang)
st.sidebar.markdown("### 📏 Métricas do PDF")
for k,v in metrics.items():
    st.sidebar.write(f"**{k.replace('_',' ')}:** {v}")

# Cabeçalho
st.header(f"📚 PDF: `{pdf_sel}` — Idioma: **{lang.upper()}**")
c1,c2=st.columns(2)
c1.metric("Tokens WordPiece", len(tokens_wp))
c2.metric("Tokens Reconstr.", len(tokens_rec))

# Resumo Groq
st.subheader("📜 Resumo (Groq)")
with st.spinner("Gerando resumo…"):
    st.write(resumo_groq(texto))

st.divider()

# Frequências / n‑grams / nuvens
def blocos(tokens,label):
    freq_all=Counter(tokens)
    freq_fil=Counter([t for t in tokens if t not in stop])
    barra(freq_all,f"Top {label} (com stopwords)","Barra = contagem.")
    barra(freq_fil,f"Top {label} (sem stopwords)","Stopwords removidas.")
    nuvem(freq_all,f"Nuvem {label} (com stopwords)","Palavras maiores = mais frequentes.")
    nuvem(freq_fil,f"Nuvem {label} (sem stopwords)","Realça termos-chave.")
    for n in (2,3):
        ng=ngram_freq(tokens,n)
        if ng:
            barra(ng,f"Top {n}-grams ({label})",f"Barras = {n}-grams mais frequentes.")
            nuvem(ng,f"Nuvem {n}-grams ({label})","")

st.subheader("📈 WordPiece");       blocos(tokens_wp,"WordPiece")
st.subheader("📈 Reconstruídos");   blocos(tokens_rec,"Reconstruídos")

# Distribuições
st.subheader("🔵 Distribuições de Tamanho")
histo([len(t) for t in tokens_wp],"WordPiece","Tamanho em caracteres.")
histo([len(t) for t in tokens_wp if t not in stop],"WordPiece (sem stopwords)","")
histo([len(t) for t in tokens_rec],"Reconstruídos","Tokens completos tendem a ser maiores.")
histo([len(t) for t in tokens_rec if t not in stop],"Reconstruídos (sem stopwords)","")

# Estatísticas detalhadas
st.subheader("📊 Estatísticas detalhadas")
st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

st.divider()

# Tópicos + Dendrograma
st.subheader("🧠 Tópicos (LDA global)")
all_clean=[limpar_texto(extrair_texto_pdf(f)) for f in upl]
for i,top in enumerate(topicos_globais(all_clean,N_TOPICS),1):
    st.write(f"**Tópico {i}:** {', '.join(top)}")

st.subheader("🌳 Dendrograma global")
dendro(all_clean)
