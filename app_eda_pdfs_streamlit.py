# app_eda_pdfs_ngram_metrica_refcut.py

import re, string, json, requests
from glob import glob
from collections import Counter

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import PyPDF2, nltk, spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect
from transformers import BertTokenizer, BertTokenizerFast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.cluster.hierarchy import linkage, dendrogram
from wordcloud import WordCloud
import os

# ==== CONFIGURAÇÕES INICIAIS ====

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="EDA PDFs + métricas", layout="wide")

# NLTK downloads
for pkg in ("punkt", "stopwords", "averaged_perceptron_tagger"):
    nltk.download(pkg, quiet=True)

# spaCy
try: nlp_pt = spacy.load("pt_core_news_lg")
except: nlp_pt = None
try: nlp_en = spacy.load("en_core_web_lg")
except: nlp_en = None

MODEL_NAME = "bert-base-uncased"
tok_fast = BertTokenizerFast.from_pretrained(MODEL_NAME)
tok_slow = BertTokenizer.from_pretrained(MODEL_NAME)
CHUNK_SIZE = 512
CHUNK_STRIDE = 50
N_TOPICS = 5
TOP_N = 20

# ==== FUNÇÕES UTILITÁRIAS ====

def extrair_texto_pdf(file):
    textos = []
    for p in PyPDF2.PdfReader(file).pages:
        textos.append(p.extract_text() or "")
    full = " ".join(textos)
    parts = re.split(r'(?mi)^[ \t]*(references|referências)\b', full)
    return parts[0]

def detectar_idioma(txt):
    try: return detect(txt.strip()) or "en"
    except: return "en"

def limpar_texto(txt):
    t = re.sub(r"\s+", " ", txt.lower())
    return re.sub(f"[{re.escape(string.punctuation)}]", "", t).strip()

def split_sentences(txt, lang):
    if lang.startswith("pt"):
        try:
            return [s.text for s in nlp_pt(txt).sents]
        except:
            nlp_fallback = spacy.blank("pt")
            nlp_fallback.add_pipe("sentencizer")
            return [s.text for s in nlp_fallback(txt).sents]
    elif lang.startswith("en"):
        try:
            return [s.text for s in nlp_en(txt).sents]
        except:
            nlp_fallback = spacy.blank("en")
            nlp_fallback.add_pipe("sentencizer")
            return [s.text for s in nlp_fallback(txt).sents]
    else:
        nlp_fallback = spacy.blank("xx")
        nlp_fallback.add_pipe("sentencizer")
        return [s.text for s in nlp_fallback(txt).sents]


def word_tokens(sent, lang):
    if lang.startswith("pt") and nlp_pt: return [t.text for t in nlp_pt(sent)]
    if lang.startswith("en") and nlp_en: return [t.text for t in nlp_en(sent)]
    return word_tokenize(sent, language="portuguese" if lang.startswith("pt") else "english")

def count_pos(tokens, lang):
    n=v=p=0
    if lang.startswith("pt") and nlp_pt:
        for tok in nlp_pt(" ".join(tokens)):
            if tok.pos_=="NOUN": n+=1
            if tok.pos_=="VERB": v+=1
            if tok.pos_=="ADP":  p+=1
    elif lang.startswith("en") and nlp_en:
        for tok in nlp_en(" ".join(tokens)):
            if tok.pos_=="NOUN": n+=1
            if tok.pos_=="VERB": v+=1
            if tok.pos_=="ADP":  p+=1
    else:
        for _, tg in nltk.pos_tag(tokens):
            if tg.startswith("NN"): n+=1
            if tg.startswith("VB"): v+=1
            if tg=="IN":           p+=1
    return n,v,p

def chunk_text(txt):
    enc = tok_fast(txt, add_special_tokens=False, return_attention_mask=False)
    ids = enc["input_ids"]
    chunks = []; start=0
    while start < len(ids):
        block = ids[start:start+CHUNK_SIZE]
        block = [tok_fast.cls_token_id] + block + [tok_fast.sep_token_id]
        chunks.append({
            "input_ids": torch.tensor(block).unsqueeze(0),
            "attention_mask": torch.tensor([1]*len(block)).unsqueeze(0)
        })
        start += CHUNK_SIZE-CHUNK_STRIDE
    return chunks

def reconstruct(tokens, tokenizer):
    enc = tokenizer(tokens, return_offsets_mapping=True, add_special_tokens=True)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    out = []; cur = ""
    for t, off in zip(toks, enc["offset_mapping"]):
        if t in ("[CLS]","[SEP]"): continue
        if t.startswith("##"):
            cur += t[2:]
        else:
            if cur: out.append(cur)
            cur = t
    if cur: out.append(cur)
    return out

def metrics(txt_raw, toks, lang, label):
    sents = split_sentences(txt_raw, lang)
    num_s = len(sents)
    mean_s = np.mean([len(word_tokens(s,lang)) for s in sents]) if num_s else 0
    num_t = len(toks)
    mean_t = num_t/num_s if num_s else 0
    freq = Counter(toks)
    top10 = ", ".join([w for w,_ in freq.most_common(10)])
    low10 = ", ".join([w for w,_ in freq.most_common()[-10:]])
    n,v,p = count_pos(toks, lang)
    chunks = len(chunk_text(limpar_texto(txt_raw)))
    return {
        "Conjunto": label, "Idioma": lang, "Sentenças": num_s,
        "Média Sent.": round(mean_s,2), "Tokens": num_t,
        "Média Tokens/Sent": round(mean_t,2), "Top‑10": top10,
        "Down‑10": low10, "Substantivos": n, "Verbos": v,
        "Preposições": p, "Chunks": chunks
    }

# ==== STREAMLIT ====

st.title("📄 EDA de PDFs — sem referências")

files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if not files: st.stop()

choices = [f.name for f in files]
sel = st.sidebar.selectbox("Escolha o PDF", choices)
raw = extrair_texto_pdf(files[choices.index(sel)])
lang = detectar_idioma(raw)
clean = limpar_texto(raw)
sw = set(stopwords.words("portuguese" if lang.startswith("pt") else "english"))

wp_fast     = tok_fast.tokenize(clean)
wp_fast_ns  = [t for t in wp_fast if t not in sw]
wp_slow     = tok_slow.tokenize(clean)
wp_slow_ns  = [t for t in wp_slow if t not in sw]
rec_fast    = reconstruct(clean, tok_fast)
rec_fast_ns = [t for t in rec_fast if t not in sw]

df = pd.DataFrame([
    metrics(raw, wp_fast,    lang, "Fast + stop"),
    metrics(raw, wp_fast_ns, lang, "Fast – stop"),
    metrics(raw, wp_slow,    lang, "Slow + stop"),
    metrics(raw, wp_slow_ns, lang, "Slow – stop"),
    metrics(raw, rec_fast,   lang, "Reconstr + stop"),
    metrics(raw, rec_fast_ns,lang, "Reconstr – stop"),
]).set_index("Conjunto")

# ====== VISUALIZAÇÕES ======
base = df.loc["Fast + stop"]
st.sidebar.markdown("### 🔎 Resumo rápido")
st.sidebar.write(f"**Idioma:** {base['Idioma']}")
st.sidebar.write(f"**Sentenças:** {base['Sentenças']}, **Tokens:** {base['Tokens']}")

st.header(f"📚 `{sel}` — {lang.upper()}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Fast tokens", len(wp_fast))
c2.metric("Fast – stop", len(wp_fast_ns))
c3.metric("Slow tokens", len(wp_slow))
c4.metric("Reconstr tokens", len(rec_fast))

st.subheader("🗒️ Sumário Executivo")
ins=[]
ins.append("Sentenças longas" if base["Média Sent."]>20 else "Sentenças moderadas")
ratio= base["Substantivos"]/(base["Verbos"] or 1)
ins.append("Tom técnico" if ratio>2 else "Tom narrativo")
ins.append(f"{base['Chunks']} chunk(s) de 512 tok")
ins.append(f"Token mais freq.: {base['Top‑10'].split(',')[0]}")
for i in ins: st.write(f"• {i}")

st.subheader("📊 Métricas completas")
st.dataframe(df, use_container_width=True)

st.subheader("📈 Gráficos comparativos")
def plot_bar(df, col, title):
    fig, ax = plt.subplots(figsize=(8,4))
    df[col].plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(col)
    plt.xticks(rotation=45)
    st.pyplot(fig)

plot_bar(df, ["Sentenças","Tokens"], "Sentenças vs Tokens")
plot_bar(df, ["Substantivos","Verbos","Preposições"], "Comparativo PoS")
plot_bar(df, ["Chunks"], "Chunks necessários")

# ====== n-grams, nuvens, LDA, Dendrograma ======

st.subheader("🔍 Frequências & n‑grams")
def barra(freq, title, label): 
    fig, ax = plt.subplots()
    top = dict(freq.most_common(20))
    ax.bar(top.keys(), top.values())
    ax.set_title(title); ax.set_ylabel(label)
    plt.xticks(rotation=45); st.pyplot(fig)

def nuvem(freq, title, label):
    wc = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(freq)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear')
    ax.axis("off"); st.pyplot(fig)

def blocos(tok,label):
    f_all = Counter(tok); f_ns = Counter([t for t in tok if t not in sw])
    barra(f_all,f"Top {label} (com stop)",""); barra(f_ns,f"Top {label} (sem stop)","")
    nuvem(f_all,f"Nuvem {label} (com stop)",""); nuvem(f_ns,f"Nuvem {label} (sem stop)","")
    for n in (2,3):
        ng=Counter(" ".join(tok[i:i+n]) for i in range(len(tok)-n+1))
        barra(ng,f"Top {n}-grams ({label})",""); nuvem(ng,f"Nuvem {n}-grams ({label})","")

st.subheader("• WordPiece Fast"); blocos(wp_fast,"Fast")
st.subheader("• Reconstruídos"); blocos(rec_fast,"Reconstr")

# LDA
st.subheader("🧠 Tópicos (LDA global)")
all_txt=[limpar_texto(extrair_texto_pdf(f)) for f in files]
vec = CountVectorizer(stop_words="english")
X = vec.fit_transform(all_txt)
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42)
lda.fit(X)
terms = vec.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top_words = [terms[i] for i in topic.argsort()[:-TOP_N - 1:-1]]
    st.write(f"**Tópico {idx+1}:** {', '.join(top_words)}")

# Dendrograma
st.subheader("🌳 Dendrograma global")
def dendro(textos, top_n=50):
    vec = TfidfVectorizer(stop_words="english", max_features=top_n)
    X = vec.fit_transform(textos).toarray()
    linkage_matrix = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, labels=[f.name for f in files], ax=ax)
    st.pyplot(fig)
dendro(all_txt)

# Resumo Groq
st.subheader("📜 Resumo (Groq)")
def resumo_groq(txt):
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": f"Resuma o seguinte texto:\n\n{txt}"}]
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]

with st.spinner("Gerando…"):
    try:
        resumo = resumo_groq(clean[:4096])  # segurança contra limite
        st.write(resumo)
    except Exception as e:
        st.error(f"Erro ao resumir: {e}")
