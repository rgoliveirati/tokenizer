# app_eda_pdfs_ngram_metrica.py
# ---------------------------------------------------------------
# EDA completa: WordPiece x Reconstruídos, métricas detalhadas,
# n‑grams, PoS, LDA, dendrograma e resumo via Groq.
# ---------------------------------------------------------------

# === IMPORTS ====================================================
import re, string, json, requests
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

# --- downloads NLTK --------------------------------------------
for pkg in ("punkt","stopwords","averaged_perceptron_tagger"): nltk.download(pkg)
try: nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError: nltk.download("averaged_perceptron_tagger_eng")
try: nltk.data.find("tokenizers/punkt_tab")
except LookupError: nltk.download("punkt_tab")

# === CONFIG =====================================================
MODEL_NAME = "bert-base-uncased"
N_TOPICS   = 5
TOP_N      = 20

# --- spaCy PT ---------------------------------------------------
try: nlp_pt = spacy.load("pt_core_news_sm")
except: nlp_pt = None

# ================================================================
def extrair_texto_pdf(file):
    return " ".join(p.extract_text() for p in PyPDF2.PdfReader(file).pages)

def detectar_idioma(txt):
    try: return detect(txt)
    except: return "en"

def limpar_texto(t):
    t = re.sub(r"\s+", " ", t.lower())
    return re.sub(f"[{re.escape(string.punctuation)}]", "", t).strip()

# ---------- sentenças & PoS -------------------------------------
def sentencas(txt, lang):
    if lang.startswith("pt") and nlp_pt:
        return [s.text.strip() for s in nlp_pt(txt).sents]
    return sent_tokenize(txt)

def tokens_palavras(sent, lang):
    if lang.startswith("pt") and nlp_pt: return [t.text for t in nlp_pt(sent)]
    return word_tokenize(sent)

def pos_counts(tokens, lang):
    n=v=p=0
    if lang.startswith("pt") and nlp_pt:
        for t in nlp_pt(" ".join(tokens)):
            if t.pos_=="NOUN": n+=1
            if t.pos_=="VERB": v+=1
            if t.pos_=="ADP":  p+=1
    else:
        for _,tg in nltk.pos_tag(tokens):
            if tg.startswith("NN"): n+=1
            if tg.startswith("VB"): v+=1
            if tg=="IN":            p+=1
    return n,v,p

# ---------- WordPiece reconstrução ------------------------------
def tokens_reconstruidos(txt, tokenizer):
    enc  = tokenizer(txt, return_offsets_mapping=True, add_special_tokens=True)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offs = enc["offset_mapping"]
    out, cur = [], ""
    for t,(s,e) in zip(toks,offs):
        if t in ("[CLS]","[SEP]"): continue
        if t.startswith("##"): cur += t[2:]
        else:
            if cur: out.append(cur)
            cur = t
    if cur: out.append(cur)
    return out

# ---------- métricas --------------------------------------------
def gerar_metricas(txt_raw, tokens, lang, label):
    sents   = sentencas(txt_raw, lang)
    num_s   = len(sents)
    mean_s  = np.mean([len(tokens_palavras(s,lang)) for s in sents]) if num_s else 0
    num_t   = len(tokens)
    mean_t  = num_t/num_s if num_s else 0
    freq    = Counter(tokens)
    top10   = ", ".join([w for w,_ in freq.most_common(10)])
    low10   = ", ".join([w for w,_ in freq.most_common()[-10:]]) if len(freq)>=10 else ", ".join(freq)
    n,v,p   = pos_counts(tokens, lang)
    return dict(
        Set                 = label,
        Num_Sentences       = num_s,
        Avg_Sent_Length     = round(mean_s,2),
        Num_Tokens          = num_t,
        Avg_Tok_per_Sent    = round(mean_t,2),
        Top10               = top10,
        Down10              = low10,
        Nouns               = n,
        Verbs               = v,
        Preps               = p
    )

# ---------- n‑grams & tópicos -----------------------------------
def ngram_freq(toks,n): return Counter(" ".join(toks[i:i+n]) for i in range(len(toks)-n+1))

def topicos_globais(textos, n_topics):
    vec = CountVectorizer(stop_words="english"); X = vec.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(X)
    words = vec.get_feature_names_out()
    return [[words[i] for i in comp.argsort()[-10:][::-1]] for comp in lda.components_]

# ---------- visual helpers (barra, nuvem, histo, dendro) --------
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
    dendrogram(linkage(X,method="ward"),labels=words,leaf_rotation=90,leaf_font_size=10,ax=ax)
    ax.set_title("Dendrograma de Palavras‑Chave"); st.pyplot(fig)
    st.caption("Linhas horizontais = união de clusters; altura = dissimilaridade.")

# ---------- resumo Groq -----------------------------------------
def resumo_groq(txt):
    try: key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.warning("⚠️ Defina GROQ_API_KEY em secrets."); return ""
    r=requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        data=json.dumps({"model":"meta-llama/llama-4-scout-17b-16e-instruct",
                         "messages":[{"role":"user","content":f"Resuma:\n\n{txt}, sempre no idioma português brasileiro."}],
                         "temperature":0.3,"max_tokens":1024}))
    return r.json()["choices"][0]["message"]["content"] if r.ok else "Erro Groq"

# === STREAMLIT ==================================================
st.set_page_config(page_title="EDA PDFs + métricas", layout="wide")
st.title("📄 EDA de PDFs — WordPiece × Reconstruídos, métricas & Groq")

upl = st.sidebar.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)
if not upl: st.stop()

names=[u.name for u in upl]
pdf_sel=st.sidebar.selectbox("Escolha o PDF:", names)
idx=names.index(pdf_sel)
texto_raw=extrair_texto_pdf(upl[idx])

lang = detectar_idioma(texto_raw)
stop = set(stopwords.words("portuguese" if lang.startswith("pt") else "english"))
tok  = BertTokenizerFast.from_pretrained(MODEL_NAME)

texto        = limpar_texto(texto_raw)
wp           = tok.tokenize(texto)
wp_ns        = [t for t in wp if t not in stop]
rec          = tokens_reconstruidos(texto, tok)
rec_ns       = [t for t in rec if t not in stop]

# -------- métricas DataFrame ------------------------------------
df_metrics = pd.DataFrame([
    gerar_metricas(texto_raw, wp,     lang, "WordPiece + stop"),
    gerar_metricas(texto_raw, wp_ns,  lang, "WordPiece – stop"),
    gerar_metricas(texto_raw, rec,    lang, "Reconstr. + stop"),
    gerar_metricas(texto_raw, rec_ns, lang, "Reconstr. – stop")
]).set_index("Set")

# Sidebar resumo rápido
quick = df_metrics.loc["WordPiece + stop"]
for lab in ("Num_Sentences","Num_Tokens","Avg_Tok_per_Sent"):
    st.sidebar.write(f"**{lab.replace('_',' ')}:** {quick[lab]}")

# Cabeçalho
st.header(f"📚 `{pdf_sel}` — Idioma: **{lang.upper()}**")
col = st.columns(4)
col[0].metric("WP tokens", len(wp))
col[1].metric("WP sem stop", len(wp_ns))
col[2].metric("Rec tokens", len(rec))
col[3].metric("Rec sem stop", len(rec_ns))

# Resumo Groq
st.subheader("📜 Resumo (Groq)")
with st.spinner("Gerando resumo…"):
    st.write(resumo_groq(texto))

st.divider()

# -------------- FREQUÊNCIAS & N‑GRAMS ---------------------------
def blocos(tok,label):
    f_all=Counter(tok)
    f_ns = Counter([t for t in tok if t not in stop])
    barra(f_all,f"Top {label} (com stop)","Comprimento = contagem.")
    barra(f_ns ,f"Top {label} (sem stop)","Stopwords removidas.")
    nuvem(f_all,f"Nuvem {label} (com stop)","Palavras maiores = mais frequentes.")
    nuvem(f_ns ,f"Nuvem {label} (sem stop)","Realça termos‑chave.")
    for n in (2,3):
        ngr=ngram_freq(tok,n)
        if ngr:
            barra(ngr,f"Top {n}-grams ({label})",f"{n}-grams mais frequentes.")
            nuvem(ngr,f"Nuvem {n}-grams ({label})","")

st.subheader("📈 WordPiece");     blocos(wp,"WordPiece")
st.subheader("📈 Reconstruídos"); blocos(rec,"Reconstruídos")

# -------------- DISTRIBUIÇÕES -----------------------------------
st.subheader("🔵 Distribuições de Tamanho")
histo([len(t) for t in wp],    "WordPiece", "")
histo([len(t) for t in wp_ns], "WordPiece (sem stop)", "")
histo([len(t) for t in rec],   "Reconstruídos", "")
histo([len(t) for t in rec_ns],"Reconstr. (sem stop)", "")

# -------------- TABELA MÉTRICAS ---------------------------------
st.subheader("📊 Métricas completas")
st.dataframe(df_metrics, use_container_width=True)

st.divider()

# -------------- TÓPICOS & DENDRO --------------------------------
st.subheader("🧠 Tópicos (LDA global)")
todos=[limpar_texto(extrair_texto_pdf(f)) for f in upl]
for i,top in enumerate(topicos_globais(todos,N_TOPICS),1):
    st.write(f"**Tópico {i}:** {', '.join(top)}")

st.subheader("🌳 Dendrograma global")
dendro(todos)
