# app_eda_pdfs_ngram_metrica.py
# ------------------------------------------------------------------
# EDA de PDFs com WordPiece, reconstrução, n‑grams, métricas
# adicionais (sentenças, PoS), gráficos explicativos e resumo Groq.
# ------------------------------------------------------------------

# === IMPORTS =========
import re, string, json, requests, os
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import streamlit as st
import PyPDF2, nltk, spacy, pkg_resources
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect
from transformers import BertTokenizerFast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# === CONFIG ==========
MODEL_NAME = "bert-base-uncased"
N_TOPICS = 5
TOP_N    = 20

# tenta carregar modelo spaCy pt → se não achar, continua sem
try:
    _ = pkg_resources.get_distribution("spacy")
    try:
        nlp_pt = spacy.load("pt_core_news_sm")
    except:
        nlp_pt = None
except:
    nlp_pt = None

# === FUNÇÕES ÚTEIS ===
def extrair_texto_pdf(file):
    txt=""
    for p in PyPDF2.PdfReader(file).pages:
        txt += p.extract_text() + " "
    return txt

def limpar_texto(t):
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return re.sub(f"[{re.escape(string.punctuation)}]", "", t).strip()

def sentencas(texto, lang):
    if lang.startswith("pt") and nlp_pt:
        return [s.text.strip() for s in nlp_pt(texto).sents]
    return sent_tokenize(texto)

def tokens_palavras(sent, lang):
    if lang.startswith("pt") and nlp_pt:
        return [t.text for t in nlp_pt(sent)]
    return word_tokenize(sent)

def pos_counts(tokens, lang):
    nouns=verbs=preps=0
    if lang.startswith("pt"):
        if nlp_pt:
            doc = nlp_pt(" ".join(tokens))
            for tok in doc:
                if tok.pos_=="NOUN": nouns+=1
                if tok.pos_=="VERB": verbs+=1
                if tok.pos_=="ADP":  preps+=1
        # fallback simples
    else:
        tagged = nltk.pos_tag(tokens)
        for _,tag in tagged:
            if tag.startswith("NN"): nouns+=1
            if tag.startswith("VB"): verbs+=1
            if tag=="IN":           preps+=1
    return nouns,verbs,preps

# ----- n‑gram contagem
def ngram_freq(toks,n): return Counter([" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)])

# ----- métricas principais
def gerar_metricas(texto_cln, tokens, lang):
    sent_list = sentencas(texto_cln, lang)
    num_sent  = len(sent_list)
    mean_sent_len = np.mean([len(tokens_palavras(s,lang)) for s in sent_list]) if num_sent else 0

    num_tok   = len(tokens)
    mean_tok  = num_tok/num_sent if num_sent else 0

    freq      = Counter(tokens)
    top10     = ", ".join([w for w,_ in freq.most_common(10)])
    down10    = ", ".join([w for w,_ in freq.most_common()[-10:]]) if len(freq)>=10 else ", ".join(freq)

    nouns,verbs,preps = pos_counts(tokens, lang)

    return dict(Nº_Sentenças=num_sent,
                Média_Sent_Len=round(mean_sent_len,2),
                Nº_Tokens=num_tok,
                Média_Tokens_por_Sent=round(mean_tok,2),
                Top10=top10,
                Down10=down10,
                Substantivos=nouns,
                Verbos=verbs,
                Preposições=preps)

# === VISUAL HELPERS (barra_frequencia, nuvem, histograma) — idênticos ao script anterior ===
# (por brevidade não repetidos aqui; use os mesmos que já funcionam)

# ... ===== (cole aqui as funções de visualização iguais ao script anterior) ===== ...

# === STREAMLIT ====================================================
st.set_page_config(page_title="EDA PDFs + métricas extras", layout="wide")
st.title("📄 EDA de PDFs com métricas avançadas (n‑grams, PoS, Groq)")

upl = st.sidebar.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)
if not upl: st.stop()

names  = [u.name for u in upl]
texts  = [extrair_texto_pdf(f) for f in upl]
pdf_sel= st.sidebar.selectbox("Escolha o PDF:", names)
idx    = names.index(pdf_sel)
texto  = texts[idx]

lang  = detectar_idioma(texto)
stop  = set(stopwords.words("portuguese" if lang.startswith("pt") else "english"))
tok   = BertTokenizerFast.from_pretrained(MODEL_NAME)
texto_cln = limpar_texto(texto)
tokens_wp = tok.tokenize(texto_cln)
tokens_rec= tokens_reconstruidos(texto_cln, tok)

# === Métricas textuais ============================================
metrics_pdf = gerar_metricas(texto_cln, tokens_rec, lang)
st.sidebar.markdown("### 📏 Métricas rápidas")
for k,v in metrics_pdf.items():
    st.sidebar.write(f"**{k.replace('_',' ')}:** {v}")

# (o resto do script continua igual, usando as visuais e seções já enviadas.
#  Basta colar novamente a parte de gráficos, n‑grams, distribuições,
#  tabela estatística, tópicos e dendrograma.)
