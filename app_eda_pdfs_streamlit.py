# app_eda_pdfs_streamlit_completo_final.py

import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from wordcloud import WordCloud
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import PyPDF2
import nltk
from nltk.corpus import stopwords
from langdetect import detect
from transformers import BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

# === Configurações iniciais ===
MODEL_NAME = "bert-base-uncased"
N_TOPICS = 5

nltk.download('stopwords')

# === Funções auxiliares ===

def extrair_texto_pdf(uploaded_file):
    texto = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        texto += page.extract_text() + " "
    return texto

def detectar_idioma(texto):
    try:
        idioma = detect(texto)
    except:
        idioma = 'en'
    return idioma

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(f'[{re.escape(string.punctuation)}]', '', texto)
    return texto.strip()

def gerar_nuvem_palavras(frequencia, title):
    wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(frequencia)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.title(title, fontsize=16)
    st.pyplot(fig)

def plotar_top_palavras(freq_dict, title, n=20):
    mais_comuns = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)[:n])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(mais_comuns.values()), y=list(mais_comuns.keys()), ax=ax)
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

def analisar_tamanho_tokens(tokens, title="Distribuição dos Tamanhos dos Tokens"):
    tamanhos = [len(token) for token in tokens]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(tamanhos, bins=20, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Tamanho do Token")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

def calcular_coerencia_topicos(modelo_lda, matriz_X):
    topico_distribuicao = modelo_lda.transform(matriz_X)
    similaridade = cosine_similarity(topico_distribuicao)
    tril = np.tril(similaridade, k=-1)
    coerencia = tril.sum() / (tril != 0).sum()
    return coerencia

def gerar_dendrograma(lista_textos, top_n_palavras=50):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n_palavras)
    X = vectorizer.fit_transform(lista_textos)
    palavras = vectorizer.get_feature_names_out()
    linkage_matrix = linkage(X.toarray().T, method='ward')
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(linkage_matrix, labels=palavras, leaf_rotation=90, leaf_font_size=12, ax=ax)
    ax.set_title("Dendrograma de Palavras-Chave")
    st.pyplot(fig)

def modelar_topicos_globais(lista_textos, n_topics):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(lista_textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    palavras = vectorizer.get_feature_names_out()
    topicos = []
    for idx, topic in enumerate(lda.components_):
        topicos.append([palavras[i] for i in topic.argsort()[:-11:-1]])
    coerencia = calcular_coerencia_topicos(lda, X)
    return topicos, coerencia

def reconstruir_tokens_com_offset(texto, tokenizer):
    encoding = tokenizer(texto, return_offsets_mapping=True, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    palavras_reconstruidas = []
    palavra_atual = ""
    for token, (start, end) in zip(tokens, offsets):
        if token in ["[CLS]", "[SEP]"]:
            continue
        if token.startswith("##"):
            palavra_atual += token[2:]
        else:
            if palavra_atual:
                palavras_reconstruidas.append(palavra_atual)
            palavra_atual = token
    if palavra_atual:
        palavras_reconstruidas.append(palavra_atual)
    return palavras_reconstruidas

def resumir_com_groq(texto, api_key, modelo="meta-llama/llama-4-scout-17b-16e-instruct"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": modelo,
        "messages": [{
            "role": "user",
            "content": f"Resuma de forma clara e objetiva o seguinte texto:\n\n{texto}"
        }],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        resposta_json = response.json()
        resumo = resposta_json['choices'][0]['message']['content']
        return resumo.strip()
    else:
        raise Exception(f"Erro na chamada da API Groq: {response.status_code} - {response.text}")

# === Início do App Streamlit ===

st.set_page_config(page_title="EDA PDFs + Groq Summarization", layout="wide")
st.title("📄 Análise Técnica de PDFs - Upload + Tokens + Tópicos + Resumo via Groq")

uploaded_files = st.sidebar.file_uploader("📂 Faça upload de um ou mais PDFs", type=['pdf'], accept_multiple_files=True)

if not uploaded_files:
    st.warning("🚨 Por favor, envie pelo menos um arquivo PDF.")
    st.stop()

file_names = [f.name for f in uploaded_files]
selected_file = st.sidebar.selectbox("Escolha o PDF para análise:", file_names)

texts = [extrair_texto_pdf(f) for f in uploaded_files]

# === Processamento do PDF selecionado ===
index_selected = file_names.index(selected_file)
texto_original_selected = texts[index_selected]
idioma_detectado = detectar_idioma(texto_original_selected)
idioma = 'portuguese' if idioma_detectado.startswith('pt') else 'english'
stop_words_set = set(stopwords.words(idioma))
texto_limpo_selected = limpar_texto(texto_original_selected)

# === Tokenização ===
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
tokens = tokenizer.tokenize(texto_limpo_selected)
tokens_reconstruidos = reconstruir_tokens_com_offset(texto_limpo_selected, tokenizer)

# === Contagens ===
contagem_tokens = Counter(tokens)
contagem_tokens_reconstruidos = Counter(tokens_reconstruidos)

# === Frequências ===
frequencia_com_stopwords = dict(contagem_tokens)
frequencia_sem_stopwords = {w: f for w, f in contagem_tokens.items() if w not in stop_words_set}
frequencia_com_stopwords_reconstruidos = dict(contagem_tokens_reconstruidos)
frequencia_sem_stopwords_reconstruidos = {w: f for w, f in contagem_tokens_reconstruidos.items() if w not in stop_words_set}

# === Exibição ===
st.header(f"📚 Análise do PDF: `{selected_file}`")

col1, col2 = st.columns(2)
col1.metric("🗣️ Idioma detectado", idioma_detectado.upper())
col2.metric("🧩 Tokens WordPiece", len(tokens))

# === Resumo via Groq ===
st.subheader("📜 Resumo Inteligente via Groq")

try:
    api_key_groq = st.secrets["GROQ_API_KEY"]

    with st.spinner("Chamando a API Groq para gerar o resumo..."):
        resumo = resumir_com_groq(texto_limpo_selected, api_key_groq)
        st.success(resumo)

except KeyError:
    st.error("🚨 A chave 'GROQ_API_KEY' não foi encontrada no secrets. Configure corretamente o `.streamlit/secrets.toml`.")
except Exception as e:
    st.error(f"Erro na API Groq: {e}")

st.divider()

# === Frequência e Nuvens ===
st.subheader("📈 Frequência de Palavras WordPiece")
plotar_top_palavras(frequencia_com_stopwords, "Top WordPiece com stopwords")
plotar_top_palavras(frequencia_sem_stopwords, "Top WordPiece sem stopwords")

st.subheader("📈 Frequência de Palavras Reconstruídas")
plotar_top_palavras(frequencia_com_stopwords_reconstruidos, "Top Reconstruídas com stopwords")
plotar_top_palavras(frequencia_sem_stopwords_reconstruidos, "Top Reconstruídas sem stopwords")

st.subheader("☁️ Nuvens de Palavras")
gerar_nuvem_palavras(frequencia_sem_stopwords, "Nuvem WordPiece")
gerar_nuvem_palavras(frequencia_sem_stopwords_reconstruidos, "Nuvem Reconstruída")

st.subheader("🔵 Distribuição dos Tamanhos dos Tokens")
analisar_tamanho_tokens(tokens, "Distribuição WordPiece")
analisar_tamanho_tokens(tokens_reconstruidos, "Distribuição Tokens Reconstruídos")

# === Tópicos e Dendrograma ===
st.subheader("🧠 Modelagem Global de Tópicos")

texts_limpos = [limpar_texto(t) for t in texts]
topicos, coerencia = modelar_topicos_globais(texts_limpos, N_TOPICS)

st.write(f"🔍 Coerência dos tópicos: **{coerencia:.4f}**")
for idx, topico in enumerate(topicos):
    st.info(f"Tópico {idx+1}: {', '.join(topico)}")

st.subheader("🌳 Dendrograma de Palavras-Chave")
gerar_dendrograma(texts_limpos)

