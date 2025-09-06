import os
import shutil
import time
import re
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from datetime import datetime
import openai

############################################
# Configuration NLTK pour Streamlit Cloud (héritée)
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
os.environ['NLTK_DATA'] = nltk_data_dir

def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name, download_dir=nltk_data_dir)

download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')

# création du dossier punkt_tab/english si nécessaire
punkt_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt")
punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab")
english_tab_dir = os.path.join(punkt_tab_dir, "english")
if not os.path.exists(english_tab_dir):
    os.makedirs(english_tab_dir, exist_ok=True)
    english_pickle = os.path.join(punkt_dir, "english.pickle")
    if os.path.exists(english_pickle):
        shutil.copy(english_pickle, english_tab_dir)

############################################
# Clés & modèle OpenAI

API_KEY = "0028f009242fa540c86c474f429c330e8108"
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Modèle à grande fenêtre de contexte (changeable via Secrets si besoin)
MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4o")
# Paramètres conseillés pour extraction factuelle
TEMPERATURE = 0.1
MAX_TOKENS_OUT = 1200
CHUNK_SIZE = 30 if MODEL_NAME == "gpt-4o" else 40

############################################
# Interface Streamlit

st.title("🔎 Extraction et Analyse d'Articles PubMed (sans limite)")

st.write(
    "Cette application extrait **tous** les articles PubMed correspondant à une requête et une plage de dates, "
    "puis agrège les résultats dans un fichier Excel et propose une analyse via ChatGPT."
)
st.info("⚠️ Le traitement par ChatGPT ne sera pas exécuté si la requête renvoie plus de 50 articles.")

st.caption(f"Modèle d'analyse : {MODEL_NAME} · Température : {TEMPERATURE} · max_tokens: {MAX_TOKENS_OUT} · chunk={CHUNK_SIZE}")

query_input = st.text_input("Entrez votre requête PubMed :", "cancer AND immunotherapy")

date_range = st.date_input(
    "Sélectionnez la période de publication (début et fin) :",
    [pd.to_datetime("2000-01-01"), pd.to_datetime("2025-01-01")]
)
if len(date_range) == 2:
    start_str = date_range[0].strftime("%Y/%m/%d")
    end_str   = date_range[1].strftime("%Y/%m/%d")
    date_query = f" AND ({start_str}[dp] : {end_str}[dp])"
else:
    date_query = ""

full_query = query_input + date_query
st.markdown(f"**Requête complète PubMed :** `{full_query}`")

############################################
# Récupération PMIDs

def get_all_pubmed_pmids(query, api_key, batch=500):
    # Count
    url0 = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={query}&retmode=xml&api_key={api_key}&retmax=0"
    )
    r0 = requests.get(url0)
    root0 = ET.fromstring(r0.content)
    count = int(root0.findtext("Count", "0"))
    pmids = []
    # Paging
    for start in range(0, count, batch):
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={query}&retmode=xml&api_key={api_key}"
            f"&retstart={start}&retmax={batch}"
        )
        resp = requests.get(url)
        part = ET.fromstring(resp.content)
        pmids.extend(idn.text for idn in part.findall(".//Id"))
        time.sleep(0.3)
    return pmids

############################################
# Détails articles

def fetch_pubmed_details(pmids, api_key):
    articles = []
    for pmid in pmids:
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={pmid}&retmode=xml&api_key={api_key}"
        )
        r = requests.get(fetch_url)
        if r.status_code == 429:
            time.sleep(1)
            r = requests.get(fetch_url)
        if r.status_code != 200:
            continue
        xr = ET.fromstring(r.content)
        art = xr.find(".//PubmedArticle/MedlineCitation/Article")
        if art is None:
            continue

        def get_text(elem, path, default="N/A"):
            e = elem.find(path)
            return e.text if e is not None and e.text else default

        title    = get_text(art, "ArticleTitle")
        abstract = " ".join(e.text for e in art.findall("Abstract/AbstractText") if e.text) or "N/A"
        journal  = get_text(art, "Journal/Title")
        pmc_id   = xr.findtext(".//PubmedData/ArticleIdList/ArticleId[@IdType='pmc']", "N/A")

        authors = []
        for au in art.findall("AuthorList/Author"):
            fn, ln = au.findtext("ForeName",""), au.findtext("LastName","")
            name = (fn + " " + ln).strip()
            if name:
                authors.append(name)
        auth = "; ".join(authors) or "N/A"

        issn   = get_text(art, "Journal/ISSN")
        vol    = get_text(art, "Journal/JournalIssue/Volume")
        issue  = get_text(art, "Journal/JournalIssue/Issue")
        pages  = get_text(art, "Pagination/MedlinePgn")
        year   = get_text(art, "Journal/JournalIssue/PubDate/Year")
        lang   = get_text(art, "Language")
        mesh   = "; ".join(
            mh.findtext("DescriptorName","")
            for mh in xr.findall(".//MeshHeadingList/MeshHeading")
        ) or "N/A"
        grants = "; ".join(
            g.findtext("GrantID","")
            for g in xr.findall(".//GrantList/Grant")
        ) or "N/A"

        articles.append({
            "PMID": pmid,
            "PMC_ID": pmc_id,
            "Title": title,
            "Abstract": abstract,
            "Journal": journal,
            "ISSN": issn,
            "Volume": vol,
            "Issue": issue,
            "Pages": pages,
            "Year": year,
            "Language": lang,
            "Authors": auth,
            "MeSH_Terms": mesh,
            "Grant_Numbers": grants,
            "URL": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })
        time.sleep(0.3)
    return articles

############################################
# Analyse ChatGPT avec CITATION EXACTE + LOG TOKENS

def analyze_extracted_data(articles):
    # Split phrases via regex (pas de dépendance NLTK)
    def split_into_sentences(text):
        return re.split(r'(?<=[\.\!\?])\s+', text.strip())

    # Keywords pour préfiltrer (réduit le bruit et la taille des prompts)
    keywords = ["adverse", "side effect", "toxicity", "safety", "tolerab", "tolerance", "AE", "AEs", "serious adverse"]

    # Construire des extraits pertinents par article (titre + phrases pertinentes + PMID)
    relevant_texts = []
    for a in articles:
        pmid = a["PMID"]
        title = a["Title"]
        abstract = a["Abstract"] or ""
        sentences = split_into_sentences(abstract)
        filtered = [s for s in sentences if any(k in s.lower() for k in keywords)]
        if not filtered:
            filtered = ["Aucune mention explicite d'effet indésirable."]
        block = f"PMID: {pmid}\nTitle: {title}\nCandidate sentences:\n" + "\n".join(f"- {s}" for s in filtered)
        relevant_texts.append(block)

    if not relevant_texts:
        return "Aucun passage lié aux effets indésirables n’a été détecté dans les abstracts.", {
            "model": MODEL_NAME,
            "chunks": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    summaries = []
    usage_total_prompt = 0
    usage_total_completion = 0
    usage_total = 0

    for i in range(0, len(relevant_texts), CHUNK_SIZE):
        chunk = relevant_texts[i : i + CHUNK_SIZE]
        text_to_analyze = "\n\n---\n\n".join(chunk)

        # IMPORTANT : on demande une sortie structurée avec citation exacte et PMID
        prompt = (
            "Tu es un expert en pharmacovigilance. On te fournit, par article, un PMID, un titre, et une liste de "
            "phrases candidates extraites depuis l'abstract, susceptibles de mentionner des effets indésirables ou des informations de tolérance ou de sécurité.\n\n"
            "Tâche : pour chaque article du lot, fournis \n"
            "1. La liste des effets indésirables principaux mentionnés.\n" 
            "2. Leur fréquence (lorsque disponible).\n" 
            "3. Toute information sur la tolérance ou la sécurité.\n\n" 
            "Enfin, fournis une synthèse globale à travers tous les articles pour les 3 points listés précédemment. Cette synthèse doit êtreclaire et structurée et indique pour chaque point clef sur quel article tu t'appuies."
            "Contraintes :\n"
            "- Ne crée pas d'information absente des phrases candidates.\n"
            "- Si aucun effet ou information sur la tolérance ou la sécurité n'est présent pour un article, renvoie: 'rien'.\n"
        )

        try:
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Tu es un assistant d'extraction factuelle strict qui est un expert en pharmacovigilane. Tu respectes les contraintes et ne cites que le texte fourni."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_OUT
            )

            # Récupérer texte + usage tokens
            chunk_text = resp.choices[0].message.content
            summaries.append(chunk_text)

            if hasattr(resp, "usage") and resp.usage is not None:
                usage_total_prompt     += getattr(resp.usage, "prompt_tokens", 0)
                usage_total_completion += getattr(resp.usage, "completion_tokens", 0)
                usage_total            += getattr(resp.usage, "total_tokens", 0)

        except Exception as e:
            summaries.append(f'{{"error":"Appel OpenAI échoué sur le chunk {i//CHUNK_SIZE+1}: {e}"}}')

        time.sleep(0.5)  # petite pause

    full_analysis = "\n\n===== CHUNK SUIVANT =====\n\n".join(summaries)

    usage_summary = {
        "model": MODEL_NAME,
        "chunks": (len(relevant_texts) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "prompt_tokens": usage_total_prompt,
        "completion_tokens": usage_total_completion,
        "total_tokens": usage_total
    }
    return full_analysis, usage_summary

############################################
# Bouton d'extraction et d'analyse

if st.button("Run Search & Analyze"):
    with st.spinner("Fetching PMIDs..."):
        pmids = get_all_pubmed_pmids(full_query, API_KEY)

    with st.spinner(f"Fetching details for {len(pmids)} articles..."):
        articles = fetch_pubmed_details(pmids, API_KEY)

    st.success(f"Extracted {len(articles)} articles")
    df_articles = pd.DataFrame(articles)
    st.dataframe(df_articles)

    # Fichier Excel multi-onglets : Articles + Token_Log
    out = "pubmed_results.xlsx"
    exec_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Préparer/mettre à jour la feuille Articles (append)
    if os.path.exists(out):
        old = pd.read_excel(out, sheet_name=None)
        old_articles = old.get("Articles")
        if old_articles is not None and not old_articles.empty:
            df_articles = pd.concat([old_articles, df_articles], ignore_index=True)

    # Token log de la session (initialisé ; complété plus bas)
    token_log_row = {
        "ExecutionDate": exec_date,
        "Query": full_query,
        "Model": MODEL_NAME,
        "Num_Articles": len(articles),
        "Chunks": 0,
        "Prompt_Tokens": 0,
        "Completion_Tokens": 0,
        "Total_Tokens": 0
    }
    df_token_log_session = pd.DataFrame([token_log_row])

    # Si >100 articles : pas d'analyse, log à 0 tokens
    if len(articles) > 100:
        st.warning("ℹ️ Le nombre d’articles extrait est supérieur à 100. L’analyse ChatGPT ne sera pas exécutée.")

        # Écriture Excel (Articles + Token_Log (append))
        if os.path.exists(out):
            # Charger anciens logs si présents
            old = pd.read_excel(out, sheet_name=None)
            old_token = old.get("Token_Log")
            if old_token is not None and not old_token.empty:
                df_token_log_session = pd.concat([old_token, df_token_log_session], ignore_index=True)

        with pd.ExcelWriter(out, engine="openpyxl", mode="w") as writer:
            df_articles.to_excel(writer, index=False, sheet_name="Articles")
            df_token_log_session.to_excel(writer, index=False, sheet_name="Token_Log")

        with open(out, "rb") as f:
            st.download_button("Download Excel", f, out)

    else:
        # Analyse ChatGPT
        with st.spinner("Analyzing with ChatGPT…"):
            analysis_text, usage = analyze_extracted_data(articles)

        st.markdown("### ChatGPT Analysis (JSON par chunk)")
        st.write(analysis_text)

        # Mettre à jour le token log avec les usages réels
        token_log_row.update({
            "Chunks": usage["chunks"],
            "Prompt_Tokens": usage["prompt_tokens"],
            "Completion_Tokens": usage["completion_tokens"],
            "Total_Tokens": usage["total_tokens"]
        })
        df_token_log_session = pd.DataFrame([token_log_row])

        # Si le fichier existe, on concatène Articles & Token_Log
        if os.path.exists(out):
            old = pd.read_excel(out, sheet_name=None)
            old_articles = old.get("Articles")
            if old_articles is not None and not old_articles.empty:
                df_articles = pd.concat([old_articles, df_articles], ignore_index=True)
            old_token = old.get("Token_Log")
            if old_token is not None and not old_token.empty:
                df_token_log_session = pd.concat([old_token, df_token_log_session], ignore_index=True)

        # Écriture Excel (2 feuilles)
        with pd.ExcelWriter(out, engine="openpyxl", mode="w") as writer:
            df_articles.to_excel(writer, index=False, sheet_name="Articles")
            df_token_log_session.to_excel(writer, index=False, sheet_name="Token_Log")

        with open(out, "rb") as f:
            st.download_button("Download Excel", f, out)

        # Affichage synthèse des tokens
        st.markdown("#### Token usage (cette exécution)")
        st.write(pd.DataFrame([usage]))


