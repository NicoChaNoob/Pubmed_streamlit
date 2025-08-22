nimport os
import shutil
import time
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from datetime import datetime
import openai

############################################
# Configuration NLTK pour Streamlit Cloud

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
# Clés API

API_KEY = "0028f009242fa540c86c474f429c330e8108"
openai.api_key = st.secrets["OPENAI_API_KEY"]

############################################
# Interface Streamlit

st.title("🔎 Extraction et Analyse d'Articles PubMed (sans limite)")

st.write(
    "Cette application extrait **tous** les articles PubMed correspondant à une requête et une plage de dates, "
    "puis agrège les résultats dans un fichier Excel et propose une analyse via ChatGPT."
)

st.info("⚠️ Note : le traitement par ChatGPT ne sera pas exécuté si la requête renvoie plus de 50 articles.")

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
# Fonction de récupération paginée

def get_all_pubmed_pmids(query, api_key, batch=500):
    # 1) Obtenir le nombre total d’articles
    url0 = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={query}&retmode=xml&api_key={api_key}&retmax=0"
    )
    r0 = requests.get(url0)
    root0 = ET.fromstring(r0.content)
    count = int(root0.findtext("Count", "0"))
    pmids = []
    # 2) Paginer par lots de 'batch'
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

# Fonction d'analyse ChatGPT, avec découpe en « chunks »

def analyze_extracted_data(articles):
    # 1) Fonction interne pour découper en phrases sans NLTK
    import re
    def split_into_sentences(text):
        return re.split(r'(?<=[\.\!\?])\s+', text.strip())

    # 2) Mots-clés pour repérer les phrases liées aux effets indésirables
    keywords = ["adverse", "side effect", "toxicity", "safety", "tolerance"]
    relevant_texts = []

    # 3) Parcourir chaque article et extraire les phrases contenant un mot-clé
    for a in articles:
        title = a["Title"]
        abstract = a["Abstract"]

        sentences = split_into_sentences(abstract)
        filtered = [s for s in sentences if any(k in s.lower() for k in keywords)]
        if filtered:
            joined = "\n".join(filtered)
        else:
            joined = "Aucune mention explicite d'effet indésirable."
        relevant_texts.append(f"Title: {title}\nRelevant sentences:\n{joined}")

    # 4) Si aucun texte pertinent, on renvoie un message
    if not relevant_texts:
        return "Aucun passage lié aux effets indésirables n’a été détecté dans les abstracts."

    # 5) Découper en chunks de 10 articles
    chunk_size = 10
    summaries = []
    for i in range(0, len(relevant_texts), chunk_size):
        chunk = relevant_texts[i : i + chunk_size]
        text_to_analyze = "\n\n---\n\n".join(chunk)
        prompt = (
            "Tu es un expert en pharmacovigilance. Ci-dessous figurent des extraits d'articles PubMed qui "
            "regroupent uniquement les phrases mentionnant des effets indésirables ou des questions de tolérance :\n\n"
            + text_to_analyze
            + "\n\nPeux-tu me fournir, pour ces articles :\n"
            "1. La liste des effets indésirables principaux mentionnés.\n"
            "2. Leur fréquence (lorsque disponible).\n"
            "3. Toute information sur la tolérance ou la sécurité.\n\n"
            "Enfin, fournis une synthèse globale à travers tous les articles pour les 3 points listés précédemment. Cette synthèse doit êtreclaire et structurée et indique pour chaque point clef sur quel article tu t'appuies."
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "Tu es un expert en pharmacovigilance qui travaille pour un laboratoire pharmaceutique."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=1,
                max_completion_tokens=40000
            )
            summaries.append(resp.choices[0].message.content)
        except Exception as e:
            summaries.append(f"Erreur lors de l'appel à l'API ChatGPT pour le chunk {i//chunk_size + 1} : {e}")

        time.sleep(1)  # Petite pause entre les requêtes

    # 6) Concaténer les résumés de chaque chunk
    full_analysis = "\n\n===== Chunk suivant =====\n\n".join(summaries)
    return full_analysis

############################################
# Bouton d'extraction et d'analyse

if st.button("Run Search & Analyze"):
    with st.spinner("Fetching PMIDs..."):
        pmids = get_all_pubmed_pmids(full_query, API_KEY)
    with st.spinner(f"Fetching details for {len(pmids)} articles..."):
        articles = fetch_pubmed_details(pmids, API_KEY)

    st.success(f"Extracted {len(articles)} articles")
    df = pd.DataFrame(articles)
    st.dataframe(df)

    # Agrégation Excel
    exec_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["Execution Date"], df["Query"] = exec_date, full_query
    out = "pubmed_results.xlsx"
    if os.path.exists(out):
        old = pd.read_excel(out)
        df = pd.concat([old, df], ignore_index=True)
    df.to_excel(out, index=False)
    with open(out, "rb") as f:
        st.download_button("Download Excel", f, out)

    # Si plus de 50 articles, on ne lance pas ChatGPT
    if len(articles) > 50:
        st.warning("ℹ️ Le nombre d’articles extrait est supérieur à 50. L’analyse ChatGPT ne sera pas exécutée.")
    else:
        # Analyse ChatGPT
        with st.spinner("Analyzing with ChatGPT…"):
            analysis = analyze_extracted_data(articles)
        st.markdown("### ChatGPT Analysis")
        st.write(analysis)





