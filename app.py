import os
import shutil
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime

############################################
# Configuration NLTK pour Streamlit Cloud

# Définir un dossier local pour stocker les ressources NLTK
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
os.environ['NLTK_DATA'] = nltk_data_dir

def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name, download_dir=nltk_data_dir)

# Télécharger les ressources nécessaires
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')

# Pour satisfaire la recherche du dossier "tokenizers/punkt_tab/english",
# nous créons ce dossier en copiant le fichier "english.pickle" depuis "punkt".
punkt_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt")
punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab")
english_tab_dir = os.path.join(punkt_tab_dir, "english")
if not os.path.exists(english_tab_dir):
    os.makedirs(english_tab_dir, exist_ok=True)
    english_pickle = os.path.join(punkt_dir, "english.pickle")
    if os.path.exists(english_pickle):
        shutil.copy(english_pickle, english_tab_dir)
    else:
        st.warning("Le fichier english.pickle n'a pas été trouvé dans le dossier 'punkt'.")

############################################
# Interface Streamlit

st.title("🔎 Recherche et Filtrage d'Articles PubMed")

st.write(
    "Cette application vous permet d'extraire des articles de PubMed en fonction d'une requête, "
    "de limiter la recherche à une période précise, et d'agréger les résultats dans un fichier Excel "
    "en ajoutant des informations supplémentaires."
)

# Saisie de la requête PubMed
query_input = st.text_input("Entrez votre requête PubMed :", "cancer AND immunotherapy")

# Sélection de la période de publication
date_range = st.date_input(
    "Sélectionnez la période de publication (début et fin) :",
    [pd.to_datetime("2000-01-01"), pd.to_datetime("2025-01-01")]
)
if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = date_range
    # Format PubMed : YYYY/MM/DD
    start_str = start_date.strftime("%Y/%m/%d")
    end_str = end_date.strftime("%Y/%m/%d")
    date_query = f" AND ({start_str}[dp] : {end_str}[dp])"
else:
    date_query = ""

# Nombre d'articles à récupérer
max_results = st.slider("Nombre d'articles à récupérer :", 5, 50, 10)

# Mots-clés pour le filtrage des articles
keywords_input = st.text_input("Mots-clés pour le filtrage (séparés par des virgules) :", "treatment, therapy, trial")
keywords = [kw.strip().lower() for kw in keywords_input.split(",")]

# Requête complète (incluant la période)
full_query = query_input + date_query
st.markdown(f"**Requête PubMed complète :** `{full_query}`")

############################################
# Fonction pour récupérer les articles depuis PubMed via l'API

def get_pubmed_articles(query, max_results):
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={query}&retmax={max_results}&retmode=xml"
    )
    search_response = requests.get(search_url)
    try:
        search_root = ET.fromstring(search_response.content)
    except Exception as e:
        st.error(f"Erreur lors du parsing de la réponse de recherche : {e}")
        return []
    
    pmid_list = [elem.text for elem in search_root.findall(".//Id")]
    
    articles = []
    # Pour chaque PMID, récupérer les détails de l'article
    for pmid in pmid_list:
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={pmid}&retmode=xml"
        )
        fetch_response = requests.get(fetch_url)
        if fetch_response.status_code != 200:
            st.warning(f"Erreur lors de la récupération du PMID {pmid}: statut {fetch_response.status_code}")
            continue
        
        try:
            fetch_root = ET.fromstring(fetch_response.content)
        except Exception as e:
            st.warning(f"Erreur de parsing XML pour le PMID {pmid}: {e}")
            continue
        
        # Extraction des informations
        title_elem = fetch_root.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "N/A"
        
        journal_elem = fetch_root.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "N/A"
        
        year_elem = fetch_root.find(".//PubDate/Year")
        year = year_elem.text if year_elem is not None else "N/A"
        
        abstract_elem = fetch_root.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else "N/A"
        
        # Lien vers la fiche détaillée sur PubMed
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        articles.append({
            "PMID": pmid,
            "Title": title,
            "Journal": journal,
            "Year": year,
            "Abstract": abstract,
            "URL": url
        })
    return articles

############################################
# Fonction pour filtrer les articles selon les mots-clés

def filter_articles(articles, keywords):
    filtered = []
    stop_words = set(stopwords.words("english"))
    
    for article in articles:
        # Concaténer titre et abstract pour l'analyse
        text = (article["Title"] + " " + article["Abstract"]).lower()
        # Nettoyer le texte : ne conserver que les lettres et espaces
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        
        # Calculer un score de pertinence (nombre d'apparitions des mots-clés)
        score = sum(1 for word in tokens if word in keywords)
        if score > 0:
            article["Pertinence"] = score
            filtered.append(article)
    
    # Trier les articles par score décroissant
    filtered.sort(key=lambda x: x["Pertinence"], reverse=True)
    return filtered

############################################
# Action lors du clic sur le bouton "Lancer la recherche"

if st.button("Lancer la recherche"):
    with st.spinner("Récupération des articles depuis PubMed..."):
        articles = get_pubmed_articles(full_query, max_results)
    filtered_articles = filter_articles(articles, keywords)
    
    if filtered_articles:
        df = pd.DataFrame(filtered_articles)
        st.success(f"Articles pertinents trouvés : {len(df)}")
        st.dataframe(df)
        
        # Ajouter trois colonnes supplémentaires :
        # - Date d'exécution de la requête
        # - Contenu de la requête complète
        # - Liste des mots-clés utilisés
        execution_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["Execution Date"] = execution_date
        df["Query"] = full_query
        df["Keywords"] = keywords_input
        
        # Chemin du fichier Excel dans lequel les résultats seront agrégés
        output_file = "pubmed_filtered_results.xlsx"
        
        # Si le fichier existe déjà, le charger et concaténer les nouveaux résultats
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_excel(output_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                st.warning(f"Erreur lors du chargement du fichier Excel existant : {e}")
                combined_df = df
        else:
            combined_df = df
        
        # Sauvegarder le DataFrame combiné dans le fichier Excel
        combined_df.to_excel(output_file, index=False)
        
        # Proposer le téléchargement du fichier Excel
        with open(output_file, "rb") as file:
            st.download_button(
                label="Télécharger le fichier Excel",
                data=file,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Aucun article pertinent n'a été trouvé avec les critères définis.")
