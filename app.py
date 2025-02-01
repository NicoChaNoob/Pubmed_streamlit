import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import os

# Télécharger les ressources NLTK nécessaires (à la première exécution)
nltk.download("punkt")
nltk.download("stopwords")

st.title("🔎 Recherche et Filtrage d'Articles PubMed")

st.write(
    "Cette application vous permet d'extraire des articles de PubMed en fonction d'une requête, "
    "de limiter la recherche à une période de publication précise, d'obtenir un lien vers chaque article, "
    "et d'agréger les résultats dans un fichier Excel avec des informations supplémentaires."
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
    date_query = f" AND ({start_str}[Date - Publication] : {end_str}[Date - Publication])"
else:
    date_query = ""

# Nombre d'articles à récupérer
max_results = st.slider("Nombre d'articles à récupérer :", 5, 50, 10)

# Mots-clés pour le filtrage des articles
keywords_input = st.text_input("Mots-clés pour le filtrage (séparés par des virgules) :", "treatment, therapy, trial")
keywords = [kw.strip().lower() for kw in keywords_input.split(",")]

# Requête complète intégrant la période
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
    search_root = ET.fromstring(search_response.content)
    pmid_list = [elem.text for elem in search_root.findall(".//Id")]
    
    articles = []
    # Pour chaque PMID, récupérer les détails de l'article
    for pmid in pmid_list:
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={pmid}&retmode=xml"
        )
        fetch_response = requests.get(fetch_url)
        
        # Vérifier que la réponse est OK
        if fetch_response.status_code != 200:
            st.warning(f"Erreur lors de la récupération du PMID {pmid}: statut {fetch_response.status_code}")
            continue  # passer au PMID suivant
        
        # Tenter de parser la réponse XML
        try:
            fetch_root = ET.fromstring(fetch_response.content)
        except Exception as e:
            st.warning(f"Erreur de parsing XML pour le PMID {pmid}: {e}")
            # Vous pouvez également logger fetch_response.text pour diagnostiquer le contenu renvoyé.
            continue

        # Extraire les informations souhaitées
        title_elem = fetch_root.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "N/A"
        
        journal_elem = fetch_root.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "N/A"
        
        year_elem = fetch_root.find(".//PubDate/Year")
        year = year_elem.text if year_elem is not None else "N/A"
        
        abstract_elem = fetch_root.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else "N/A"
        
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
# Fonction pour filtrer les articles selon la présence des mots-clés
def filter_articles(articles, keywords):
    filtered = []
    stop_words = set(stopwords.words("english"))
    
    for article in articles:
        # Concaténer le titre et l'abstract pour l'analyse
        text = (article["Title"] + " " + article["Abstract"]).lower()
        # Nettoyer le texte en ne gardant que des lettres et espaces
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        
        # Calculer un score de pertinence en comptant la présence des mots-clés
        score = sum(1 for word in tokens if word in keywords)
        if score > 0:
            article["Pertinence"] = score
            filtered.append(article)
    
    # Trier les articles par score décroissant
    filtered.sort(key=lambda x: x["Pertinence"], reverse=True)
    return filtered

############################################
# Action lors du clic sur le bouton de lancement
if st.button("Lancer la recherche"):
    with st.spinner("Récupération des articles depuis PubMed..."):
        articles = get_pubmed_articles(full_query, max_results)
    filtered_articles = filter_articles(articles, keywords)
    
    if filtered_articles:
        df = pd.DataFrame(filtered_articles)
        st.success(f"Articles pertinents trouvés : {len(df)}")
        st.dataframe(df)
        
        # Ajouter trois nouvelles colonnes :
        # 1. La date d'exécution de la requête
        # 2. Le contenu de la requête complète
        # 3. La liste des mots-clés utilisés pour le filtrage
        execution_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["Execution Date"] = execution_date
        df["Query"] = full_query
        df["Keywords"] = keywords_input
        
        # Définir le chemin du fichier Excel
        output_file = "pubmed_filtered_results.xlsx"
        
        # Si le fichier existe déjà, le charger et y concaténer les nouveaux résultats
        if os.path.exists(output_file):
            existing_df = pd.read_excel(output_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df
        
        # Enregistrer le DataFrame combiné dans le fichier Excel
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
