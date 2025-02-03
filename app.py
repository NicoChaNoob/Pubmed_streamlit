import os
import shutil
import time
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize  # Not used here now but left in case needed later
from datetime import datetime

############################################
# NLTK Configuration for Streamlit Cloud

# Define a local folder to store NLTK resources
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
os.environ['NLTK_DATA'] = nltk_data_dir

def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name, download_dir=nltk_data_dir)

# Download necessary resources
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')

# To satisfy the search for "tokenizers/punkt_tab/english", create that folder
punkt_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt")
punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab")
english_tab_dir = os.path.join(punkt_tab_dir, "english")
if not os.path.exists(english_tab_dir):
    os.makedirs(english_tab_dir, exist_ok=True)
    english_pickle = os.path.join(punkt_dir, "english.pickle")
    if os.path.exists(english_pickle):
        shutil.copy(english_pickle, english_tab_dir)
    else:
        st.warning("The file english.pickle was not found in the 'punkt' directory.")

############################################
# API Key (provided)
API_KEY = "0028f009242fa540c86c474f429c330e8108"

############################################
# Streamlit Interface

st.title("🔎 PubMed Data Extraction")
st.write(
    "This application extracts articles from PubMed based on a query and a publication date range. "
    "All retrieved data—including additional bibliographic information—is aggregated into an Excel file."
)

# Input for PubMed query
query_input = st.text_input("Enter your PubMed query:", "cancer AND immunotherapy")

# Publication date range input
date_range = st.date_input(
    "Select publication date range (start and end):",
    [pd.to_datetime("2000-01-01"), pd.to_datetime("2025-01-01")]
)
if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = date_range
    start_str = start_date.strftime("%Y/%m/%d")
    end_str = end_date.strftime("%Y/%m/%d")
    date_query = f" AND ({start_str}[dp] : {end_str}[dp])"
else:
    date_query = ""

# Number of articles to retrieve
max_results = st.slider("Number of articles to retrieve:", 5, 50, 10)

# Construct the full query including date range
full_query = query_input + date_query
st.markdown(f"**Complete PubMed query:** `{full_query}`")

############################################
# Function to retrieve articles from PubMed and extract additional fields

def get_pubmed_articles(query, max_results, api_key):
    # Build the search URL with API key
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={query}&retmax={max_results}&retmode=xml&api_key={api_key}"
    )
    search_response = requests.get(search_url)
    try:
        search_root = ET.fromstring(search_response.content)
    except Exception as e:
        st.error(f"Error parsing the search response: {e}")
        return []
    
    pmid_list = [elem.text for elem in search_root.findall(".//Id")]
    
    articles = []
    for pmid in pmid_list:
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={pmid}&retmode=xml&api_key={api_key}"
        )
        fetch_response = requests.get(fetch_url)
        if fetch_response.status_code == 429:
            st.warning(f"Error retrieving PMID {pmid}: status 429 (Too many requests). Waiting 1 second and retrying.")
            time.sleep(1)
            fetch_response = requests.get(fetch_url)
        if fetch_response.status_code != 200:
            st.warning(f"Error retrieving PMID {pmid}: status {fetch_response.status_code}")
            continue
        
        try:
            fetch_root = ET.fromstring(fetch_response.content)
        except Exception as e:
            st.warning(f"XML parsing error for PMID {pmid}: {e}")
            continue
        
        # Some efetch responses wrap the article inside <PubmedArticle>
        pubmed_article = fetch_root.find("PubmedArticle")
        if pubmed_article is None:
            pubmed_article = fetch_root
        
        # Retrieve the main Article element from MedlineCitation/Article
        article_elem = pubmed_article.find("MedlineCitation/Article")
        if article_elem is None:
            article_elem = pubmed_article.find("Article")
        if article_elem is None:
            continue
        
        # Title
        title_elem = article_elem.find("ArticleTitle")
        title = title_elem.text if title_elem is not None else "N/A"
        
        # Abstract (concatenate multiple AbstractText if present)
        abstract = "N/A"
        abs_elem = article_elem.find("Abstract")
        if abs_elem is not None:
            abs_texts = [elem.text for elem in abs_elem.findall("AbstractText") if elem.text]
            if abs_texts:
                abstract = " ".join(abs_texts)
        
        # Journal details
        journal_elem = article_elem.find("Journal")
        if journal_elem is not None:
            journal_title_elem = journal_elem.find("Title")
            journal_title = journal_title_elem.text if journal_title_elem is not None else "N/A"
            issn_elem = journal_elem.find("ISSN")
            issn = issn_elem.text if issn_elem is not None else "N/A"
            journal_issue_elem = journal_elem.find("JournalIssue")
            if journal_issue_elem is not None:
                volume_elem = journal_issue_elem.find("Volume")
                volume = volume_elem.text if volume_elem is not None else "N/A"
                issue_elem = journal_issue_elem.find("Issue")
                issue = issue_elem.text if issue_elem is not None else "N/A"
                pubdate_elem = journal_issue_elem.find("PubDate")
                year_elem = pubdate_elem.find("Year") if pubdate_elem is not None else None
                year = year_elem.text if year_elem is not None else "N/A"
            else:
                volume = issue = year = "N/A"
            iso_elem = journal_elem.find("ISOAbbreviation")
            source = iso_elem.text if iso_elem is not None else journal_title
            publisher_elem = journal_elem.find("Publisher")
            if publisher_elem is not None:
                place_elem = publisher_elem.find("PublisherLocation")
                place_pub = place_elem.text if place_elem is not None else "N/A"
            else:
                place_pub = "N/A"
        else:
            journal_title = issn = volume = issue = year = source = place_pub = "N/A"
        
        # Authors and Affiliations
        authors_list = []
        affiliations_list = []
        for author in article_elem.findall("AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None and fore is not None:
                full_name = f"{fore.text} {last.text}"
            elif last is not None:
                full_name = last.text
            elif fore is not None:
                full_name = fore.text
            else:
                full_name = ""
            if full_name:
                authors_list.append(full_name)
            aff_elem = author.find("AffiliationInfo/Affiliation")
            if aff_elem is not None and aff_elem.text:
                affiliations_list.append(aff_elem.text.strip())
        authors_joined = "; ".join(authors_list) if authors_list else "N/A"
        affiliations_joined = "; ".join(affiliations_list) if affiliations_list else "N/A"
        
        # Language
        language_elem = article_elem.find("Language")
        language = language_elem.text if language_elem is not None else "N/A"
        
        # Publication Type(s)
        pub_types = article_elem.findall("PublicationTypeList/PublicationType")
        pub_types_text = "; ".join([pt.text for pt in pub_types if pt.text]) if pub_types else "N/A"
        
        # Pages
        pages_elem = article_elem.find("Pagination/MedlinePgn")
        pages = pages_elem.text if pages_elem is not None else "N/A"
        
        # MeSH Terms
        mesh_terms = []
        for mesh in pubmed_article.findall(".//MeshHeadingList/MeshHeading"):
            desc = mesh.find("DescriptorName")
            if desc is not None and desc.text:
                mesh_terms.append(desc.text.strip())
        mesh_terms_joined = "; ".join(mesh_terms) if mesh_terms else "N/A"
        
        # Grant Numbers
        grant_numbers = []
        for grant in pubmed_article.findall(".//GrantList/Grant"):
            grant_id = grant.find("GrantID")
            if grant_id is not None and grant_id.text:
                grant_numbers.append(grant_id.text.strip())
        grant_numbers_joined = "; ".join(grant_numbers) if grant_numbers else "N/A"
        
        # PMC ID from PubmedData
        pmc_elem = fetch_root.find(".//PubmedData/ArticleIdList/ArticleId[@IdType='pmc']")
        pmc_id = pmc_elem.text if pmc_elem is not None else "N/A"
        
        # Construct URL to the PubMed record
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        article_data = {
            "PMID": pmid,
            "PMC_ID": pmc_id,
            "Title": title,
            "Abstract": abstract,
            "Journal": journal_title,
            "ISSN": issn,
            "Volume": volume,
            "Issue": issue,
            "Year": year,
            "Pages": pages,
            "Place": place_pub,
            "Language": language,
            "Publication_Type": pub_types_text,
            "Source": source,
            "Authors": authors_joined,
            "Affiliations": affiliations_joined,
            "MeSH_Terms": mesh_terms_joined,
            "Grant_Numbers": grant_numbers_joined,
            "URL": url
        }
        articles.append(article_data)
        time.sleep(0.5)  # Delay to avoid rate limiting
    return articles

############################################
# Action when clicking the "Run Search" button

if st.button("Run Search"):
    with st.spinner("Retrieving articles from PubMed..."):
        articles = get_pubmed_articles(full_query, max_results, API_KEY)
    
    if articles:
        df = pd.DataFrame(articles)
        st.success(f"Retrieved {len(df)} articles")
        st.dataframe(df)
        
        # Add extra columns: Execution Date and the complete Query used
        execution_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["Execution Date"] = execution_date
        df["Query"] = full_query
        
        # Define the Excel file path to aggregate results
        output_file = "pubmed_filtered_results.xlsx"
        
        # If the file exists, load it and append new results; otherwise, use current results
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_excel(output_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                st.warning(f"Error loading existing Excel file: {e}")
                combined_df = df
        else:
            combined_df = df
        
        # Save the combined DataFrame to the Excel file
        combined_df.to_excel(output_file, index=False)
        
        # Provide a download button for the Excel file
        with open(output_file, "rb") as file:
            st.download_button(
                label="Download Excel file",
                data=file,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("No articles were found for the given query.")
