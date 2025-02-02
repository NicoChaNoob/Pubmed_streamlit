import os
import shutil
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime

############################################
# Configuration NLTK for Streamlit Cloud

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

# To satisfy the request for "tokenizers/punkt_tab/english",
# create that folder by copying the "english.pickle" file from "punkt".
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
# Streamlit Interface

st.title("🔎 PubMed Data Extraction")

st.write(
    "This application allows you to extract articles from PubMed based on a query "
    "and a publication date range. The results are aggregated into an Excel file."
)

# Input for the PubMed query
query_input = st.text_input("Enter your PubMed query:", "cancer AND immunotherapy")

# Input for publication date range
date_range = st.date_input(
    "Select publication date range (start and end):",
    [pd.to_datetime("2000-01-01"), pd.to_datetime("2025-01-01")]
)
if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = date_range
    # Format for PubMed: YYYY/MM/DD
    start_str = start_date.strftime("%Y/%m/%d")
    end_str = end_date.strftime("%Y/%m/%d")
    date_query = f" AND ({start_str}[dp] : {end_str}[dp])"
else:
    date_query = ""

# Number of articles to retrieve
max_results = st.slider("Number of articles to retrieve:", 5, 50, 10)

# Construct the full query including the date range
full_query = query_input + date_query
st.markdown(f"**Complete PubMed query:** `{full_query}`")

############################################
# Function to retrieve articles from PubMed via the API

def get_pubmed_articles(query, max_results):
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={query}&retmax={max_results}&retmode=xml"
    )
    search_response = requests.get(search_url)
    try:
        search_root = ET.fromstring(search_response.content)
    except Exception as e:
        st.error(f"Error parsing the search response: {e}")
        return []
    
    pmid_list = [elem.text for elem in search_root.findall(".//Id")]
    
    articles = []
    # For each PMID, retrieve article details
    for pmid in pmid_list:
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={pmid}&retmode=xml"
        )
        fetch_response = requests.get(fetch_url)
        if fetch_response.status_code != 200:
            st.warning(f"Error retrieving PMID {pmid}: status {fetch_response.status_code}")
            continue
        
        try:
            fetch_root = ET.fromstring(fetch_response.content)
        except Exception as e:
            st.warning(f"XML parsing error for PMID {pmid}: {e}")
            continue
        
        # Extract relevant information
        title_elem = fetch_root.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "N/A"
        
        journal_elem = fetch_root.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "N/A"
        
        year_elem = fetch_root.find(".//PubDate/Year")
        year = year_elem.text if year_elem is not None else "N/A"
        
        abstract_elem = fetch_root.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else "N/A"
        
        # Link to detailed PubMed page
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
# Action when clicking the "Run Search" button

if st.button("Run Search"):
    with st.spinner("Retrieving articles from PubMed..."):
        articles = get_pubmed_articles(full_query, max_results)
    
    if articles:
        df = pd.DataFrame(articles)
        st.success(f"Retrieved {len(df)} articles")
        st.dataframe(df)
        
        # Add extra columns:
        # - Execution Date
        # - The complete query used
        execution_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["Execution Date"] = execution_date
        df["Query"] = full_query
        
        # Define the Excel file path where results will be aggregated
        output_file = "pubmed_filtered_results.xlsx"
        
        # If the file already exists, load it and concatenate new results
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
