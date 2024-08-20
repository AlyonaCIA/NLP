"""This script downloads articles from arXiv related to solar physics
and machine learning, and saves them to a CSV file. The script uses
the arXiv API to fetch the data.
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict


def get_arxiv_abstracts(
        search_query: str,
        max_results: int = 1000) -> List[Dict[str, str]]:
    """
    Fetches abstracts from arXiv based on the given search query.

    Args:
    - search_query (str): The query to search for in arXiv.
    - max_results (int): The maximum number of articles to fetch.
    (Default is 1000)

    Returns:
    - List[Dict[str, str]]: A list of dictionaries containing 
    article information (title, summary, published date).
    """
    abstracts = []
    batch_size = 100
    for i in range(0, max_results, batch_size):
        url = f"http://export.arxiv.org/api/query?search_query={
            search_query}&start={i}&max_results={batch_size}"
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find(
                '{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find(
                '{http://www.w3.org/2005/Atom}summary').text.strip()
            published = entry.find(
                '{http://www.w3.org/2005/Atom}published').text.strip()
            abstracts.append({
                "title": title,
                "summary": summary,
                "published": published
            })
        print(f'Fetched {len(abstracts)} articles so far...')
    return abstracts


def save_abstracts_to_csv(
        abstracts: List[Dict[str, str]],
        filename: str = 'solar_ml_abstracts.csv') -> None:
    """
    Saves the fetched abstracts to a CSV file.

    Args:
    - abstracts (List[Dict[str, str]]): A list of dictionaries
    containing article information.
    - filename (str): The name of the CSV file to save the data.
    """
    df = pd.DataFrame(abstracts)
    df.to_csv(filename, index=False)
    print(f'Saved {len(abstracts)} articles to {filename}')


if __name__ == "__main__":
    search_query: str = "cat:astro-ph.SR+AND+all:machine+learning"
    articles: List[Dict[str, str]] = get_arxiv_abstracts(
        search_query, max_results=1000)
    save_abstracts_to_csv(articles)
