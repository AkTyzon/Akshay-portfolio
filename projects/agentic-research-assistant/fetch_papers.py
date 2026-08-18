import requests
import fitz
import re
from io import BytesIO
import xml.etree.ElementTree as ET

def fetch_arxiv_pdf(arxiv_id: str) -> str:
    """
    Download a paper from arXiv by ID and return text.
    Example: '2307.03172'
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PDF from {url}")
    
    pdf_stream = BytesIO(response.content)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    text = []
    for page in doc:
        text.append(page.get_text())

    return re.sub(r"\s+", " ", " ".join(text)).strip()


def fetch_papers_by_keyword(query: str, max_results: int = 3):
    """
    Search arXiv for a keyword and return list of (title, id, summary).
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception("Failed to fetch arXiv search results")

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        link = entry.find("atom:id", ns).text.strip()
        arxiv_id = link.split("/")[-1]
        results.append({"id": arxiv_id, "title": title, "summary": summary})

    return results
