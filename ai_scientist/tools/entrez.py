import os
import requests
import time
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class EntrezSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchEntrez",
        description: str = (
            "Search for relevant literature using NCBI Entrez API. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY")
        self.ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "your.email@example.com")
        if not self.ENTREZ_API_KEY:
            warnings.warn(
                "No Entrez API key found. Requests will be subject to stricter rate limits. "
                "Set the ENTREZ_API_KEY environment variable for higher limits."
            )

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None
        
        # First, search for paper IDs
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results,
            "retmode": "json",
            "sort": "relevance",
            "email": self.ENTREZ_EMAIL,
        }
        
        if self.ENTREZ_API_KEY:
            params["api_key"] = self.ENTREZ_API_KEY
        
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_response = requests.get(search_url, params=params)
        print(f"Search Response Status Code: {search_response.status_code}")
        search_response.raise_for_status()
        
        search_data = search_response.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return None
        
        # Then, fetch details for those IDs
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "email": self.ENTREZ_EMAIL,
        }
        
        if self.ENTREZ_API_KEY:
            fetch_params["api_key"] = self.ENTREZ_API_KEY
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        print(f"Fetch Response Status Code: {fetch_response.status_code}")
        fetch_response.raise_for_status()
        
        # Parse XML response
        papers = self._parse_pubmed_xml(fetch_response.text)
        return papers

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response into a list of paper dictionaries."""
        papers = []
        root = ET.fromstring(xml_text)
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # Extract basic article info
                article_data = {}
                
                # Title
                title_element = article.find(".//ArticleTitle")
                article_data["title"] = title_element.text if title_element is not None else "Unknown Title"
                
                # Authors
                authors = []
                author_list = article.findall(".//Author")
                for author in author_list:
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    if last_name is not None and fore_name is not None:
                        authors.append({"name": f"{fore_name.text} {last_name.text}"})
                    elif last_name is not None:
                        authors.append({"name": last_name.text})
                article_data["authors"] = authors
                
                # Journal/Venue
                journal_element = article.find(".//Journal/Title")
                article_data["venue"] = journal_element.text if journal_element is not None else "Unknown Venue"
                
                # Year
                year_element = article.find(".//PubDate/Year")
                article_data["year"] = year_element.text if year_element is not None else "Unknown Year"
                
                # Abstract
                abstract_elements = article.findall(".//AbstractText")
                abstract_text = " ".join([elem.text for elem in abstract_elements if elem.text])
                article_data["abstract"] = abstract_text if abstract_text else "No abstract available."
                
                # Citation count (not directly available from PubMed)
                article_data["citationCount"] = "N/A"
                
                # Generate BibTeX
                pmid_element = article.find(".//PMID")
                pmid = pmid_element.text if pmid_element is not None else "unknown"
                
                # Create a citation key from first author's last name and year
                first_author = "unknown"
                if authors and "name" in authors[0]:
                    first_author = authors[0]["name"].split()[-1].lower()
                
                bibtex = self._generate_bibtex(
                    pmid, 
                    first_author, 
                    article_data["year"], 
                    article_data["title"],
                    article_data["authors"],
                    article_data["venue"]
                )
                
                article_data["citationStyles"] = {"bibtex": bibtex}
                
                papers.append(article_data)
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue
        
        return papers

    def _generate_bibtex(self, pmid, first_author, year, title, authors, journal):
        """Generate a BibTeX entry for a paper."""
        # Create citation key
        cite_key = f"{first_author}{year}pmid{pmid}"
        
        # Format authors
        author_str = " and ".join([author["name"] for author in authors])
        
        # Create BibTeX entry
        bibtex = f"""@article{{{cite_key},
  title = {{{title}}},
  author = {{{author_str}}},
  journal = {{{journal}}},
  year = {{{year}}},
  pmid = {{{pmid}}},
}}"""
        return bibtex

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    """Standalone function to search for papers using the Entrez API."""
    ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY")
    ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "your.email@example.com")
    
    if not query:
        return None
    
    # First, search for paper IDs
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": result_limit,
        "retmode": "json",
        "sort": "relevance",
        "email": ENTREZ_EMAIL,
    }
    
    if ENTREZ_API_KEY:
        params["api_key"] = ENTREZ_API_KEY
    
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_response = requests.get(search_url, params=params)
    print(f"Search Response Status Code: {search_response.status_code}")
    search_response.raise_for_status()
    
    search_data = search_response.json()
    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    
    if not id_list:
        return None
    
    # Then, fetch details for those IDs
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
        "email": ENTREZ_EMAIL,
    }
    
    if ENTREZ_API_KEY:
        fetch_params["api_key"] = ENTREZ_API_KEY
    
    fetch_response = requests.get(fetch_url, params=fetch_params)
    print(f"Fetch Response Status Code: {fetch_response.status_code}")
    fetch_response.raise_for_status()
    
    # Parse XML response
    tool = EntrezSearchTool()
    papers = tool._parse_pubmed_xml(fetch_response.text)
    
    time.sleep(1.0)  # Be nice to the API
    return papers
