"""
SEC EDGAR data fetching module.

Handles downloading and parsing of SEC 10-K filings from the EDGAR database.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class SecAPI:
    """Rate-limited SEC EDGAR API client."""

    BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    HEADERS = {
        "User-Agent": "NLP Alpha Research nlp-research@example.com",
        "Accept-Encoding": "gzip, deflate",
    }
    MIN_REQUEST_INTERVAL = 0.12  # ~8 requests/sec, within SEC limit of 10/sec

    def __init__(self):
        self._last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def get(self, url):
        self._rate_limit()
        response = requests.get(url, headers=self.HEADERS)
        response.raise_for_status()
        return response.text


def get_filing_urls(sec_api, cik, doc_type="10-K", start=0, count=60, max_date="2018-01-01"):
    """
    Fetch a list of filing URLs from SEC EDGAR for a given CIK.

    Parameters
    ----------
    sec_api : SecAPI
        Rate-limited API client.
    cik : str
        SEC Central Index Key for the company.
    doc_type : str
        Filing type to search for (default "10-K").
    start : int
        Starting index for results.
    count : int
        Number of results to return.
    max_date : str
        Only include filings on or before this date (YYYY-MM-DD).

    Returns
    -------
    list of tuple
        Each tuple contains (filing_url, filing_type, filing_date).
    """
    import pandas as pd

    rss_url = (
        f"{SecAPI.BASE_URL}?action=getcompany"
        f"&CIK={cik}&type={doc_type}&start={start}&count={count}"
        f"&owner=exclude&output=atom"
    )
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode("ascii"), "xml").feed
    max_dt = pd.to_datetime(max_date)

    entries = []
    for entry in feed.find_all("entry", recursive=False):
        filing_date = entry.content.find("filing-date").getText()
        if pd.to_datetime(filing_date) <= max_dt:
            entries.append((
                entry.content.find("filing-href").getText(),
                entry.content.find("filing-type").getText(),
                filing_date,
            ))
    return entries


def download_filings(sec_api, cik_lookup, doc_type="10-K"):
    """
    Download raw filing text for all tickers.

    Parameters
    ----------
    sec_api : SecAPI
        Rate-limited API client.
    cik_lookup : dict
        Mapping of ticker -> CIK string.
    doc_type : str
        Filing type to download.

    Returns
    -------
    dict
        Nested dict: {ticker: {filing_date: raw_text}}.
    """
    raw_filings = {}

    for ticker, cik in cik_lookup.items():
        filing_urls = get_filing_urls(sec_api, cik, doc_type)
        raw_filings[ticker] = {}

        for index_url, file_type, file_date in tqdm(
            filing_urls, desc=f"Downloading {ticker} filings", unit="filing"
        ):
            if file_type == doc_type:
                file_url = index_url.replace("-index.htm", ".txt").replace(".txtl", ".txt")
                raw_filings[ticker][file_date] = sec_api.get(file_url)

    return raw_filings


def extract_documents(text):
    """
    Extract individual documents from a SEC filing.

    Parameters
    ----------
    text : str
        Raw filing text containing <DOCUMENT>...</DOCUMENT> sections.

    Returns
    -------
    list of str
        Extracted document strings (without the tags).
    """
    doc_start_pattern = re.compile(r"<DOCUMENT>")
    doc_end_pattern = re.compile(r"</DOCUMENT>")

    starts = [m.end() for m in doc_start_pattern.finditer(text)]
    ends = [m.start() for m in doc_end_pattern.finditer(text)]

    return [text[s:e] for s, e in zip(starts, ends)]


def get_document_type(doc):
    """
    Extract the document type from a SEC document.

    Parameters
    ----------
    doc : str
        Document string containing a <TYPE> tag.

    Returns
    -------
    str
        Document type in lowercase.
    """
    type_pattern = re.compile(r"<TYPE>[^\n]+")
    match = type_pattern.search(doc)
    if match:
        return match.group()[len("<TYPE>"):].strip().lower()
    return ""


def get_ten_k_filings(raw_filings, cik_lookup):
    """
    Filter raw filings to extract only 10-K documents.

    Parameters
    ----------
    raw_filings : dict
        Nested dict: {ticker: {filing_date: raw_text}}.
    cik_lookup : dict
        Mapping of ticker -> CIK string.

    Returns
    -------
    dict
        {ticker: list of dict} where each dict has keys 'cik', 'file', 'file_date'.
    """
    ten_ks = {}

    for ticker, filings in raw_filings.items():
        ten_ks[ticker] = []
        for file_date, filing_text in tqdm(
            filings.items(), desc=f"Parsing {ticker} documents", unit="filing"
        ):
            for doc in extract_documents(filing_text):
                if get_document_type(doc) == "10-k":
                    ten_ks[ticker].append({
                        "cik": cik_lookup[ticker],
                        "file": doc,
                        "file_date": file_date,
                    })

    return ten_ks
