"""
Text preprocessing module for SEC filings.

Handles HTML removal, tokenization, lemmatization, and stopword filtering.
"""

import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm


WORD_PATTERN = re.compile(r"\w+")


def remove_html_tags(text):
    """Remove HTML tags from text using BeautifulSoup."""
    return BeautifulSoup(text, "html.parser").get_text()


def clean_text(text):
    """Lowercase text and strip HTML tags."""
    return remove_html_tags(text.lower())


def tokenize(text):
    """Extract words from text using regex."""
    return WORD_PATTERN.findall(text)


def lemmatize_words(words):
    """
    Lemmatize a list of words (verbs to base form).

    Parameters
    ----------
    words : list of str
        Words to lemmatize.

    Returns
    -------
    list of str
        Lemmatized words.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, "v") for w in words]


def get_stopwords():
    """Return lemmatized English stopwords."""
    return set(lemmatize_words(stopwords.words("english")))


def remove_stopwords(words, stop_words=None):
    """
    Remove stopwords from a list of words.

    Parameters
    ----------
    words : list of str
        Input word list.
    stop_words : set, optional
        Stopwords to remove. If None, uses NLTK English stopwords (lemmatized).

    Returns
    -------
    list of str
        Filtered word list.
    """
    if stop_words is None:
        stop_words = get_stopwords()
    return [w for w in words if w not in stop_words]


def preprocess_filings(ten_ks):
    """
    Run the full preprocessing pipeline on 10-K filings.

    Cleans HTML, tokenizes, lemmatizes, and removes stopwords for each filing.

    Parameters
    ----------
    ten_ks : dict
        {ticker: list of dict} where each dict has 'file' key with raw text.

    Returns
    -------
    dict
        Same structure with added 'file_clean' and 'file_lemma' keys.
    """
    stop_words = get_stopwords()

    for ticker, filings in ten_ks.items():
        for filing in tqdm(filings, desc=f"Preprocessing {ticker}", unit="10-K"):
            cleaned = clean_text(filing["file"])
            filing["file_clean"] = cleaned
            words = tokenize(cleaned)
            lemmatized = lemmatize_words(words)
            filing["file_lemma"] = remove_stopwords(lemmatized, stop_words)

    return ten_ks
