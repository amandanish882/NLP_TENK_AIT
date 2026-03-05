"""
Sentiment analysis module using Loughran-McDonald financial word lists.

Provides bag-of-words and TF-IDF representations for sentiment categories.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from text_processing import lemmatize_words


SENTIMENT_CATEGORIES = [
    "negative", "positive", "uncertainty",
    "litigious", "constraining", "strong_modal",
]


def load_loughran_mcdonald(filepath):
    """
    Load and preprocess the Loughran-McDonald sentiment dictionary.

    Parameters
    ----------
    filepath : str
        Path to the Loughran-McDonald master dictionary CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'word' column and boolean columns for each sentiment category.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]

    available = [s for s in SENTIMENT_CATEGORIES if s in df.columns]
    if not available:
        raise ValueError(
            f"None of {SENTIMENT_CATEGORIES} found in CSV columns: {df.columns.tolist()}"
        )

    df = df[available + ["word"]]
    # Sentiment columns are integers (0 = absent, nonzero = present)
    df[available] = df[available].apply(pd.to_numeric, errors="coerce").fillna(0).astype(bool)
    df = df[df[available].any(axis=1)]

    df["word"] = lemmatize_words(df["word"].str.lower().tolist())
    df = df.drop_duplicates("word")

    return df


def get_sentiment_words(sentiment_df, sentiment):
    """
    Get the list of words for a given sentiment category.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        Loughran-McDonald sentiment DataFrame.
    sentiment : str
        Sentiment category name.

    Returns
    -------
    pd.Series
        Words belonging to that sentiment.
    """
    return sentiment_df[sentiment_df[sentiment]]["word"]


def get_bag_of_words(sentiment_words, docs):
    """
    Generate a bag-of-words count matrix for sentiment words across documents.

    Parameters
    ----------
    sentiment_words : array-like
        Vocabulary of sentiment words.
    docs : list of str
        Documents (space-joined lemmatized words).

    Returns
    -------
    np.ndarray
        Shape (n_docs, n_words) with word counts.
    """
    vectorizer = CountVectorizer(vocabulary=sentiment_words)
    return vectorizer.fit_transform(docs).toarray()


def get_tfidf(sentiment_words, docs):
    """
    Generate TF-IDF matrix for sentiment words across documents.

    Parameters
    ----------
    sentiment_words : array-like
        Vocabulary of sentiment words.
    docs : list of str
        Documents (space-joined lemmatized words).

    Returns
    -------
    np.ndarray
        Shape (n_docs, n_words) with TF-IDF values.
    """
    vectorizer = TfidfVectorizer(vocabulary=sentiment_words)
    return vectorizer.fit_transform(docs).toarray()


def compute_sentiment_matrices(ten_ks, sentiment_df, method="both"):
    """
    Compute sentiment BoW and/or TF-IDF matrices for all tickers and sentiments.

    Parameters
    ----------
    ten_ks : dict
        {ticker: list of dict} with 'file_lemma' key.
    sentiment_df : pd.DataFrame
        Loughran-McDonald sentiment DataFrame.
    method : str
        "bow", "tfidf", or "both".

    Returns
    -------
    tuple of dict
        (bow_matrices, tfidf_matrices) where each is
        {ticker: {sentiment: np.ndarray}}.
    """
    bow_results = {}
    tfidf_results = {}

    for ticker, filings in ten_ks.items():
        lemma_docs = [" ".join(f["file_lemma"]) for f in filings]
        bow_results[ticker] = {}
        tfidf_results[ticker] = {}

        for sentiment in SENTIMENT_CATEGORIES:
            words = get_sentiment_words(sentiment_df, sentiment)
            if method in ("bow", "both"):
                bow_results[ticker][sentiment] = get_bag_of_words(words, lemma_docs)
            if method in ("tfidf", "both"):
                tfidf_results[ticker][sentiment] = get_tfidf(words, lemma_docs)

    return bow_results, tfidf_results
