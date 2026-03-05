"""
Similarity computation module.

Computes Jaccard and Cosine similarity between consecutive annual filings
to measure year-over-year textual change in sentiment.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_jaccard_similarity(bow_matrix):
    """
    Compute Jaccard similarity between consecutive document bag-of-words vectors.

    Converts counts to boolean presence/absence before computing similarity.

    Parameters
    ----------
    bow_matrix : np.ndarray
        Shape (n_docs, n_words) bag-of-words matrix.

    Returns
    -------
    list of float
        Jaccard similarities for each consecutive pair (length = n_docs - 1).
    """
    bool_matrix = bow_matrix.astype(bool)
    similarities = []

    for i in range(len(bool_matrix) - 1):
        u = bool_matrix[i]
        v = bool_matrix[i + 1]
        intersection = np.sum(u & v)
        union = np.sum(u | v)
        score = intersection / union if union > 0 else 0.0
        similarities.append(float(score))

    return similarities


def compute_cosine_similarity(tfidf_matrix):
    """
    Compute cosine similarity between consecutive TF-IDF vectors.

    Parameters
    ----------
    tfidf_matrix : np.ndarray
        Shape (n_docs, n_words) TF-IDF matrix.

    Returns
    -------
    list of float
        Cosine similarities for each consecutive pair (length = n_docs - 1).
    """
    sim_matrix = cosine_similarity(tfidf_matrix)
    return np.diag(sim_matrix, k=1).tolist()


def compute_all_similarities(sentiment_matrices, method="cosine"):
    """
    Compute similarities across all tickers and sentiment categories.

    Parameters
    ----------
    sentiment_matrices : dict
        {ticker: {sentiment: np.ndarray}}.
    method : str
        "cosine" or "jaccard".

    Returns
    -------
    dict
        {ticker: {sentiment: list of float}}.
    """
    sim_func = compute_cosine_similarity if method == "cosine" else compute_jaccard_similarity
    results = {}

    for ticker, sentiments in sentiment_matrices.items():
        results[ticker] = {}
        for sentiment, matrix in sentiments.items():
            results[ticker][sentiment] = sim_func(matrix)

    return results
