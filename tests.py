"""
Unit tests for the NLP 10-K alpha factor project.

Run with: pytest tests.py -v
"""

import numpy as np
import pandas as pd
import pytest

from sec_data import extract_documents, get_document_type
from text_processing import (
    remove_html_tags, clean_text, tokenize,
    lemmatize_words, remove_stopwords,
)
from sentiment import get_bag_of_words, get_tfidf
from similarity import compute_jaccard_similarity, compute_cosine_similarity
from factor_evaluation import compute_sharpe_ratio


# ---------------------------------------------------------------------------
# sec_data tests
# ---------------------------------------------------------------------------

class TestExtractDocuments:
    def test_extracts_correct_count(self):
        text = "<DOCUMENT>doc one</DOCUMENT><DOCUMENT>doc two</DOCUMENT>"
        docs = extract_documents(text)
        assert len(docs) == 2

    def test_extracts_content_without_tags(self):
        text = "<DOCUMENT>hello world</DOCUMENT>"
        docs = extract_documents(text)
        assert docs[0] == "hello world"
        assert "<DOCUMENT>" not in docs[0]

    def test_empty_input(self):
        assert extract_documents("no documents here") == []

    def test_multiple_documents_correct_content(self):
        text = "<DOCUMENT>first</DOCUMENT>noise<DOCUMENT>second</DOCUMENT>"
        docs = extract_documents(text)
        assert docs[0] == "first"
        assert docs[1] == "second"


class TestGetDocumentType:
    def test_returns_lowercase_type(self):
        doc = "<TYPE>10-K\n<SEQUENCE>1\nsome content"
        assert get_document_type(doc) == "10-k"

    def test_handles_exhibit(self):
        doc = "<TYPE>EX-21.1\n<SEQUENCE>3"
        assert get_document_type(doc) == "ex-21.1"

    def test_empty_doc(self):
        assert get_document_type("no type tag here") == ""


# ---------------------------------------------------------------------------
# text_processing tests
# ---------------------------------------------------------------------------

class TestRemoveHtmlTags:
    def test_strips_html(self):
        html = "<p>Hello <b>world</b></p>"
        assert remove_html_tags(html) == "Hello world"

    def test_plain_text_unchanged(self):
        text = "no html here"
        assert remove_html_tags(text) == "no html here"


class TestCleanText:
    def test_lowercases_and_strips_html(self):
        text = "<P>Hello WORLD</P>"
        assert clean_text(text) == "hello world"


class TestTokenize:
    def test_extracts_words(self):
        assert tokenize("hello, world! 123") == ["hello", "world", "123"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestLemmatizeWords:
    def test_lemmatizes_verbs(self):
        words = ["running", "walked", "computing"]
        result = lemmatize_words(words)
        assert "run" in result
        assert "walk" in result
        assert "compute" in result

    def test_nouns_unchanged(self):
        words = ["cat", "dog"]
        result = lemmatize_words(words)
        assert result == ["cat", "dog"]

    def test_empty_list(self):
        assert lemmatize_words([]) == []


class TestRemoveStopwords:
    def test_removes_common_stopwords(self):
        words = ["the", "cat", "be", "on", "mat"]
        result = remove_stopwords(words)
        assert "cat" in result
        assert "mat" in result
        # "the" and "be" (lemmatized "is/are") should be removed
        assert "the" not in result


# ---------------------------------------------------------------------------
# sentiment tests
# ---------------------------------------------------------------------------

class TestGetBagOfWords:
    def test_correct_shape(self):
        vocab = pd.Series(["good", "bad", "risk"])
        docs = ["good good risk", "bad bad bad"]
        result = get_bag_of_words(vocab, docs)
        assert result.shape == (2, 3)

    def test_nonnegative_values(self):
        vocab = pd.Series(["profit", "loss"])
        docs = ["profit profit", "loss"]
        result = get_bag_of_words(vocab, docs)
        assert (result >= 0).all()

    def test_correct_counts(self):
        vocab = pd.Series(["alpha", "beta"])
        docs = ["alpha alpha beta"]
        result = get_bag_of_words(vocab, docs)
        assert result[0, 0] == 2  # alpha
        assert result[0, 1] == 1  # beta


class TestGetTfidf:
    def test_correct_shape(self):
        vocab = pd.Series(["profit", "loss", "risk"])
        docs = ["profit loss risk", "profit profit"]
        result = get_tfidf(vocab, docs)
        assert result.shape == (2, 3)

    def test_values_in_range(self):
        vocab = pd.Series(["up", "down"])
        docs = ["up up down", "down down"]
        result = get_tfidf(vocab, docs)
        assert (result >= 0).all()
        assert (result <= 1).all()


# ---------------------------------------------------------------------------
# similarity tests
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    def test_correct_length(self):
        matrix = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        result = compute_jaccard_similarity(matrix)
        assert len(result) == 2

    def test_values_in_range(self):
        matrix = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1]])
        result = compute_jaccard_similarity(matrix)
        assert all(0 <= v <= 1 for v in result)

    def test_identical_vectors(self):
        matrix = np.array([[1, 1, 1], [1, 1, 1]])
        result = compute_jaccard_similarity(matrix)
        assert result[0] == pytest.approx(1.0)

    def test_disjoint_vectors(self):
        matrix = np.array([[1, 0, 0], [0, 0, 1]])
        result = compute_jaccard_similarity(matrix)
        assert result[0] == pytest.approx(0.0)


class TestCosineSimilarity:
    def test_correct_length(self):
        matrix = np.array([[0.5, 0.3], [0.2, 0.8], [0.1, 0.9]])
        result = compute_cosine_similarity(matrix)
        assert len(result) == 2

    def test_values_in_range(self):
        matrix = np.array([[0.5, 0.3, 0.1], [0.2, 0.8, 0.4]])
        result = compute_cosine_similarity(matrix)
        assert all(0 <= v <= 1 for v in result)

    def test_identical_vectors(self):
        matrix = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        result = compute_cosine_similarity(matrix)
        assert result[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# factor_evaluation tests
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_returns_float(self):
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])
        result = compute_sharpe_ratio(returns)
        assert isinstance(result, float)

    def test_positive_for_positive_mean(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
        assert compute_sharpe_ratio(returns) > 0

    def test_negative_for_negative_mean(self):
        returns = pd.Series([-0.01, -0.02, -0.03, -0.01, -0.02])
        assert compute_sharpe_ratio(returns) < 0

    def test_zero_for_empty_series(self):
        assert compute_sharpe_ratio(pd.Series(dtype=float)) == 0.0

    def test_zero_for_constant_returns(self):
        returns = pd.Series([0.01, 0.01, 0.01])
        assert compute_sharpe_ratio(returns) == 0.0

    def test_annualization_factor(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        sharpe_daily = compute_sharpe_ratio(returns, annualization_factor=np.sqrt(252))
        sharpe_yearly = compute_sharpe_ratio(returns, annualization_factor=1.0)
        assert sharpe_daily > sharpe_yearly
