"""
Full pipeline runner for NLP 10-K Alpha Factor project.

Downloads SEC filings, preprocesses text, computes sentiment factors,
evaluates alpha signals, and saves all plots and tables to output/.

Usage:
    python run_pipeline.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import yfinance as yf

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from sec_data import SecAPI, download_filings, get_ten_k_filings
from text_processing import preprocess_filings
from sentiment import (
    load_loughran_mcdonald,
    SENTIMENT_CATEGORIES,
    compute_sentiment_matrices,
)
from similarity import compute_all_similarities
from factor_evaluation import (
    build_factor_dataframe,
    compute_factor_returns,
    compute_quantile_returns,
    compute_factor_rank_autocorrelation,
    compute_sharpe_ratio,
    plot_similarities,
    plot_cumulative_factor_returns,
    plot_quantile_returns,
    plot_factor_rank_autocorrelation,
)

sns.set_style("whitegrid")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Universe
# -------------------------------------------------------------------------
cik_lookup = {
    "AMZN": "0001018724",
    "BMY": "0000014272",
    "CNP": "0001130310",
    "CVX": "0000093410",
    "FRT": "0000034903",
    "HON": "0000773840",
}
tickers = list(cik_lookup.keys())
example_ticker = "AMZN"
print(f"Universe: {tickers}")

# -------------------------------------------------------------------------
# 2. Download 10-K filings from SEC EDGAR
# -------------------------------------------------------------------------
print("\n--- Downloading 10-K filings from SEC EDGAR ---")
sec_api = SecAPI()
raw_filings = download_filings(sec_api, cik_lookup)
ten_ks = get_ten_k_filings(raw_filings, cik_lookup)

print("\nFilings per ticker:")
for ticker, filings in ten_ks.items():
    print(f"  {ticker}: {len(filings)} 10-K filings")

# -------------------------------------------------------------------------
# 3. Preprocess text
# -------------------------------------------------------------------------
print("\n--- Preprocessing filings ---")
ten_ks = preprocess_filings(ten_ks)

# -------------------------------------------------------------------------
# 4. Load sentiment dictionary
# -------------------------------------------------------------------------
LM_DICT_URL = "https://drive.google.com/uc?id=1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY&export=download"
LM_DICT_PATH = "data/loughran_mcdonald_master_dic_2016.csv"

if os.path.exists(LM_DICT_PATH) and os.path.getsize(LM_DICT_PATH) < 1024:
    os.remove(LM_DICT_PATH)

if not os.path.exists(LM_DICT_PATH):
    import requests
    print("\nDownloading Loughran-McDonald dictionary...")
    r = requests.get(LM_DICT_URL)
    r.raise_for_status()
    os.makedirs("data", exist_ok=True)
    with open(LM_DICT_PATH, "wb") as f:
        f.write(r.content)

sentiment_df = load_loughran_mcdonald(LM_DICT_PATH)
print(f"\nSentiment dictionary: {len(sentiment_df)} words")
print(f"Categories: {SENTIMENT_CATEGORIES}")

# -------------------------------------------------------------------------
# 5. Compute sentiment matrices
# -------------------------------------------------------------------------
print("\n--- Computing sentiment BoW and TF-IDF matrices ---")
bow_matrices, tfidf_matrices = compute_sentiment_matrices(
    ten_ks, sentiment_df, method="both"
)

# -------------------------------------------------------------------------
# 6. Year-over-year similarity
# -------------------------------------------------------------------------
print("\n--- Computing year-over-year similarities ---")
jaccard_similarities = compute_all_similarities(bow_matrices, method="jaccard")
cosine_similarities = compute_all_similarities(tfidf_matrices, method="cosine")

file_dates = {
    ticker: [f["file_date"] for f in filings] for ticker, filings in ten_ks.items()
}

# Save similarity tables
print("\nSample cosine similarities (AMZN):")
for sent in SENTIMENT_CATEGORIES:
    vals = cosine_similarities[example_ticker][sent]
    dates = file_dates[example_ticker][1:]
    print(f"  {sent}: {[f'{d}: {v:.4f}' for d, v in zip(dates[-3:], vals[-3:])]}")

# Plot 1: Jaccard similarity
plot_similarities(
    [jaccard_similarities[example_ticker][s] for s in SENTIMENT_CATEGORIES],
    file_dates[example_ticker][1:],
    f"Jaccard Similarity: {example_ticker}",
    SENTIMENT_CATEGORIES,
    save_path=os.path.join(OUTPUT_DIR, "01_jaccard_similarity.png"),
)
plt.close("all")
print(f"Saved: {OUTPUT_DIR}/01_jaccard_similarity.png")

# Plot 2: Cosine similarity
plot_similarities(
    [cosine_similarities[example_ticker][s] for s in SENTIMENT_CATEGORIES],
    file_dates[example_ticker][1:],
    f"Cosine Similarity: {example_ticker}",
    SENTIMENT_CATEGORIES,
    save_path=os.path.join(OUTPUT_DIR, "02_cosine_similarity.png"),
)
plt.close("all")
print(f"Saved: {OUTPUT_DIR}/02_cosine_similarity.png")

# -------------------------------------------------------------------------
# 7. Download pricing data
# -------------------------------------------------------------------------
print("\n--- Downloading pricing data ---")
data = yf.download(tickers, start="1993-01-01", end="2019-01-01", interval="1mo", progress=False)

if "Adj Close" in data.columns.get_level_values(0):
    pricing = data["Adj Close"]
else:
    pricing = data["Close"]

pricing = pricing.resample("YE").last()
pricing.index = pricing.index.to_period("Y").to_timestamp()
pricing = pricing.dropna(axis=1, how="all")

print(f"Pricing shape: {pricing.shape}")
print(f"Tickers with data: {pricing.columns.tolist()}")

# -------------------------------------------------------------------------
# 8. Alpha factor evaluation
# -------------------------------------------------------------------------
print("\n--- Evaluating alpha factors ---")
factor_df = build_factor_dataframe(cosine_similarities, file_dates)

# Factor returns
ls_returns = {}
for sentiment in SENTIMENT_CATEGORIES:
    ls_returns[sentiment] = compute_factor_returns(factor_df, pricing, sentiment)

# Plot 3: Cumulative factor returns
plot_cumulative_factor_returns(
    ls_returns,
    save_path=os.path.join(OUTPUT_DIR, "03_cumulative_factor_returns.png"),
)
plt.close("all")
print(f"Saved: {OUTPUT_DIR}/03_cumulative_factor_returns.png")

# Plot 4: Quantile returns
quantile_returns = {}
for sentiment in SENTIMENT_CATEGORIES:
    quantile_returns[sentiment] = compute_quantile_returns(
        factor_df, pricing, sentiment
    )

plot_quantile_returns(
    quantile_returns,
    SENTIMENT_CATEGORIES,
    save_path=os.path.join(OUTPUT_DIR, "04_quantile_returns.png"),
)
plt.close("all")
print(f"Saved: {OUTPUT_DIR}/04_quantile_returns.png")

# Plot 5: Factor rank autocorrelation
fra_dict = {}
for sentiment in SENTIMENT_CATEGORIES:
    fra_dict[sentiment] = compute_factor_rank_autocorrelation(factor_df, sentiment)

plot_factor_rank_autocorrelation(
    fra_dict,
    save_path=os.path.join(OUTPUT_DIR, "05_factor_rank_autocorrelation.png"),
)
plt.close("all")
print(f"Saved: {OUTPUT_DIR}/05_factor_rank_autocorrelation.png")

# -------------------------------------------------------------------------
# 9. Sharpe ratios
# -------------------------------------------------------------------------
sharpe_ratios = {
    sentiment: compute_sharpe_ratio(returns, annualization_factor=1.0)
    for sentiment, returns in ls_returns.items()
}

sharpe_df = pd.DataFrame.from_dict(
    sharpe_ratios, orient="index", columns=["Sharpe Ratio"]
)
sharpe_df = sharpe_df.sort_values("Sharpe Ratio", ascending=False)

print("\n" + "=" * 50)
print("Annualized Sharpe Ratios by Sentiment Factor:")
print("=" * 50)
print(sharpe_df.round(2))

# Save Sharpe table as CSV
sharpe_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "sharpe_ratios.csv"))
print(f"\nSaved: {OUTPUT_DIR}/sharpe_ratios.csv")

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("Pipeline complete. Output files:")
print("=" * 50)
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f:45s} {size_kb:6.1f} KB")
