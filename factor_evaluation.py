"""
Alpha factor evaluation module.

Custom implementation of factor return analysis, quantile decomposition,
factor rank autocorrelation, and Sharpe ratio computation.
Replaces Alphalens with purpose-built evaluation functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def build_factor_dataframe(cosine_similarities, file_dates):
    """
    Convert similarity dictionaries into a tidy DataFrame of factor values.

    Parameters
    ----------
    cosine_similarities : dict
        {ticker: {sentiment: list of float}}.
    file_dates : dict
        {ticker: list of str} filing dates per ticker.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, sentiment, value.
    """
    records = []
    for ticker, sentiments in cosine_similarities.items():
        dates = file_dates[ticker][1:]  # similarities are between consecutive pairs
        for sentiment, values in sentiments.items():
            for date_str, value in zip(dates, values):
                records.append({
                    "date": date_str,
                    "ticker": ticker,
                    "sentiment": sentiment,
                    "value": value,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.year
    df["date"] = pd.to_datetime(df["date"], format="%Y")
    return df


def compute_factor_returns(factor_df, pricing, sentiment, n_quantiles=5):
    """
    Compute long-short factor returns for a given sentiment factor.

    Assigns stocks to quantiles based on factor value each period, then
    computes the return spread between the top and bottom quantiles.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Tidy factor DataFrame with columns: date, ticker, sentiment, value.
    pricing : pd.DataFrame
        Yearly pricing with DatetimeIndex and ticker columns.
    sentiment : str
        Sentiment category to evaluate.
    n_quantiles : int
        Number of quantile buckets.

    Returns
    -------
    pd.Series
        Long-short factor returns indexed by date.
    """
    cs = factor_df[factor_df["sentiment"] == sentiment].copy()
    cs = cs.pivot(index="date", columns="ticker", values="value")

    # Compute forward returns from pricing
    fwd_returns = pricing.pct_change().shift(-1).dropna()

    # Align dates
    common_dates = cs.index.intersection(fwd_returns.index)
    cs = cs.loc[common_dates]
    fwd_returns = fwd_returns.loc[common_dates]

    factor_returns = []
    for date in common_dates:
        factor_vals = cs.loc[date].dropna()
        ret_vals = fwd_returns.loc[date]

        # Only use tickers present in both
        common_tickers = factor_vals.index.intersection(ret_vals.dropna().index)
        if len(common_tickers) < n_quantiles:
            continue

        fv = factor_vals[common_tickers]
        rv = ret_vals[common_tickers]

        # Assign quantiles
        quantile_labels = pd.qcut(fv, n_quantiles, labels=False, duplicates="drop")

        top_q = quantile_labels.max()
        bottom_q = quantile_labels.min()

        long_ret = rv[quantile_labels == top_q].mean()
        short_ret = rv[quantile_labels == bottom_q].mean()
        factor_returns.append({"date": date, "return": long_ret - short_ret})

    result = pd.DataFrame(factor_returns)
    if result.empty:
        return pd.Series(dtype=float)
    return result.set_index("date")["return"]


def compute_quantile_returns(factor_df, pricing, sentiment, n_quantiles=5):
    """
    Compute mean forward return by quantile bucket for a sentiment factor.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Tidy factor DataFrame.
    pricing : pd.DataFrame
        Yearly pricing DataFrame.
    sentiment : str
        Sentiment category.
    n_quantiles : int
        Number of quantile buckets.

    Returns
    -------
    pd.Series
        Mean return per quantile (1 = lowest factor, n = highest).
    """
    cs = factor_df[factor_df["sentiment"] == sentiment].copy()
    cs = cs.pivot(index="date", columns="ticker", values="value")

    fwd_returns = pricing.pct_change().shift(-1).dropna()
    common_dates = cs.index.intersection(fwd_returns.index)

    all_quantile_returns = {q: [] for q in range(1, n_quantiles + 1)}

    for date in common_dates:
        factor_vals = cs.loc[date].dropna()
        ret_vals = fwd_returns.loc[date]
        common_tickers = factor_vals.index.intersection(ret_vals.dropna().index)
        if len(common_tickers) < n_quantiles:
            continue

        fv = factor_vals[common_tickers]
        rv = ret_vals[common_tickers]

        try:
            quantile_labels = pd.qcut(fv, n_quantiles, labels=range(1, n_quantiles + 1), duplicates="drop")
        except ValueError:
            continue

        for q in range(1, n_quantiles + 1):
            mask = quantile_labels == q
            if mask.any():
                all_quantile_returns[q].append(rv[mask].mean())

    return pd.Series({q: np.mean(rets) if rets else np.nan for q, rets in all_quantile_returns.items()})


def compute_factor_rank_autocorrelation(factor_df, sentiment):
    """
    Compute factor rank autocorrelation over time.

    Measures how stable the cross-sectional ranking of stocks is from
    one period to the next. Higher FRA means more stable (lower turnover).

    Parameters
    ----------
    factor_df : pd.DataFrame
        Tidy factor DataFrame.
    sentiment : str
        Sentiment category.

    Returns
    -------
    pd.Series
        Rank autocorrelation indexed by date.
    """
    cs = factor_df[factor_df["sentiment"] == sentiment].copy()
    cs = cs.pivot(index="date", columns="ticker", values="value")

    fra_values = []
    dates = cs.index.sort_values()

    for i in range(1, len(dates)):
        prev = cs.loc[dates[i - 1]].dropna()
        curr = cs.loc[dates[i]].dropna()
        common = prev.index.intersection(curr.index)

        if len(common) < 3:
            continue

        rank_corr = prev[common].rank().corr(curr[common].rank())
        fra_values.append({"date": dates[i], "fra": rank_corr})

    result = pd.DataFrame(fra_values)
    if result.empty:
        return pd.Series(dtype=float)
    return result.set_index("date")["fra"]


def compute_sharpe_ratio(factor_returns, annualization_factor=None):
    """
    Compute the annualized Sharpe ratio of a factor return series.

    Parameters
    ----------
    factor_returns : pd.Series
        Period returns.
    annualization_factor : float, optional
        Scaling factor. Default is sqrt(252) for daily, but for yearly
        data use sqrt(1) = 1.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if factor_returns.empty or factor_returns.std() == 0:
        return 0.0
    if annualization_factor is None:
        annualization_factor = np.sqrt(252)
    return float(annualization_factor * factor_returns.mean() / factor_returns.std())


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_similarities(similarities_list, dates, title, labels, save_path=None):
    """Plot similarity time series for multiple sentiment categories."""
    plt.figure(figsize=(10, 7))
    for similarities, label in zip(similarities_list, labels):
        plt.plot(dates, similarities, label=label, marker="o", markersize=3)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=90)
    plt.ylabel("Similarity")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cumulative_factor_returns(factor_returns_dict, save_path=None):
    """
    Plot cumulative factor returns for each sentiment.

    Parameters
    ----------
    factor_returns_dict : dict
        {sentiment: pd.Series of returns}.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 7))
    for sentiment, returns in factor_returns_dict.items():
        if not returns.empty:
            cumulative = (1 + returns).cumprod()
            plt.plot(cumulative.index, cumulative.values, label=sentiment, marker="o", markersize=3)
    plt.title("Cumulative Factor Returns by Sentiment")
    plt.legend()
    plt.xticks(rotation=90)
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_quantile_returns(quantile_returns_dict, sentiments, save_path=None):
    """
    Plot bar charts of mean return by quantile for each sentiment.

    Parameters
    ----------
    quantile_returns_dict : dict
        {sentiment: pd.Series of quantile returns}.
    sentiments : list of str
        Sentiment categories to plot.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    n = len(sentiments)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharey=True)
    axes = axes.flatten() if n > 1 else [axes]

    for idx, sentiment in enumerate(sentiments):
        ax = axes[idx]
        qr = quantile_returns_dict.get(sentiment, pd.Series())
        if not qr.empty:
            (qr * 10000).plot.bar(ax=ax, color=sns.color_palette("coolwarm", len(qr)))
        ax.set_title(sentiment.capitalize())
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Mean Return (bps)")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Basis Points per Period by Quantile", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_factor_rank_autocorrelation(fra_dict, save_path=None):
    """
    Plot factor rank autocorrelation over time for each sentiment.

    Parameters
    ----------
    fra_dict : dict
        {sentiment: pd.Series of FRA values}.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 7))
    for sentiment, fra in fra_dict.items():
        if not fra.empty:
            plt.plot(fra.index, fra.values, label=sentiment, marker="o", markersize=3)
    plt.title("Factor Rank Autocorrelation")
    plt.legend()
    plt.xticks(rotation=90)
    plt.ylabel("Rank Autocorrelation")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
