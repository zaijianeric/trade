from __future__ import annotations

import numpy as np
import pandas as pd

from etf_mvp.features.asof import rolling_zscore, to_trading_dates


def _scope_features(articles: pd.DataFrame, dates: pd.DatetimeIndex, prefix: str) -> dict:
    daily = articles.groupby("trading_date").agg(
        count=("article_id", "size"), sentiment_sum=("sentiment", "sum")
    )
    daily = daily.reindex(dates)
    first = articles["trading_date"].min()
    covered = dates >= first  # before coverage starts, "no news" is unknown, not zero
    count = daily["count"].fillna(0).where(covered)
    sentiment_sum = daily["sentiment_sum"].fillna(0).where(covered)

    columns: dict[str, pd.Series] = {}
    for window in (5, 20):
        total = count.rolling(window, min_periods=1).sum()
        columns[f"{prefix}__sentiment_{window}d"] = sentiment_sum.rolling(
            window, min_periods=1
        ).sum() / total.where(total > 0)
    columns[f"{prefix}__volume_z_60d"] = rolling_zscore(count.rolling(5, min_periods=1).sum(), 60)
    return columns


def build_news_features(
    news: pd.DataFrame, dates: pd.DatetimeIndex, topics: list[str]
) -> pd.DataFrame:
    """Rolling news sentiment and attention, overall and per configured topic."""
    if news.empty:
        return pd.DataFrame(index=dates)
    articles = news.dropna(subset=["sentiment"]).copy()
    articles["trading_date"] = to_trading_dates(articles["available_date"], dates)
    articles = articles.dropna(subset=["trading_date"])
    if articles.empty:
        return pd.DataFrame(index=dates)

    columns: dict[str, pd.Series] = {}
    sources = (
        articles["source"]
        if "source" in articles
        else pd.Series("", index=articles.index, dtype="object")
    )
    sf_fed = articles.loc[sources == "sf_fed"]
    if not sf_fed.empty:
        index_level = (
            sf_fed.groupby("trading_date")["sentiment"]
            .last()
            .reindex(dates)
            .ffill()
        )
        columns["news__economic_sentiment__level"] = index_level
        columns["news__economic_sentiment__chg_5d"] = index_level.diff(5)
        columns["news__economic_sentiment__z_60d"] = rolling_zscore(index_level, 60)
        columns["news__economic_sentiment__z_252d"] = rolling_zscore(index_level, 252)

    article_news = articles.loc[sources != "sf_fed"]
    if article_news.empty:
        return pd.DataFrame(columns, index=dates).replace([np.inf, -np.inf], np.nan)

    columns |= _scope_features(article_news, dates, "news__all")
    article_topics = article_news["topics"].fillna("").str.split(",")
    for topic in topics:
        mask = article_topics.map(lambda values, t=topic: t in values).astype(bool)
        if mask.any():
            columns |= _scope_features(article_news.loc[mask], dates, f"news__{topic}")
    return pd.DataFrame(columns, index=dates).replace([np.inf, -np.inf], np.nan)
