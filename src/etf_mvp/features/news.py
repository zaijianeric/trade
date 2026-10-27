from __future__ import annotations

import numpy as np
import pandas as pd

from etf_mvp.features.asof import rolling_zscore, to_trading_dates


def _scope_features(articles: pd.DataFrame, dates: pd.DatetimeIndex, prefix: str) -> dict:
    working = articles.copy()
    if "source_name" not in working:
        working["source_name"] = working.get("source", "unknown")
    daily = working.groupby("trading_date").agg(
        count=("article_id", "size"),
        sentiment_sum=("sentiment", "sum"),
        sentiment_std=("sentiment", "std"),
        positive=("sentiment", lambda values: values.gt(0.15).sum()),
        negative=("sentiment", lambda values: values.lt(-0.15).sum()),
        source_count=("source_name", "nunique"),
    )
    daily = daily.reindex(dates)
    first = articles["trading_date"].min()
    covered = dates >= first  # before coverage starts, "no news" is unknown, not zero
    count = daily["count"].fillna(0).where(covered)
    sentiment_sum = daily["sentiment_sum"].fillna(0).where(covered)

    columns: dict[str, pd.Series] = {}
    for window in (1, 3, 5, 7, 20):
        total = count.rolling(window, min_periods=1).sum()
        columns[f"{prefix}__sentiment_{window}d"] = sentiment_sum.rolling(
            window, min_periods=1
        ).sum() / total.where(total > 0)
        if window in (5, 20):
            columns[f"{prefix}__attention_{window}d"] = total
    total_5d = count.rolling(5, min_periods=1).sum()
    columns[f"{prefix}__volume_z_60d"] = rolling_zscore(count.rolling(5, min_periods=1).sum(), 60)
    columns[f"{prefix}__dispersion_5d"] = daily["sentiment_std"].reindex(dates).rolling(
        5, min_periods=1
    ).mean().where(covered)
    columns[f"{prefix}__positive_share_5d"] = daily["positive"].reindex(dates).fillna(0).rolling(
        5, min_periods=1
    ).sum().div(total_5d.where(total_5d > 0)).where(covered)
    columns[f"{prefix}__negative_share_5d"] = daily["negative"].reindex(dates).fillna(0).rolling(
        5, min_periods=1
    ).sum().div(total_5d.where(total_5d > 0)).where(covered)
    columns[f"{prefix}__source_diversity_5d"] = daily["source_count"].reindex(
        dates
    ).fillna(0).rolling(5, min_periods=1).mean().where(covered)
    return columns


EVENT_PATTERNS = {
    "monetary_policy": r"fed|federal reserve|interest rate|rate cut|rate hike|powell",
    "inflation_growth": r"inflation|cpi|jobs|employment|gdp|recession|growth",
    "earnings": r"earnings|revenue|profit|guidance|quarterly results",
    "regulation_trade": r"regulation|antitrust|tariff|trade restriction|export control",
    "geopolitical": r"war|conflict|sanction|missile|invasion|geopolitical",
    "supply_chain": r"supply chain|shortage|shipping|factory|production halt",
}


def _event_features(
    articles: pd.DataFrame, dates: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    titles = articles.get("title", pd.Series("", index=articles.index)).fillna("")
    columns: dict[str, pd.Series] = {}
    for event_name, pattern in EVENT_PATTERNS.items():
        scoped = articles.loc[titles.str.contains(pattern, case=False, regex=True)]
        if scoped.empty:
            continue
        daily = scoped.groupby("trading_date").agg(
            count=("article_id", "size"), sentiment=("sentiment", "mean")
        ).reindex(dates)
        first = scoped["trading_date"].min()
        covered = dates >= first
        columns[f"news__event_{event_name}__count_5d"] = daily["count"].fillna(0).rolling(
            5, min_periods=1
        ).sum().where(covered)
        columns[f"news__event_{event_name}__sentiment_3d"] = daily[
            "sentiment"
        ].rolling(3, min_periods=1).mean().where(covered)
    return columns


def build_news_features(
    news: pd.DataFrame,
    dates: pd.DatetimeIndex,
    topics: list[str],
    symbols: list[str] | tuple[str, ...] = (),
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
    article_tickers = article_news.get(
        "tickers", pd.Series("", index=article_news.index)
    ).fillna("").str.split(",")
    for symbol in symbols:
        mask = article_tickers.map(lambda values, target=symbol: target in values).astype(bool)
        if mask.any():
            columns |= _scope_features(
                article_news.loc[mask], dates, f"news__ticker_{symbol.lower()}"
            )
    columns |= _event_features(article_news, dates)
    return pd.DataFrame(columns, index=dates).replace([np.inf, -np.inf], np.nan)
