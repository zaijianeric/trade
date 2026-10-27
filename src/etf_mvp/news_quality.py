from __future__ import annotations

from collections import Counter

import pandas as pd


def summarize_news_quality(
    news: pd.DataFrame, market_prices: pd.DataFrame
) -> dict[str, object]:
    if news.empty:
        return {
            "status": "missing",
            "rows": 0,
            "sources": [],
        }
    frame = news.copy()
    frame["available_date"] = pd.to_datetime(frame["available_date"])
    frame["published_at"] = pd.to_datetime(frame["published_at"])
    latest_market = pd.to_datetime(market_prices["date"]).max()
    latest_news = frame["available_date"].max()
    topics = Counter()
    for value in frame["topics"].fillna(""):
        topics.update(topic for topic in str(value).split(",") if topic)
    source_rows = []
    for source, scoped in frame.groupby("source", dropna=False):
        source_rows.append(
            {
                "source": str(source),
                "rows": len(scoped),
                "coverage_start": scoped["available_date"].min().date().isoformat(),
                "latest_available": scoped["available_date"].max().date().isoformat(),
                "active_days": int(scoped["available_date"].nunique()),
            }
        )
    article_rows = frame.loc[frame["source"] != "sf_fed"]
    return {
        "status": "available",
        "rows": len(frame),
        "article_rows": len(article_rows),
        "coverage_start": frame["available_date"].min().date().isoformat(),
        "latest_available": latest_news.date().isoformat(),
        "latest_market": latest_market.date().isoformat(),
        "stale_calendar_days": max(0, int((latest_market - latest_news).days)),
        "unique_articles": int(frame["article_id"].nunique()),
        "active_days": int(frame["available_date"].nunique()),
        "sources": source_rows,
        "top_topics": [
            {"topic": topic, "rows": rows} for topic, rows in topics.most_common(10)
        ],
    }
