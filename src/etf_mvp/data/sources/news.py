"""News with sentiment from the Alpha Vantage NEWS_SENTIMENT endpoint."""

from __future__ import annotations

import hashlib
import os
import re
from io import StringIO

import numpy as np
import pandas as pd

from etf_mvp.data.base import DataSource, FetchRequest
from etf_mvp.data.calendar import available_after_close
from etf_mvp.data.http import HttpClient
from etf_mvp.data.kinds import DataKind

NEWS_COLUMNS = [
    "article_id",
    "published_at",
    "available_date",
    "title",
    "source_name",
    "url",
    "topics",
    "tickers",
    "sentiment",
    "source",
]


def topic_slug(name: str) -> str:
    """'Economy - Monetary' -> 'economy_monetary' (the API's query-parameter form)."""
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def article_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16]


def parse_news_feed(payload: dict) -> pd.DataFrame:
    for key in ("Information", "Note", "Error Message"):
        if key in payload and "feed" not in payload:
            raise RuntimeError(f"Alpha Vantage: {payload[key]}")
    feed = payload.get("feed", [])
    if not feed:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    rows = pd.DataFrame(
        {
            "title": [item.get("title", "") for item in feed],
            "url": [item.get("url", "") for item in feed],
            "source_name": [item.get("source", "") for item in feed],
            "published_at": pd.to_datetime(
                [item["time_published"] for item in feed], format="%Y%m%dT%H%M%S"
            ),
            "topics": [
                ",".join(sorted(topic_slug(t["topic"]) for t in item.get("topics", [])))
                for item in feed
            ],
            "tickers": [
                ",".join(sorted(t["ticker"] for t in item.get("ticker_sentiment", [])))
                for item in feed
            ],
            "sentiment": [float(item.get("overall_sentiment_score", np.nan)) for item in feed],
        }
    )
    # time_published is US Eastern; after-close stories count from the next session.
    rows["available_date"] = available_after_close(rows["published_at"])
    rows["article_id"] = [article_id(u, t) for u, t in zip(rows["url"], rows["title"])]
    rows["source"] = "alphavantage"
    return rows[NEWS_COLUMNS]


class AlphaVantageNewsSource(DataSource):
    kind = DataKind.NEWS
    name = "alphavantage"
    url = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None, client: HttpClient | None = None) -> None:
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required for news provider 'alphavantage'")
        self.client = client or HttpClient(min_interval_seconds=1.2)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        topics: list[str] = request.options.get("topics", [])
        window = pd.Timedelta(days=int(request.options.get("window_days", 30)))
        budget = int(request.options.get("max_requests", 20))
        start = max(pd.Timestamp(request.start), pd.Timestamp("2022-03-01"))
        end = pd.Timestamp(request.end) + pd.Timedelta(days=1)

        frames: list[pd.DataFrame] = []
        calls = 0
        for topic in topics:
            window_start = start
            while window_start < end:
                if calls >= budget:
                    # Free tier allows ~25 calls a day; the rest is picked up next run.
                    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                window_end = min(window_start + window, end)
                payload = self.client.get(
                    self.url,
                    params={
                        "function": "NEWS_SENTIMENT",
                        "topics": topic,
                        "time_from": window_start.strftime("%Y%m%dT%H%M"),
                        "time_to": window_end.strftime("%Y%m%dT%H%M"),
                        "sort": "EARLIEST",
                        "limit": 1000,
                        "apikey": self.api_key,
                    },
                ).json()
                frames.append(parse_news_feed(payload))
                calls += 1
                window_start = window_end
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_sf_fed_sentiment(text: str, start: str, end: str) -> pd.DataFrame:
    raw = pd.read_csv(StringIO(text))
    observation_date = pd.to_datetime(raw["date"], format="%m/%d/%Y")
    sentiment = pd.to_numeric(raw["Daily news sentiment"], errors="coerce")
    frame = pd.DataFrame(
        {
            "observation_date": observation_date,
            "sentiment": sentiment,
        }
    ).dropna(subset=["sentiment"])
    frame = frame.loc[frame["observation_date"].dt.dayofweek < 5].copy()
    mask = frame["observation_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    frame = frame.loc[mask].copy()
    frame["published_at"] = frame["observation_date"] + pd.Timedelta(hours=16)
    frame["available_date"] = frame["observation_date"] + pd.Timedelta(days=7)
    frame["article_id"] = frame["observation_date"].dt.strftime("sf-fed-news-%Y%m%d")
    frame["title"] = "SF Fed Daily News Sentiment Index"
    frame["source_name"] = "Federal Reserve Bank of San Francisco"
    frame["url"] = (
        "https://www.frbsf.org/research-and-insights/data-and-indicators/"
        "daily-news-sentiment-index/"
    )
    frame["topics"] = "economy_macro,financial_markets"
    frame["tickers"] = ""
    frame["source"] = "sf_fed"
    return frame[NEWS_COLUMNS]


class SfFedNewsSource(DataSource):
    kind = DataKind.NEWS
    name = "sf_fed"
    url = "https://www.frbsf.org/wp-content/uploads/news-sentiment-chart-1.csv"

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient(min_interval_seconds=0.5)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        response = self.client.get(self.url)
        return parse_sf_fed_sentiment(response.text, request.start, request.end)


class SyntheticNewsSource(DataSource):
    kind = DataKind.NEWS
    name = "synthetic"
    requires_network = False

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        topics: list[str] = request.options.get("topics", []) or ["financial_markets"]
        rows: list[dict] = []
        for day in pd.bdate_range(request.start, request.end):
            # Seeded per day so overlapping incremental fetches return identical articles.
            random = np.random.default_rng([self.random_seed, day.toordinal()])
            day_mood = 0.3 * np.sin(day.toordinal() / 45)
            for number in range(int(random.poisson(6))):
                published = day + pd.Timedelta(minutes=int(random.integers(6 * 60, 22 * 60)))
                topic = topics[int(random.integers(len(topics)))]
                title = f"Synthetic {topic} story {day.date()} #{number}"
                url = f"https://example.invalid/{day.date()}/{number}"
                rows.append(
                    {
                        "article_id": article_id(url, title),
                        "published_at": published,
                        "title": title,
                        "source_name": "synthetic",
                        "url": url,
                        "topics": topic,
                        "tickers": "",
                        "sentiment": float(np.tanh(day_mood + random.normal(0, 0.25))),
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=NEWS_COLUMNS)
        frame["available_date"] = available_after_close(frame["published_at"])
        frame["source"] = self.name
        return frame[NEWS_COLUMNS]
