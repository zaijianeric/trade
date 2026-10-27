"""Point-in-time news from SF Fed, GDELT and Alpha Vantage."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import xml.etree.ElementTree as ET
from io import StringIO

import numpy as np
import pandas as pd
from requests import RequestException

from etf_mvp.data.base import DataSource, FetchRequest
from etf_mvp.data.calendar import available_after_close
from etf_mvp.data.http import HttpClient
from etf_mvp.data.kinds import DataKind, normalize

logger = logging.getLogger(__name__)

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


POSITIVE_WORDS = {
    "advance",
    "beat",
    "boom",
    "bullish",
    "gain",
    "growth",
    "optimism",
    "rally",
    "record",
    "rebound",
    "recovery",
    "strong",
    "surge",
    "upgrade",
}
NEGATIVE_WORDS = {
    "bearish",
    "conflict",
    "crash",
    "crisis",
    "cut",
    "decline",
    "downgrade",
    "fear",
    "inflation",
    "loss",
    "recession",
    "risk",
    "sanction",
    "slump",
    "tariff",
    "war",
    "weak",
}


def title_sentiment(title: str) -> float:
    words = re.findall(r"[a-z]+", title.lower())
    positive = sum(word in POSITIVE_WORDS for word in words)
    negative = sum(word in NEGATIVE_WORDS for word in words)
    return float(np.tanh((positive - negative) / 2))


def deduplicate_news(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    result = frame.copy()
    normalized_title = result["title"].fillna("").str.lower().str.replace(
        r"[^a-z0-9]+", " ", regex=True
    ).str.strip()
    result["_title_key"] = normalized_title.map(
        lambda value: hashlib.sha1(value.encode()).hexdigest()[:16]
    )
    published = pd.to_datetime(result["published_at"], errors="coerce", utc=True)
    result["_published_day"] = published.dt.tz_convert(None).dt.normalize()
    priority = result["source"].map(
        {"alphavantage": 4, "gdelt": 3, "google_news": 2, "sf_fed": 1}
    ).fillna(0)
    result["_priority"] = priority
    result = result.sort_values(["_priority", "published_at"]).drop_duplicates(
        ["_title_key", "_published_day"], keep="last"
    )
    return result[NEWS_COLUMNS].sort_values("published_at").reset_index(drop=True)


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


def parse_gdelt_articles(
    payload: dict, topic: str, tickers: tuple[str, ...] = ()
) -> pd.DataFrame:
    articles = payload.get("articles", [])
    if not articles:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    frame = pd.DataFrame(articles)
    published = (
        pd.to_datetime(frame.get("seendate"), errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    titles = frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
    urls = frame.get("url", pd.Series("", index=frame.index)).fillna("").astype(str)
    result = pd.DataFrame(
        {
            "published_at": published,
            "title": titles,
            "source_name": frame.get("domain", pd.Series("GDELT", index=frame.index)),
            "url": urls,
            "topics": topic,
            "tickers": ",".join(tickers),
            "sentiment": titles.map(title_sentiment),
        }
    ).dropna(subset=["published_at"])
    result["available_date"] = available_after_close(result["published_at"])
    result["article_id"] = [
        article_id(url, title)
        for url, title in zip(result["url"], result["title"], strict=True)
    ]
    result["source"] = "gdelt"
    return result[NEWS_COLUMNS]


class GdeltNewsSource(DataSource):
    kind = DataKind.NEWS
    name = "gdelt"
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    DEFAULT_QUERIES = {
        "financial_markets": '("stock market" OR "wall street" OR "S&P 500" OR Nasdaq)',
        "economy_macro": '(inflation OR recession OR "Federal Reserve" OR "interest rates")',
        "technology": '(semiconductor OR chipmaker OR "artificial intelligence")',
        "geopolitical": '(war OR sanctions OR tariff OR geopolitical)',
    }
    DEFAULT_TICKERS = {
        "financial_markets": ("SPY", "QQQ", "IWM", "SOXX"),
        "economy_macro": ("SPY", "QQQ", "IWM", "SOXX"),
        "technology": ("QQQ", "SOXX"),
        "geopolitical": ("SPY", "QQQ", "IWM", "SOXX"),
    }

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient(min_interval_seconds=1.0)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        queries = request.options.get("gdelt_queries", self.DEFAULT_QUERIES)
        max_records = int(request.options.get("gdelt_max_records", 250))
        timespan = str(request.options.get("gdelt_timespan", "3months"))
        frames: list[pd.DataFrame] = []
        for topic, query in queries.items():
            payload = self.client.get(
                self.url,
                params={
                    "query": f"{query} sourcelang:english",
                    "mode": "artlist",
                    "format": "json",
                    "maxrecords": min(max_records, 250),
                    "timespan": timespan,
                    "sort": "datedesc",
                },
            ).json()
            tickers = tuple(self.DEFAULT_TICKERS.get(topic, ()))
            frames.append(parse_gdelt_articles(payload, topic, tickers))
        return deduplicate_news(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def parse_google_news_rss(
    text: str, topic: str, tickers: tuple[str, ...] = ()
) -> pd.DataFrame:
    root = ET.fromstring(text)
    rows: list[dict[str, object]] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title", default="").strip()
        url = item.findtext("link", default="").strip()
        published_text = item.findtext("pubDate", default="").strip()
        if not title or not published_text:
            continue
        rows.append(
            {
                "title": title,
                "url": url,
                "source_name": item.findtext("source", default="Google News").strip(),
                "published_at": published_text,
            }
        )
    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    result = pd.DataFrame(rows)
    result["published_at"] = (
        pd.to_datetime(result["published_at"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    result = result.dropna(subset=["published_at"])
    result["available_date"] = available_after_close(result["published_at"])
    result["topics"] = topic
    result["tickers"] = ",".join(tickers)
    result["sentiment"] = result["title"].map(title_sentiment)
    result["article_id"] = [
        article_id(url, title)
        for url, title in zip(result["url"], result["title"], strict=True)
    ]
    result["source"] = "google_news"
    return result[NEWS_COLUMNS]


class GoogleNewsRssSource(DataSource):
    kind = DataKind.NEWS
    name = "google_news"
    url = "https://news.google.com/rss/search"

    DEFAULT_QUERIES = {
        "financial_markets": '("stock market" OR Nasdaq OR "S&P 500")',
        "economy_macro": '(inflation OR recession OR "Federal Reserve")',
        "technology": '(semiconductor OR chipmaker OR "artificial intelligence")',
        "geopolitical": '(war OR sanctions OR tariff OR geopolitical)',
    }

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient(min_interval_seconds=1.5)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        queries = request.options.get("google_news_queries", self.DEFAULT_QUERIES)
        lookback_days = min(int(request.options.get("google_news_days", 30)), 90)
        frames: list[pd.DataFrame] = []
        for topic, query in queries.items():
            response = self.client.get(
                self.url,
                params={
                    "q": f"{query} when:{lookback_days}d",
                    "hl": "en-US",
                    "gl": "US",
                    "ceid": "US:en",
                },
            )
            tickers = tuple(GdeltNewsSource.DEFAULT_TICKERS.get(topic, ()))
            frames.append(parse_google_news_rss(response.text, topic, tickers))
        return deduplicate_news(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


class HybridNewsSource(DataSource):
    kind = DataKind.NEWS
    name = "hybrid"

    def __init__(self) -> None:
        self.sf_fed = SfFedNewsSource()
        self.gdelt = GdeltNewsSource()
        self.google_news = GoogleNewsRssSource()

    def fetch(self, request: FetchRequest) -> pd.DataFrame:
        return normalize(self.kind, self._fetch(request))

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        frames = [self.sf_fed.fetch(request)]
        article_frame = pd.DataFrame(columns=NEWS_COLUMNS)
        try:
            article_frame = self.gdelt.fetch(request)
        except (RequestException, RuntimeError, ValueError) as error:
            logger.warning("GDELT news fetch failed; using Google News RSS: %s", error)
        if article_frame.empty:
            try:
                article_frame = self.google_news.fetch(request)
            except (ET.ParseError, RequestException, RuntimeError, ValueError) as error:
                logger.warning("Google News RSS fetch failed; keeping SF Fed data: %s", error)
        frames.append(article_frame)
        if os.getenv("ALPHAVANTAGE_API_KEY"):
            try:
                frames.append(AlphaVantageNewsSource().fetch(request))
            except (RequestException, RuntimeError, ValueError) as error:
                logger.warning("Alpha Vantage news fetch failed: %s", error)
        return deduplicate_news(pd.concat(frames, ignore_index=True))


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
