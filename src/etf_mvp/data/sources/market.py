"""Market data: daily adjusted OHLCV per symbol."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from etf_mvp.data.base import DataSource, FetchRequest
from etf_mvp.data.http import HttpClient
from etf_mvp.data.kinds import SCHEMAS, DataKind

PRICE_COLUMNS = list(SCHEMAS[DataKind.MARKET].columns)
ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


class AlpacaMarketSource(DataSource):
    """Adjusted daily OHLCV bars for the configured ETF universe."""

    kind = DataKind.MARKET
    name = "alpaca"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY_ID", "").strip()
        self.secret_key = secret_key or os.getenv("ALPACA_API_SECRET_KEY", "").strip()
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required for "
                "market provider 'alpaca'"
            )
        self.client = HttpClient(min_interval_seconds=0.2, timeout_seconds=60)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        feed = str(request.options.get("feed", "iex"))
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            params: dict[str, object] = {
                "timeframe": "1Day",
                "start": request.start,
                "end": request.end,
                "adjustment": "all",
                "feed": feed,
                "limit": 10_000,
                "sort": "asc",
            }
            rows: list[dict[str, object]] = []
            while True:
                payload = self.client.get(
                    ALPACA_BARS_URL.format(symbol=symbol), headers=headers, params=params
                ).json()
                if payload.get("message") and not payload.get("bars"):
                    raise RuntimeError(
                        f"Alpaca market data error for {symbol}: {payload['message']}"
                    )
                rows.extend(payload.get("bars") or [])
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
                params["page_token"] = page_token
            if not rows:
                raise ValueError(f"Alpaca returned no daily bars for {symbol}")
            frames.append(parse_alpaca_daily_prices(rows, symbol))
        return pd.concat(frames, ignore_index=True)


def parse_alpaca_daily_prices(rows: list[dict[str, object]], symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["t"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize(),
            "symbol": symbol,
            "open": frame["o"],
            "high": frame["h"],
            "low": frame["l"],
            "close": frame["c"],
            "volume": frame["v"],
        }
    )
    return result[PRICE_COLUMNS]


class TiingoMarketSource(DataSource):
    kind = DataKind.MARKET
    name = "tiingo"
    base_url = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str | None = None, client: HttpClient | None = None) -> None:
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY is required for market provider 'tiingo'")
        self.client = client or HttpClient()

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        headers = {"Authorization": f"Token {self.api_key}"}
        for symbol in request.symbols:
            rows = self.client.get(
                f"{self.base_url}/{symbol}/prices",
                headers=headers,
                params={
                    "startDate": request.start,
                    "endDate": request.end,
                    "resampleFreq": "daily",
                },
            ).json()
            if not rows:
                raise ValueError(f"Tiingo returned no rows for {symbol}")
            frames.append(parse_tiingo_prices(rows, symbol))
        return pd.concat(frames, ignore_index=True)


def parse_tiingo_prices(rows: list[dict], symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame.assign(
        date=pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None).dt.normalize(),
        symbol=symbol,
        open=frame.get("adjOpen", frame["open"]),
        high=frame.get("adjHigh", frame["high"]),
        low=frame.get("adjLow", frame["low"]),
        close=frame.get("adjClose", frame["close"]),
        volume=frame.get("adjVolume", frame["volume"]),
    )
    return frame[PRICE_COLUMNS]


class NasdaqMarketSource(DataSource):
    kind = DataKind.MARKET
    name = "nasdaq"
    base_url = "https://api.nasdaq.com/api/quote"

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/141 Safari/537.36"
            ),
            min_interval_seconds=0.35,
        )

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nasdaq.com/market-activity/etf/",
        }
        for symbol in request.symbols:
            payload = self.client.get(
                f"{self.base_url}/{symbol}/historical",
                headers=headers,
                params={
                    "assetclass": "etf",
                    "fromdate": request.start,
                    "todate": request.end,
                    "limit": 5000,
                },
            ).json()
            data = payload.get("data") or {}
            rows = data.get("tradesTable", {}).get("rows") or []
            if not rows:
                status = payload.get("status") or {}
                raise ValueError(f"Nasdaq returned no rows for {symbol}: {status}")
            frames.append(parse_nasdaq_prices(rows, symbol))
        return pd.concat(frames, ignore_index=True)


def parse_nasdaq_prices(rows: list[dict], symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(
            frame[column].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], format="%m/%d/%Y"),
            "symbol": symbol,
            "open": numeric("open"),
            "high": numeric("high"),
            "low": numeric("low"),
            "close": numeric("close"),
            "volume": numeric("volume"),
        }
    )
    return parsed[PRICE_COLUMNS]


class SyntheticMarketSource(DataSource):
    """Deterministic correlated prices for development and CI only."""

    kind = DataKind.MARKET
    name = "synthetic"
    requires_network = False

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        dates = pd.bdate_range(start=request.start, end=request.end)
        if len(dates) < 260:
            raise ValueError("Synthetic dataset needs at least 260 business days")

        random = np.random.default_rng(self.random_seed)
        market_return = random.normal(0.0003, 0.009, size=len(dates))
        volatility_cycle = 1 + 0.35 * np.sin(np.arange(len(dates)) / 80)
        frames: list[pd.DataFrame] = []
        for index, symbol in enumerate(request.symbols):
            beta = 0.55 + 0.12 * (index % 5)
            drift = 0.00008 * ((index % 3) - 1)
            idiosyncratic = random.normal(0, 0.004 + index * 0.00035, size=len(dates))
            daily_return = (beta * market_return + idiosyncratic) * volatility_cycle + drift
            close = (80 + index * 12) * np.exp(np.cumsum(daily_return))
            open_price = close * np.exp(random.normal(0, 0.002, size=len(dates)))
            intraday_range = np.abs(random.normal(0.004, 0.002, size=len(dates)))
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "symbol": symbol,
                        "open": open_price,
                        "high": np.maximum(open_price, close) * (1 + intraday_range),
                        "low": np.minimum(open_price, close) * (1 - intraday_range),
                        "close": close,
                        "volume": random.lognormal(16.0 - index * 0.06, 0.25, len(dates)).round(),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)
