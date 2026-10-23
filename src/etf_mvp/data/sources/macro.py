"""Macro data from FRED, with ALFRED first-release vintages for revised series."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from etf_mvp.data.base import DataSource, FetchRequest
from etf_mvp.data.http import HttpClient
from etf_mvp.data.kinds import DataKind


@dataclass(frozen=True)
class MacroSeries:
    """One FRED series.

    ``vintage``: the series is revised after release (CPI, payrolls, claims), so use
    ALFRED and keep the first published value with its real release date.
    ``lag_days``: business days from observation to availability for non-vintage series.
    """

    series_id: str
    vintage: bool = False
    lag_days: int = 1

    @classmethod
    def from_config(cls, raw: dict | str) -> MacroSeries:
        if isinstance(raw, str):
            return cls(series_id=raw)
        return cls(
            series_id=raw["id"],
            vintage=bool(raw.get("vintage", False)),
            lag_days=int(raw.get("lag_days", 1)),
        )


def series_from_options(request: FetchRequest) -> list[MacroSeries]:
    return [MacroSeries.from_config(item) for item in request.options.get("series", [])]


class FredMacroSource(DataSource):
    kind = DataKind.MACRO
    name = "fred"
    url = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None, client: HttpClient | None = None) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY is required for macro provider 'fred'")
        # FRED's firewall stalls requests with unfamiliar User-Agents; keep the default.
        self.client = client or HttpClient(user_agent=None, min_interval_seconds=0.6)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        frames = [self.fetch_series(series, request) for series in series_from_options(request)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_series(self, series: MacroSeries, request: FetchRequest) -> pd.DataFrame:
        params = {
            "series_id": series.series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": request.start,
            "observation_end": request.end,
        }
        if series.vintage:
            # Full vintage history, so first releases before `start` keep their true dates.
            params |= {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}
        payload = self.client.get(self.url, params=params).json()
        return parse_fred_observations(payload, series)


def parse_fred_observations(payload: dict, series: MacroSeries) -> pd.DataFrame:
    rows = pd.DataFrame(payload.get("observations", []))
    if rows.empty:
        return pd.DataFrame(
            columns=["series_id", "observation_date", "available_date", "value", "source"]
        )
    rows["value"] = pd.to_numeric(rows["value"].replace(".", np.nan), errors="coerce")
    rows["observation_date"] = pd.to_datetime(rows["date"])
    if series.vintage:
        # Keep the first vintage of each observation: what the market saw on release day.
        rows["realtime_start"] = pd.to_datetime(rows["realtime_start"])
        rows = rows.dropna(subset=["value"]).sort_values("realtime_start")
        rows = rows.drop_duplicates("observation_date", keep="first")
        rows["available_date"] = rows["realtime_start"]
    else:
        rows["available_date"] = rows["observation_date"] + BDay(series.lag_days)
    rows["series_id"] = series.series_id
    rows["source"] = "fred"
    return rows[["series_id", "observation_date", "available_date", "value", "source"]]


class SyntheticMacroSource(DataSource):
    """Random-walk macro series with realistic release lags, for offline runs."""

    kind = DataKind.MACRO
    name = "synthetic"
    requires_network = False

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        random = np.random.default_rng(self.random_seed + 101)
        frames: list[pd.DataFrame] = []
        for index, series in enumerate(series_from_options(request)):
            monthly = series.vintage
            freq = "MS" if monthly else "B"
            dates = pd.date_range(request.start, request.end, freq=freq)
            if dates.empty:
                continue
            scale = 0.05 if monthly else 0.02
            values = 2.0 + index * 0.5 + np.cumsum(random.normal(0, scale, len(dates)))
            lag = BDay(25) if monthly else BDay(series.lag_days)
            frames.append(
                pd.DataFrame(
                    {
                        "series_id": series.series_id,
                        "observation_date": dates,
                        "available_date": dates + lag,
                        "value": values,
                        "source": self.name,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
