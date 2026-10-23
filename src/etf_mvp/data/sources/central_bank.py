"""FOMC meetings, policy-rate decisions and the market's read of each decision.

Meeting dates come from a bundled calendar (``reference/fomc_meetings.csv``), plus an
optional local CSV. They are announced a year ahead, so knowing the *next* meeting is
not look-ahead. Decisions come from FRED's target-range series (DFEDTARU/DFEDTARL),
which take effect the day after the meeting. Tone is the move of the 2-year Treasury
yield (DGS2) on the announcement day, in basis points: positive = hawkish.

DGS2 is published on FRED the next business day, so a decision row becomes available
the business day after the market reacted, the same lag the macro layer uses.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from etf_mvp.data.base import DataSource, FetchRequest
from etf_mvp.data.http import HttpClient
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.sources.macro import FredMacroSource, MacroSeries

BUILTIN_MEETINGS = resources.files("etf_mvp.data").joinpath("reference/fomc_meetings.csv")
# Weekend decisions (e.g. 2020-03-15) are priced on the next session.
MAX_REACTION_DAYS = 5
EVENT_COLUMNS = [
    "event_date",
    "available_date",
    "event_type",
    "is_scheduled",
    "rate_lower",
    "rate_upper",
    "hawkish_score",
    "statement_text",
    "source",
]


def load_meetings(extra_csv: str | None = None) -> pd.DataFrame:
    """Bundled FOMC decision dates, with rows from ``extra_csv`` added or overriding."""
    with BUILTIN_MEETINGS.open() as handle:
        frames = [pd.read_csv(handle, comment="#")]
    if extra_csv and Path(extra_csv).exists():
        frames.append(pd.read_csv(extra_csv, comment="#"))
    meetings = pd.concat(frames, ignore_index=True)
    meetings["event_date"] = pd.to_datetime(meetings["event_date"])
    meetings["is_scheduled"] = meetings["is_scheduled"].astype(str).str.lower().isin({"true", "1"})
    return meetings.drop_duplicates("event_date", keep="last").sort_values(
        "event_date", ignore_index=True
    )


def _decision(
    event_date: pd.Timestamp, upper: pd.Series, lower: pd.Series, two_year: pd.Series
) -> dict:
    effective = event_date + pd.Timedelta(days=1)
    before = two_year.loc[two_year.index < event_date]
    after = two_year.loc[two_year.index >= event_date]
    if upper.empty or upper.index.max() < effective or before.empty or after.empty:
        return {}  # not decided yet, or FRED has not caught up
    reaction_day = after.index[0]
    if (reaction_day - event_date).days > MAX_REACTION_DAYS:
        return {}
    return {
        "available_date": reaction_day + BDay(1),
        "rate_upper": float(upper.asof(effective)),
        "rate_lower": float(lower.asof(effective)),
        "hawkish_score": float((after.iloc[0] - before.iloc[-1]) * 100),
    }


def build_fomc_events(
    meetings: pd.DataFrame, upper: pd.Series, lower: pd.Series, two_year: pd.Series
) -> pd.DataFrame:
    """One row per meeting; decision fields stay empty until the data covers them."""
    rows: list[dict] = []
    for meeting in meetings.itertuples(index=False):
        event_date = pd.Timestamp(meeting.event_date)
        row = {
            "event_date": event_date,
            "available_date": event_date,
            "event_type": "fomc_meeting",
            "is_scheduled": bool(meeting.is_scheduled),
            "rate_lower": np.nan,
            "rate_upper": np.nan,
            "hawkish_score": np.nan,
            "statement_text": "",
        }
        rows.append(row | _decision(event_date, upper, lower, two_year))
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS[:-1])
    frame["source"] = "fred"
    return frame


class FredFomcSource(DataSource):
    kind = DataKind.CENTRAL_BANK
    name = "fred"

    def __init__(
        self,
        api_key: str | None = None,
        client: HttpClient | None = None,
        meetings_csv: str | None = None,
    ) -> None:
        api_key = api_key or os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY is required for central_bank provider 'fred'")
        self.fred = FredMacroSource(api_key=api_key, client=client)
        self.meetings_csv = meetings_csv

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        start, end = pd.Timestamp(request.start), pd.Timestamp(request.end)
        meetings = load_meetings(request.options.get("meetings_csv") or self.meetings_csv)
        meetings = meetings.loc[meetings["event_date"] >= start]
        # A little history for the pre-meeting yield; one day past `end` for the
        # target range, which changes the day after a decision.
        window = FetchRequest(
            start=(start - pd.Timedelta(days=14)).date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
        )
        upper, lower, two_year = (
            self._series(series_id, window) for series_id in ("DFEDTARU", "DFEDTARL", "DGS2")
        )
        return build_fomc_events(meetings, upper, lower, two_year.loc[:end])

    def _series(self, series_id: str, window: FetchRequest) -> pd.Series:
        frame = self.fred.fetch_series(MacroSeries(series_id), window)
        frame = frame.dropna(subset=["value"]).sort_values("observation_date")
        return frame.set_index("observation_date")["value"]


class SyntheticCentralBankSource(DataSource):
    kind = DataKind.CENTRAL_BANK
    name = "synthetic"
    requires_network = False
    meeting_months = (1, 3, 5, 6, 7, 9, 10, 12)

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        random = np.random.default_rng(self.random_seed + 404)
        start, end = pd.Timestamp(request.start), pd.Timestamp(request.end)
        rate = 1.0
        rows: list[dict] = []
        for year in range(start.year, end.year + 2):
            for month in self.meeting_months:
                # Third Wednesday of the month, as a stand-in for the real calendar.
                event_date = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
                event_date = event_date[event_date.dayofweek == 2][2]
                if event_date < start:
                    continue
                if event_date > end + pd.DateOffset(years=1):
                    break
                past = event_date <= end
                move = float(random.choice([-0.25, 0.0, 0.0, 0.0, 0.25])) if past else 0.0
                rate = max(0.0, rate + move)
                verb = "raise" if move > 0 else "lower" if move < 0 else "maintain"
                text = (
                    f"The Committee decided to {verb} the target range for the federal funds "
                    f"rate to {rate:.2f} to {rate + 0.25:.2f} percent."
                    if past
                    else ""
                )
                rows.append(
                    {
                        "event_date": event_date,
                        "available_date": event_date,
                        "event_type": "fomc_meeting",
                        "is_scheduled": True,
                        "rate_lower": rate if past else np.nan,
                        "rate_upper": rate + 0.25 if past else np.nan,
                        "hawkish_score": move * 40 if past else np.nan,
                        "statement_text": text,
                        "source": self.name,
                    }
                )
        return pd.DataFrame(rows, columns=EVENT_COLUMNS)
