"""Helpers that turn a public timestamp into the trading date it can first be used."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay

US_EASTERN = "America/New_York"
MARKET_CLOSE_HOUR = 16


def next_business_day(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.normalize() + BDay(1)


def available_after_close(timestamps: pd.Series, tz: str | None = US_EASTERN) -> pd.Series:
    """Date whose close first includes the event.

    Events at or after 16:00 ET, or on a non-business day, roll to the next business day.
    Naive timestamps are assumed to already be in ``tz``.
    """
    stamps = pd.to_datetime(timestamps)
    if stamps.dt.tz is not None and tz:
        stamps = stamps.dt.tz_convert(tz).dt.tz_localize(None)
    day = stamps.dt.normalize()
    late = stamps.dt.hour >= MARKET_CLOSE_HOUR
    rolled = day.where(~late, day + BDay(1))
    weekend = rolled.dt.dayofweek >= 5
    return rolled.where(~weekend, rolled + BDay(1))
