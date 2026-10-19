"""Point-in-time joins: a value is visible on a trading date only once it was available."""

from __future__ import annotations

import pandas as pd


def to_trading_dates(available: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Map each availability date to the first trading date on or after it (NaT if none)."""
    positions = dates.searchsorted(pd.to_datetime(available).to_numpy(), side="left")
    mapped = pd.Series(pd.NaT, index=available.index, dtype="datetime64[ns]")
    inside = positions < len(dates)
    mapped[inside] = dates[positions[inside]]
    return mapped


def asof_values(
    dates: pd.DatetimeIndex,
    events: pd.DataFrame,
    value_columns: list[str],
    time_column: str = "available_date",
) -> pd.DataFrame:
    """For each trading date, the latest event row with ``time_column <= date``."""
    left = pd.DataFrame({"date": dates})
    right = (
        events[[time_column, *value_columns]]
        .dropna(subset=[time_column])
        .sort_values(time_column, kind="stable")
        .drop_duplicates(time_column, keep="last")
    )
    right[time_column] = pd.to_datetime(right[time_column]).astype("datetime64[ns]")
    left["date"] = left["date"].astype("datetime64[ns]")
    merged = pd.merge_asof(left, right, left_on="date", right_on=time_column, direction="backward")
    return merged.set_index("date")[value_columns]


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.where(std > 0)
