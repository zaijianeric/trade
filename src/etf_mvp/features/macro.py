from __future__ import annotations

import pandas as pd

from etf_mvp.data.sources.macro import MacroSeries
from etf_mvp.features.asof import asof_values, rolling_zscore


def _year_over_year(observations: pd.DataFrame) -> pd.Series:
    ordered = observations.sort_values("observation_date")
    prior = pd.merge_asof(
        ordered.assign(lookup=ordered["observation_date"] - pd.DateOffset(years=1))[["lookup"]],
        ordered[["observation_date", "value"]].rename(columns={"observation_date": "lookup"}),
        on="lookup",
        direction="nearest",
        tolerance=pd.Timedelta(days=20),
    )["value"].to_numpy()
    return pd.Series(ordered["value"].to_numpy() / prior - 1, index=ordered.index)


def build_macro_features(
    macro: pd.DataFrame, dates: pd.DatetimeIndex, series: list[MacroSeries]
) -> pd.DataFrame:
    """Market-wide macro features, aligned on release (available) dates.

    Daily market series: level, 20-day change and 1-year z-score.
    Revised releases (vintage series): year-over-year change and change vs prior release.
    """
    columns: dict[str, pd.Series] = {}
    for spec in series:
        observations = macro.loc[macro["series_id"] == spec.series_id].copy()
        if observations.empty:
            continue
        prefix = f"macro__{spec.series_id.lower()}"
        if spec.vintage:
            observations = observations.sort_values("observation_date")
            observations["yoy"] = _year_over_year(observations)
            observations["change"] = observations["value"].diff()
            daily = asof_values(dates, observations, ["yoy", "change"])
            columns[f"{prefix}__yoy"] = daily["yoy"]
            columns[f"{prefix}__change"] = daily["change"]
        else:
            # Several observations can share a release date; keep the newest observation.
            observations = observations.sort_values(["available_date", "observation_date"])
            level = asof_values(dates, observations, ["value"])["value"]
            columns[f"{prefix}__level"] = level
            columns[f"{prefix}__chg_20d"] = level - level.shift(20)
            columns[f"{prefix}__z_252d"] = rolling_zscore(level, 252)
    return pd.DataFrame(columns, index=dates)
