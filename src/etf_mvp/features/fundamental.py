from __future__ import annotations

import numpy as np
import pandas as pd

from etf_mvp.features.asof import asof_values

METRICS = ("revenue_yoy", "earnings_yoy", "net_margin", "roe")


def _growth(flow: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year growth of a quarterly flow, sign-safe for negative bases."""
    ordered = flow.sort_values("period_end").reset_index(drop=True)
    prior = pd.merge_asof(
        ordered[["period_end"]].assign(lookup=ordered["period_end"] - pd.DateOffset(years=1))[
            ["lookup"]
        ],
        ordered[["period_end", "value"]].rename(columns={"period_end": "lookup"}),
        on="lookup",
        direction="nearest",
        tolerance=pd.Timedelta(days=20),
    )["value"]
    growth = (ordered["value"] - prior) / prior.abs().where(prior.abs() > 0)
    return ordered.assign(metric_value=growth)[["period_end", "available_date", "metric_value"]]


def ticker_metrics(facts: pd.DataFrame) -> pd.DataFrame:
    """Per-report metrics for one ticker, each stamped with when all inputs were public."""
    quarterly = facts.loc[facts["fiscal_period"] == "Q"]
    by_concept = {c: g for c, g in quarterly.groupby("concept")}
    equity = facts.loc[(facts["concept"] == "equity") & (facts["fiscal_period"] == "I")]
    frames: list[pd.DataFrame] = []

    for concept, metric in (("revenue", "revenue_yoy"), ("net_income", "earnings_yoy")):
        if concept in by_concept:
            frames.append(_growth(by_concept[concept]).assign(metric=metric))

    if {"revenue", "net_income"} <= by_concept.keys():
        joined = by_concept["net_income"].merge(
            by_concept["revenue"], on="period_end", suffixes=("_ni", "_rev")
        )
        frames.append(
            pd.DataFrame(
                {
                    "period_end": joined["period_end"],
                    "available_date": joined[["available_date_ni", "available_date_rev"]].max(
                        axis=1
                    ),
                    "metric_value": joined["value_ni"]
                    / joined["value_rev"].where(joined["value_rev"] != 0),
                    "metric": "net_margin",
                }
            )
        )
    if "net_income" in by_concept and not equity.empty:
        joined = by_concept["net_income"].merge(equity, on="period_end", suffixes=("_ni", "_eq"))
        frames.append(
            pd.DataFrame(
                {
                    "period_end": joined["period_end"],
                    "available_date": joined[["available_date_ni", "available_date_eq"]].max(
                        axis=1
                    ),
                    "metric_value": 4
                    * joined["value_ni"]
                    / joined["value_eq"].where(joined["value_eq"] > 0),
                    "metric": "roe",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_fundamental_features(
    facts: pd.DataFrame, dates: pd.DatetimeIndex, holdings: dict[str, list[str]]
) -> pd.DataFrame:
    """Equal-weighted holdings fundamentals per ETF, as known on each trading date."""
    per_ticker: dict[str, pd.DataFrame] = {}
    for ticker, ticker_facts in facts.groupby("ticker"):
        metrics = ticker_metrics(ticker_facts)
        if metrics.empty:
            continue
        daily = {}
        for metric, rows in metrics.groupby("metric"):
            # Same release date for two periods (restatement batch): keep the newest period.
            rows = rows.sort_values(["available_date", "period_end"])
            daily[metric] = asof_values(dates, rows, ["metric_value"])["metric_value"]
        per_ticker[str(ticker)] = pd.DataFrame(daily, index=dates)

    columns: dict[str, pd.Series] = {}
    for etf, members in holdings.items():
        frames = [per_ticker[t.upper()] for t in members if t.upper() in per_ticker]
        if not frames:
            continue
        for metric in METRICS:
            stacked = [frame[metric] for frame in frames if metric in frame]
            if stacked:
                values = pd.concat(stacked, axis=1).replace([np.inf, -np.inf], np.nan)
                columns[f"{etf}__fund_{metric}"] = values.mean(axis=1)
    return pd.DataFrame(columns, index=dates)
