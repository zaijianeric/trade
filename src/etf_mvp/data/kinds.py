"""Data kinds and their normalized, point-in-time schemas.

Every kind except market prices carries an ``available_date``: the first trading
date at whose close the record was public. Feature builders may only join on
``available_date <= date``; this is what keeps backtests free of look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class DataKind(StrEnum):
    MARKET = "market"
    MACRO = "macro"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    GEOPOLITICAL = "geopolitical"
    CENTRAL_BANK = "central_bank"

    @property
    def label(self) -> str:
        return KIND_LABELS[self]


KIND_LABELS = {
    DataKind.MARKET: "行情",
    DataKind.MACRO: "宏观",
    DataKind.FUNDAMENTAL: "基本面",
    DataKind.NEWS: "新闻",
    DataKind.GEOPOLITICAL: "地缘风险",
    DataKind.CENTRAL_BANK: "美联储议息",
}


@dataclass(frozen=True)
class Schema:
    kind: DataKind
    columns: tuple[str, ...]
    keys: tuple[str, ...]
    date_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    table: str


SCHEMAS: dict[DataKind, Schema] = {
    DataKind.MARKET: Schema(
        kind=DataKind.MARKET,
        columns=("date", "symbol", "open", "high", "low", "close", "volume"),
        keys=("date", "symbol"),
        date_columns=("date",),
        numeric_columns=("open", "high", "low", "close", "volume"),
        table="market_prices",
    ),
    DataKind.MACRO: Schema(
        kind=DataKind.MACRO,
        columns=("series_id", "observation_date", "available_date", "value", "source"),
        keys=("series_id", "observation_date"),
        date_columns=("observation_date", "available_date"),
        numeric_columns=("value",),
        table="macro_observations",
    ),
    DataKind.FUNDAMENTAL: Schema(
        kind=DataKind.FUNDAMENTAL,
        columns=(
            "ticker",
            "concept",
            "period_end",
            "fiscal_period",
            "form",
            "available_date",
            "value",
            "source",
        ),
        keys=("ticker", "concept", "period_end", "fiscal_period"),
        date_columns=("period_end", "available_date"),
        numeric_columns=("value",),
        table="fundamentals",
    ),
    DataKind.NEWS: Schema(
        kind=DataKind.NEWS,
        columns=(
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
        ),
        keys=("article_id",),
        date_columns=("available_date",),
        numeric_columns=("sentiment",),
        table="news_articles",
    ),
    DataKind.GEOPOLITICAL: Schema(
        kind=DataKind.GEOPOLITICAL,
        columns=(
            "observation_date",
            "available_date",
            "gpr",
            "threats",
            "acts",
            "source",
        ),
        keys=("observation_date",),
        date_columns=("observation_date", "available_date"),
        numeric_columns=("gpr", "threats", "acts"),
        table="geopolitical_risk",
    ),
    DataKind.CENTRAL_BANK: Schema(
        kind=DataKind.CENTRAL_BANK,
        columns=(
            "event_date",
            "available_date",
            "event_type",
            "is_scheduled",
            "rate_lower",
            "rate_upper",
            "hawkish_score",
            "statement_text",
            "source",
        ),
        keys=("event_date", "event_type"),
        date_columns=("event_date", "available_date"),
        numeric_columns=("rate_lower", "rate_upper", "hawkish_score"),
        table="central_bank_events",
    ),
}


def normalize(kind: DataKind, frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a source's output into the kind's schema, sorted and de-duplicated."""
    schema = SCHEMAS[kind]
    missing = sorted(set(schema.columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{kind} data missing columns: {missing}")

    result = frame.loc[:, list(schema.columns)].copy()
    for column in schema.date_columns:
        result[column] = pd.to_datetime(result[column]).dt.tz_localize(None).dt.normalize()
    for column in schema.numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "published_at" in result:
        result["published_at"] = pd.to_datetime(result["published_at"]).dt.tz_localize(None)
    if "symbol" in result:
        result["symbol"] = result["symbol"].astype(str).str.upper()
    if "ticker" in result:
        result["ticker"] = result["ticker"].astype(str).str.upper()

    required = [*schema.keys, *(c for c in schema.date_columns if c not in schema.keys)]
    result = result.dropna(subset=required)
    if kind is DataKind.MARKET:
        result = result.dropna(subset=["close"])
    result = result.drop_duplicates(list(schema.keys), keep="last")
    return result.sort_values(list(schema.keys)).reset_index(drop=True)


def empty_frame(kind: DataKind) -> pd.DataFrame:
    return pd.DataFrame(columns=list(SCHEMAS[kind].columns))
