"""Feature layer: one module per data kind, combined into a single daily table.

Every non-price feature is joined point-in-time (see asof.py), so a row dated T
only contains information that was public by T's close.
"""

from __future__ import annotations

import pandas as pd

from etf_mvp.config import AppConfig
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.service import DataBundle
from etf_mvp.data.sources.macro import MacroSeries
from etf_mvp.features.central_bank import build_central_bank_features
from etf_mvp.features.fundamental import build_fundamental_features
from etf_mvp.features.geopolitical import build_geopolitical_features
from etf_mvp.features.macro import build_macro_features
from etf_mvp.features.news import build_news_features
from etf_mvp.features.price import build_price_features, build_target

# Backward-compatible name: price-only features.
build_features = build_price_features

__all__ = [
    "build_feature_table",
    "build_features",
    "build_price_features",
    "build_target",
    "feature_groups",
]


def build_feature_table(bundle: DataBundle, config: AppConfig) -> pd.DataFrame:
    """Price features plus every enabled kind available in the bundle."""
    features = build_price_features(bundle.prices, config.universe.symbols).set_index("date")
    dates = pd.DatetimeIndex(features.index)
    blocks: list[pd.DataFrame] = [features]

    if config.source(DataKind.MACRO).enabled and DataKind.MACRO in bundle.frames:
        options = config.source(DataKind.MACRO).options
        series = [MacroSeries.from_config(item) for item in options.get("series", [])]
        blocks.append(build_macro_features(bundle.get(DataKind.MACRO), dates, series))
    if config.source(DataKind.CENTRAL_BANK).enabled and DataKind.CENTRAL_BANK in bundle.frames:
        blocks.append(build_central_bank_features(bundle.get(DataKind.CENTRAL_BANK), dates))
    if config.source(DataKind.FUNDAMENTAL).enabled and DataKind.FUNDAMENTAL in bundle.frames:
        holdings = config.source(DataKind.FUNDAMENTAL).options.get("holdings", {})
        blocks.append(build_fundamental_features(bundle.get(DataKind.FUNDAMENTAL), dates, holdings))
    if config.source(DataKind.NEWS).enabled and DataKind.NEWS in bundle.frames:
        topics = config.source(DataKind.NEWS).options.get("topics", [])
        blocks.append(build_news_features(bundle.get(DataKind.NEWS), dates, topics))
    if (
        config.source(DataKind.GEOPOLITICAL).enabled
        and DataKind.GEOPOLITICAL in bundle.frames
    ):
        blocks.append(
            build_geopolitical_features(bundle.get(DataKind.GEOPOLITICAL), dates)
        )

    table = pd.concat(blocks, axis=1)
    table.index.name = "date"
    return table.reset_index()


def feature_groups(columns: list[str]) -> dict[str, str]:
    """Map feature name -> data kind label, for reporting importance by kind."""
    groups = {}
    for column in columns:
        if column.startswith("macro__"):
            groups[column] = DataKind.MACRO.value
        elif column.startswith("cb__"):
            groups[column] = DataKind.CENTRAL_BANK.value
        elif column.startswith("news__"):
            groups[column] = DataKind.NEWS.value
        elif column.startswith("geo__"):
            groups[column] = DataKind.GEOPOLITICAL.value
        elif "__fund_" in column:
            groups[column] = DataKind.FUNDAMENTAL.value
        else:
            groups[column] = DataKind.MARKET.value
    return groups
