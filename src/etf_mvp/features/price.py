from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).rolling(window).mean()
    loss = -change.clip(upper=0).rolling(window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def build_price_features(prices: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = prices.pivot(index="date", columns="symbol", values="volume").sort_index()
    missing = sorted(set(symbols) - set(close.columns))
    if missing:
        raise ValueError(f"No market data for symbols: {missing}")

    feature_columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        symbol_close = close[symbol]
        symbol_volume = volume[symbol]
        returns_1d = symbol_close.pct_change()
        log_return = np.log(symbol_close).diff()

        for days in (1, 2, 5, 10, 20):
            feature_columns[f"{symbol}__return_{days}d"] = symbol_close.pct_change(days)
        for days in (5, 10, 20):
            feature_columns[f"{symbol}__volatility_{days}d"] = (
                log_return.rolling(days).std() * np.sqrt(252)
            )
        for days in (5, 10, 20, 50):
            moving_average = symbol_close.rolling(days).mean()
            feature_columns[f"{symbol}__close_to_sma_{days}"] = symbol_close / moving_average - 1

        volume_mean = symbol_volume.rolling(20).mean()
        volume_std = symbol_volume.rolling(20).std()
        feature_columns[f"{symbol}__volume_zscore_20d"] = (
            (symbol_volume - volume_mean) / volume_std.replace(0, np.nan)
        )
        feature_columns[f"{symbol}__rsi_14"] = _rsi(symbol_close) / 100
        feature_columns[f"{symbol}__downside_20d"] = (
            returns_1d.clip(upper=0).rolling(20).std() * np.sqrt(252)
        )

    features = pd.DataFrame(feature_columns, index=close.index)
    features.index.name = "date"
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.reset_index()


def build_target(prices: pd.DataFrame, symbol: str, horizon_days: int) -> pd.DataFrame:
    close = (
        prices.loc[prices["symbol"] == symbol, ["date", "close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")["close"]
    )
    forward_return = close.shift(-horizon_days) / close - 1
    target = (forward_return > 0).astype("Int64").mask(forward_return.isna())
    return pd.DataFrame(
        {
            "date": close.index,
            "forward_return": forward_return.to_numpy(),
            "target": target.to_numpy(),
        }
    )
