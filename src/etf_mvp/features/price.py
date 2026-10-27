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


def build_price_features(
    prices: pd.DataFrame, symbols: Sequence[str], enhanced: bool = True
) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    open_price = prices.pivot(index="date", columns="symbol", values="open").sort_index()
    high = prices.pivot(index="date", columns="symbol", values="high").sort_index()
    low = prices.pivot(index="date", columns="symbol", values="low").sort_index()
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
        if enhanced:
            daily_range = high[symbol] / low[symbol] - 1
            range_width = (high[symbol] - low[symbol]).replace(0, np.nan)
            feature_columns[f"{symbol}__overnight_gap"] = (
                open_price[symbol] / symbol_close.shift(1) - 1
            )
            feature_columns[f"{symbol}__intraday_return"] = (
                symbol_close / open_price[symbol] - 1
            )
            feature_columns[f"{symbol}__range_pct"] = daily_range
            feature_columns[f"{symbol}__close_location"] = (
                (symbol_close - low[symbol]) / range_width - 0.5
            )
            feature_columns[f"{symbol}__drawdown_60d"] = (
                symbol_close / symbol_close.rolling(60).max() - 1
            )
            feature_columns[f"{symbol}__return_skew_20d"] = returns_1d.rolling(20).skew()
            feature_columns[f"{symbol}__momentum_acceleration"] = (
                symbol_close.pct_change(5) - symbol_close.pct_change(20) / 4
            )

    if enhanced:
        daily_returns = close[list(symbols)].pct_change()
        spy_returns = daily_returns["SPY"] if "SPY" in daily_returns else daily_returns.mean(axis=1)
        spy_close = close["SPY"] if "SPY" in close else close[list(symbols)].mean(axis=1)
        for symbol in symbols:
            if symbol == "SPY":
                continue
            covariance = daily_returns[symbol].rolling(60).cov(spy_returns)
            market_variance = spy_returns.rolling(60).var().replace(0, np.nan)
            feature_columns[f"{symbol}__beta_spy_60d"] = covariance / market_variance
            feature_columns[f"{symbol}__correlation_spy_60d"] = (
                daily_returns[symbol].rolling(60).corr(spy_returns)
            )
            feature_columns[f"{symbol}__relative_spy_20d"] = (
                close[symbol].pct_change(20) - spy_close.pct_change(20)
            )

        feature_columns["market__breadth_positive_1d"] = daily_returns.gt(0).mean(axis=1)
        above_sma20 = close[list(symbols)].div(close[list(symbols)].rolling(20).mean()).gt(1)
        feature_columns["market__breadth_above_sma20"] = above_sma20.mean(axis=1)
        feature_columns["market__dispersion_1d"] = daily_returns.std(axis=1)
        market_volatility_5d = spy_returns.rolling(5).std()
        market_volatility_20d = spy_returns.rolling(20).std().replace(0, np.nan)
        feature_columns["market__volatility_regime"] = (
            market_volatility_5d / market_volatility_20d
        )
        feature_columns["market__trend_50_200"] = (
            spy_close.rolling(50).mean() / spy_close.rolling(200).mean() - 1
        )
        date_index = pd.DatetimeIndex(close.index)
        feature_columns["calendar__weekday_sin"] = pd.Series(
            np.sin(2 * np.pi * date_index.dayofweek / 5), index=close.index
        )
        feature_columns["calendar__weekday_cos"] = pd.Series(
            np.cos(2 * np.pi * date_index.dayofweek / 5), index=close.index
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
