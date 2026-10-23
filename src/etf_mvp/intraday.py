from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from etf_mvp.data.http import HttpClient

INTRADAY_SYMBOLS = ("QQQ", "SOXX")
INTRADAY_SLOTS = 78
NEW_YORK = ZoneInfo("America/New_York")
ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


@dataclass(frozen=True)
class IntradayPaths:
    data_dir: Path
    artifacts_dir: Path

    def bars(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol}_5min.parquet"

    def artifact_dir(self, symbol: str) -> Path:
        return self.artifacts_dir / symbol


@dataclass(frozen=True)
class IntradayResult:
    symbol: str
    forecast_date: pd.Timestamp
    model_name: str
    feature_count: int
    metrics: dict[str, float | int | str]
    forecast: pd.DataFrame
    backtest_bars: pd.DataFrame
    backtest_sessions: pd.DataFrame
    model: Pipeline


def _validate_symbol(symbol: str) -> str:
    normalized = symbol.upper()
    if normalized not in INTRADAY_SYMBOLS:
        raise ValueError(
            f"Intraday models are restricted to {', '.join(INTRADAY_SYMBOLS)}: {symbol}"
        )
    return normalized


def _alpaca_credentials() -> tuple[str, str]:
    key_id = os.getenv("ALPACA_API_KEY_ID", "").strip()
    secret_key = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    if not key_id or not secret_key:
        raise RuntimeError(
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY before fetching "
            "multi-year 5-minute bars."
        )
    return key_id, secret_key


def _request_end(end: str | None) -> pd.Timestamp:
    requested = pd.Timestamp(end or date.today().isoformat(), tz="UTC") + pd.Timedelta(
        days=1
    )
    delayed_now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    return min(requested, delayed_now)


def _as_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def parse_alpaca_bars(symbol: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    symbol = _validate_symbol(symbol)
    columns = [
        "symbol",
        "timestamp",
        "session_date",
        "slot",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "trade_count",
            "vw": "vwap",
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    local_time = frame["timestamp"].dt.tz_convert(NEW_YORK)
    minutes = local_time.dt.hour * 60 + local_time.dt.minute
    frame["session_date"] = local_time.dt.tz_localize(None).dt.normalize()
    frame["slot"] = ((minutes - 9 * 60 - 30) // 5).astype(int)
    frame["symbol"] = symbol
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    regular_session = frame["slot"].between(0, INTRADAY_SLOTS - 1)
    frame = frame.loc[regular_session, columns]
    return frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def fetch_alpaca_bars(
    symbol: str,
    start: str,
    end: str | None = None,
    feed: str = "iex",
    client: HttpClient | None = None,
) -> pd.DataFrame:
    symbol = _validate_symbol(symbol)
    key_id, secret_key = _alpaca_credentials()
    http = client or HttpClient(min_interval_seconds=0.35, timeout_seconds=60)
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
    }
    params: dict[str, object] = {
        "timeframe": "5Min",
        "start": _as_utc(start).isoformat(),
        "end": _request_end(end).isoformat(),
        "limit": 10_000,
        "adjustment": "all",
        "feed": feed,
        "sort": "asc",
    }
    rows: list[dict[str, object]] = []
    while True:
        response = http.get(
            ALPACA_BARS_URL.format(symbol=symbol), headers=headers, params=params
        )
        payload = response.json()
        if payload.get("message") and not payload.get("bars"):
            raise RuntimeError(f"Alpaca market data error: {payload['message']}")
        rows.extend(payload.get("bars") or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
        params["page_token"] = page_token
    return parse_alpaca_bars(symbol, rows)


def update_intraday_bars(
    symbol: str,
    path: Path,
    start: str,
    end: str | None = None,
    feed: str = "iex",
) -> pd.DataFrame:
    symbol = _validate_symbol(symbol)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    fetch_start = start
    if not existing.empty:
        latest = pd.to_datetime(existing["timestamp"], utc=True).max()
        fetch_start = (latest + pd.Timedelta(minutes=5)).isoformat()
    fetched = fetch_alpaca_bars(symbol, fetch_start, end=end, feed=feed)
    combined = pd.concat([existing, fetched], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.drop_duplicates("timestamp", keep="last").sort_values(
        "timestamp"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def fetch_intraday_symbols(
    paths: IntradayPaths,
    start: str = "2018-01-01",
    end: str | None = None,
    feed: str = "iex",
) -> dict[str, pd.DataFrame]:
    return {
        symbol: update_intraday_bars(symbol, paths.bars(symbol), start, end, feed)
        for symbol in INTRADAY_SYMBOLS
    }


def _intraday_regressor(random_seed: int):
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=23,
            max_depth=7,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.3,
            reg_lambda=1.2,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        )
    except (ImportError, OSError):
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=random_seed,
            n_jobs=1,
        )


def _model_pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", _intraday_regressor(random_seed)),
        ]
    )


def _session_matrices(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    frame = bars.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    counts = frame.groupby("session_date")["slot"].nunique()
    valid_dates = counts[counts >= 70].index
    frame = frame.loc[frame["session_date"].isin(valid_dates)]

    closes = frame.pivot_table(
        index="session_date", columns="slot", values="close", aggfunc="last"
    ).reindex(columns=range(INTRADAY_SLOTS))
    closes = closes.interpolate(axis=1, limit_direction="both").dropna()
    volumes = frame.pivot_table(
        index="session_date", columns="slot", values="volume", aggfunc="sum"
    ).reindex(index=closes.index, columns=range(INTRADAY_SLOTS), fill_value=0)
    ranges = frame.groupby("session_date").agg(high=("high", "max"), low=("low", "min"))
    ranges = (ranges["high"] / ranges["low"] - 1).reindex(closes.index)
    previous_close = closes[INTRADAY_SLOTS - 1].shift(1)
    returns = closes.div(previous_close, axis=0) - 1
    return closes, returns, volumes, ranges


def build_intraday_features(
    bars: pd.DataFrame,
    include_forecast_session: bool = True,
) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    closes, returns, volumes, ranges = _session_matrices(bars)
    if len(closes) < 260:
        raise ValueError(f"At least 260 complete intraday sessions are required: {len(closes)}")

    forecast_date = pd.Timestamp(closes.index.max()) + BDay(1)
    all_dates = closes.index
    if include_forecast_session:
        all_dates = all_dates.append(pd.DatetimeIndex([forecast_date]))
    returns = returns.reindex(all_dates)
    volumes = volumes.reindex(all_dates)

    lag1 = returns.shift(1)
    lag2 = returns.shift(2)
    lag5 = returns.shift(5)
    lag1_volume = np.log1p(volumes.shift(1))
    mean5 = returns.shift(1).rolling(5, min_periods=3).mean()
    mean20 = returns.shift(1).rolling(20, min_periods=10).mean()
    std20 = returns.shift(1).rolling(20, min_periods=10).std()
    daily_close = closes[INTRADAY_SLOTS - 1]
    close_return_1d = daily_close.pct_change().shift(1).reindex(all_dates)
    close_return_5d = daily_close.pct_change(5).shift(1).reindex(all_dates)
    realized_vol_20d = daily_close.pct_change().shift(1).rolling(20).std().reindex(all_dates)
    prior_range = ranges.shift(1).reindex(all_dates)
    total_volume = volumes.sum(axis=1)
    prior_volume_z = (
        (total_volume.shift(1) - total_volume.shift(1).rolling(20).mean())
        / total_volume.shift(1).rolling(20).std()
    )

    records: list[pd.DataFrame] = []
    for session_date in all_dates:
        slots = np.arange(INTRADAY_SLOTS)
        frame = pd.DataFrame(
            {
                "session_date": session_date,
                "slot": slots,
                "slot_sin": np.sin(2 * np.pi * slots / INTRADAY_SLOTS),
                "slot_cos": np.cos(2 * np.pi * slots / INTRADAY_SLOTS),
                "lag1_same_slot_return": lag1.loc[session_date].to_numpy(),
                "lag2_same_slot_return": lag2.loc[session_date].to_numpy(),
                "lag5_same_slot_return": lag5.loc[session_date].to_numpy(),
                "lag1_previous_slot_return": lag1.loc[session_date].shift(1).to_numpy(),
                "lag1_next_slot_return": lag1.loc[session_date].shift(-1).to_numpy(),
                "same_slot_mean_5d": mean5.loc[session_date].to_numpy(),
                "same_slot_mean_20d": mean20.loc[session_date].to_numpy(),
                "same_slot_std_20d": std20.loc[session_date].to_numpy(),
                "lag1_log_volume": lag1_volume.loc[session_date].to_numpy(),
                "prior_close_return_1d": close_return_1d.loc[session_date],
                "prior_close_return_5d": close_return_5d.loc[session_date],
                "prior_realized_vol_20d": realized_vol_20d.loc[session_date],
                "prior_session_range": prior_range.loc[session_date],
                "prior_volume_z_20d": prior_volume_z.loc[session_date],
                "target_return": returns.loc[session_date].to_numpy(),
            }
        )
        records.append(frame)
    features = pd.concat(records, ignore_index=True)
    latest_close = float(closes.iloc[-1, INTRADAY_SLOTS - 1])
    return features, forecast_date, latest_close


def _timestamp_for_slots(session_dates: pd.Series, slots: pd.Series) -> pd.Series:
    base = pd.to_datetime(session_dates) + pd.Timedelta(hours=9, minutes=30)
    local = base.dt.tz_localize(NEW_YORK) + pd.to_timedelta(slots * 5, unit="m")
    return local


def _walk_forward_backtest(
    dataset: pd.DataFrame,
    feature_names: list[str],
    random_seed: int,
    test_fraction: float = 0.2,
    folds: int = 5,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(dataset["session_date"].unique()))
    split_index = min(max(int(len(dates) * (1 - test_fraction)), 200), len(dates) - 40)
    test_dates = dates[split_index:]
    fold_dates = [fold for fold in np.array_split(test_dates, folds) if len(fold)]
    predictions: list[pd.DataFrame] = []
    for fold_number, validation_dates in enumerate(fold_dates, start=1):
        fold_start = pd.Timestamp(validation_dates[0])
        train = dataset.loc[dataset["session_date"] < fold_start]
        validation = dataset.loc[dataset["session_date"].isin(validation_dates)].copy()
        model = _model_pipeline(random_seed)
        model.fit(train[feature_names], train["target_return"])
        validation["predicted_return"] = model.predict(validation[feature_names])
        validation["train_end_date"] = train["session_date"].max()
        validation["fold"] = fold_number
        predictions.append(validation)
    return pd.concat(predictions, ignore_index=True)


def _session_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_date, frame in backtest.groupby("session_date", sort=True):
        ordered = frame.sort_values("slot")
        actual = ordered["target_return"]
        predicted = ordered["predicted_return"]
        previous_close = float(ordered["previous_close"].iloc[0])
        rows.append(
            {
                "session_date": session_date,
                "train_end_date": ordered["train_end_date"].iloc[0],
                "fold": int(ordered["fold"].iloc[0]),
                "previous_close": previous_close,
                "actual_open": previous_close * (1 + actual.iloc[0]),
                "predicted_open": previous_close * (1 + predicted.iloc[0]),
                "actual_high": previous_close * (1 + actual.max()),
                "predicted_high": previous_close * (1 + predicted.max()),
                "actual_low": previous_close * (1 + actual.min()),
                "predicted_low": previous_close * (1 + predicted.min()),
                "actual_close": previous_close * (1 + actual.iloc[-1]),
                "predicted_close": previous_close * (1 + predicted.iloc[-1]),
                "actual_close_return": float(actual.iloc[-1]),
                "predicted_close_return": float(predicted.iloc[-1]),
            }
        )
    result = pd.DataFrame(rows)
    result["direction_correct"] = result["actual_close_return"].ge(0) == result[
        "predicted_close_return"
    ].ge(0)
    return result


def train_intraday_symbol(
    bars: pd.DataFrame,
    symbol: str,
    random_seed: int = 42,
) -> IntradayResult:
    symbol = _validate_symbol(symbol)
    features, forecast_date, latest_close = build_intraday_features(bars)
    feature_names = [
        column
        for column in features.columns
        if column not in {"session_date", "target_return"}
    ]
    labeled = features.dropna(subset=["target_return"]).copy()
    backtest = _walk_forward_backtest(labeled, feature_names, random_seed)
    previous_close_map = (
        bars.assign(session_date=pd.to_datetime(bars["session_date"]))
        .sort_values(["session_date", "slot"])
        .groupby("session_date")["close"]
        .last()
        .shift(1)
    )
    backtest["previous_close"] = backtest["session_date"].map(previous_close_map)
    backtest["actual_price"] = backtest["previous_close"] * (
        1 + backtest["target_return"]
    )
    backtest["predicted_price"] = backtest["previous_close"] * (
        1 + backtest["predicted_return"]
    )
    backtest["timestamp"] = _timestamp_for_slots(
        backtest["session_date"], backtest["slot"]
    )
    backtest_sessions = _session_backtest(backtest)

    model = _model_pipeline(random_seed)
    model.fit(labeled[feature_names], labeled["target_return"])
    future = features.loc[features["session_date"] == forecast_date].copy()
    future["predicted_return"] = model.predict(future[feature_names])
    future["predicted_price"] = latest_close * (1 + future["predicted_return"])
    future["timestamp"] = _timestamp_for_slots(future["session_date"], future["slot"])
    forecast = future[
        ["timestamp", "session_date", "slot", "predicted_return", "predicted_price"]
    ].reset_index(drop=True)

    close_actual = backtest_sessions["actual_close_return"]
    close_predicted = backtest_sessions["predicted_close_return"]
    metrics: dict[str, float | int | str] = {
        "symbol": symbol,
        "model": model.named_steps["regressor"].__class__.__name__,
        "feature_count": len(feature_names),
        "train_sessions": int(labeled["session_date"].nunique()),
        "backtest_sessions": len(backtest_sessions),
        "backtest_start": backtest_sessions["session_date"].min().date().isoformat(),
        "backtest_end": backtest_sessions["session_date"].max().date().isoformat(),
        "bar_return_mae": float(
            mean_absolute_error(backtest["target_return"], backtest["predicted_return"])
        ),
        "path_price_rmse_pct": float(
            np.sqrt(mean_squared_error(backtest["actual_price"], backtest["predicted_price"]))
            / backtest["previous_close"].mean()
        ),
        "close_return_mae": float(mean_absolute_error(close_actual, close_predicted)),
        "direction_accuracy": float(backtest_sessions["direction_correct"].mean()),
        "high_price_mae_pct": float(
            np.mean(
                np.abs(
                    backtest_sessions["actual_high"] - backtest_sessions["predicted_high"]
                )
                / backtest_sessions["previous_close"]
            )
        ),
        "low_price_mae_pct": float(
            np.mean(
                np.abs(
                    backtest_sessions["actual_low"] - backtest_sessions["predicted_low"]
                )
                / backtest_sessions["previous_close"]
            )
        ),
        "forecast_date": forecast_date.date().isoformat(),
        "last_close": latest_close,
    }
    return IntradayResult(
        symbol=symbol,
        forecast_date=forecast_date,
        model_name=str(metrics["model"]),
        feature_count=len(feature_names),
        metrics=metrics,
        forecast=forecast,
        backtest_bars=backtest,
        backtest_sessions=backtest_sessions,
        model=model,
    )


def save_intraday_result(result: IntradayResult, paths: IntradayPaths) -> None:
    output = paths.artifact_dir(result.symbol)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output / "model.joblib")
    result.forecast.to_csv(output / "forecast.csv", index=False)
    result.backtest_bars.to_parquet(output / "backtest_bars.parquet", index=False)
    result.backtest_sessions.to_csv(output / "backtest_sessions.csv", index=False)
    with (output / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(result.metrics, metrics_file, ensure_ascii=False, indent=2)


def train_intraday_symbols(paths: IntradayPaths, random_seed: int = 42) -> list[IntradayResult]:
    results: list[IntradayResult] = []
    for index, symbol in enumerate(INTRADAY_SYMBOLS):
        bars_path = paths.bars(symbol)
        if not bars_path.exists():
            raise FileNotFoundError(f"Missing {bars_path}; run intraday-fetch first")
        bars = pd.read_parquet(bars_path)
        result = train_intraday_symbol(bars, symbol, random_seed + index)
        save_intraday_result(result, paths)
        results.append(result)
    return results


def run_intraday_pipeline(
    paths: IntradayPaths,
    start: str = "2018-01-01",
    end: str | None = None,
    feed: str = "iex",
    random_seed: int = 42,
) -> list[IntradayResult]:
    fetch_intraday_symbols(paths, start=start, end=end, feed=feed)
    return train_intraday_symbols(paths, random_seed=random_seed)
