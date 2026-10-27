from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as exchange_calendars
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from etf_mvp.data.http import HttpClient
from etf_mvp.data_quality import (
    next_xnys_session,
    prepare_intraday_file,
    write_intraday_source_metadata,
)

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

    def cleaned_bars(self, symbol: str) -> Path:
        return self.data_dir / "clean" / f"{symbol}_5min.parquet"

    def quality_sessions(self, symbol: str) -> Path:
        return self.data_dir.parent / "quality" / "intraday" / f"{symbol}_sessions.parquet"

    def quality_summary(self, symbol: str) -> Path:
        return self.artifacts_dir.parent / "data_quality" / f"{symbol}.json"

    def evaluation_bars(self, symbol: str, session_date: pd.Timestamp) -> Path:
        return self.data_dir / "evaluation" / f"{symbol}_{session_date.date()}_5min.parquet"

    def live_forecast(self, symbol: str) -> Path:
        return self.artifact_dir(symbol) / "live_forecast.csv"

    def live_status(self, symbol: str) -> Path:
        return self.artifact_dir(symbol) / "live_status.json"


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
    model: object


@dataclass
class DecomposedIntradayModel:
    shape_model: Pipeline
    quantile_models: dict[str, dict[float, Pipeline]]
    feature_names: list[str]
    session_feature_names: list[str]


@dataclass(frozen=True)
class IntradayTuningResult:
    symbol: str
    selected_params: dict[str, float | int]
    inner_start: str
    inner_end: str
    folds: int
    trials: list[dict[str, object]]


@dataclass(frozen=True)
class IntradayEnsemblePlan:
    symbol: str
    weights: dict[str, float]
    inner_start: str
    inner_end: str
    folds: int
    component_metrics: list[dict[str, object]]


INTRADAY_DEFAULT_PARAMS: dict[str, float | int] = {
    "n_estimators": 220,
    "learning_rate": 0.035,
    "num_leaves": 23,
    "max_depth": 7,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 0,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.3,
    "reg_lambda": 1.2,
}

INTRADAY_ENSEMBLE_MODELS = ("lightgbm", "extra_trees", "hist_gradient")
INTRADAY_QUANTILES = (0.1, 0.5, 0.9)
INTRADAY_TARGET_COLUMNS = {
    "target_return",
    "target_shape",
    "target_close_return",
    "target_range_return",
}


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
    return min(requested, pd.Timestamp.now(tz="UTC"))


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
    write_intraday_source_metadata(
        path,
        symbol=symbol,
        feed=feed,
        requested_start=start,
        requested_end=end,
        rows=len(combined),
    )
    return combined


def fetch_intraday_symbols(
    paths: IntradayPaths,
    start: str = "2018-01-01",
    end: str | None = None,
    feed: str = "iex",
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in INTRADAY_SYMBOLS:
        frames[symbol] = update_intraday_bars(
            symbol, paths.bars(symbol), start, end, feed
        )
        prepare_intraday_file(
            paths.bars(symbol),
            paths.cleaned_bars(symbol),
            paths.quality_sessions(symbol),
            paths.quality_summary(symbol),
            symbol,
            feed,
        )
    return frames


def prepare_intraday_symbols(
    paths: IntradayPaths, feed: str = "iex"
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for symbol in INTRADAY_SYMBOLS:
        raw_path = paths.bars(symbol)
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing {raw_path}; run intraday-fetch first")
        result = prepare_intraday_file(
            raw_path,
            paths.cleaned_bars(symbol),
            paths.quality_sessions(symbol),
            paths.quality_summary(symbol),
            symbol,
            feed,
        )
        summaries[symbol] = result.summary
    return summaries


def _active_xnys_session(now: datetime | pd.Timestamp | None = None) -> pd.Timestamp | None:
    timestamp = pd.Timestamp(now or datetime.now(UTC))
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    calendar = exchange_calendars.get_calendar("XNYS")
    local_date = timestamp.tz_convert(NEW_YORK).tz_localize(None).normalize()
    try:
        session = calendar.date_to_session(local_date, direction="none")
    except ValueError:
        return None
    if not calendar.session_open(session) <= timestamp < calendar.session_close(session):
        return None
    return pd.Timestamp(session).tz_localize(None)


def _load_intraday_forecast(
    paths: IntradayPaths, symbol: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact_dir = paths.artifact_dir(symbol)
    forecast_path = artifact_dir / "forecast.csv"
    metrics_path = artifact_dir / "metrics.json"
    if not forecast_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Missing model artifacts for {symbol}; run intraday-train first")
    with metrics_path.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return pd.read_csv(forecast_path), metrics


def refresh_live_intraday_forecasts(
    paths: IntradayPaths,
    feed: str = "iex",
    now: datetime | pd.Timestamp | None = None,
) -> dict[str, dict[str, object]]:
    """Fetch the current session and persist forecasts anchored to observed bars."""
    session_date = _active_xnys_session(now)
    if session_date is None:
        return {"status": "market_closed"}

    results: dict[str, dict[str, object]] = {}
    for symbol in INTRADAY_SYMBOLS:
        forecast, metrics = _load_intraday_forecast(paths, symbol)
        forecast_date = pd.Timestamp(str(metrics["forecast_date"])).normalize()
        if forecast_date != session_date:
            results[symbol] = {
                "status": "forecast_date_mismatch",
                "forecast_date": forecast_date.date().isoformat(),
                "session_date": session_date.date().isoformat(),
            }
            continue
        fetched = fetch_alpaca_bars(symbol, session_date.date().isoformat(), feed=feed)
        evaluation_path = paths.evaluation_bars(symbol, session_date)
        existing = pd.read_parquet(evaluation_path) if evaluation_path.exists() else pd.DataFrame()
        bars = pd.concat([existing, fetched], ignore_index=True)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(evaluation_path, index=False)
        rolled = roll_intraday_forecast(forecast, bars, float(metrics["last_close"]))
        live_path = paths.live_forecast(symbol)
        live_path.parent.mkdir(parents=True, exist_ok=True)
        rolled.to_csv(live_path, index=False)
        cutoff_slot = int(bars["slot"].max()) if not bars.empty else -1
        status = {
            "status": "updated",
            "symbol": symbol,
            "session_date": session_date.date().isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "observed_bars": len(bars),
            "cutoff_slot": cutoff_slot,
            "cutoff_time": (
                (pd.Timestamp("09:30") + pd.Timedelta(minutes=5 * cutoff_slot)).strftime("%H:%M")
                if cutoff_slot >= 0
                else None
            ),
            "forecast_path": str(live_path),
            "bars_path": str(evaluation_path),
        }
        with paths.live_status(symbol).open("w", encoding="utf-8") as status_file:
            json.dump(status, status_file, ensure_ascii=False, indent=2)
        results[symbol] = status
    return results


def _intraday_regressor(
    random_seed: int, model_params: dict[str, float | int] | None = None
):
    params = {**INTRADAY_DEFAULT_PARAMS, **(model_params or {})}
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression_l1",
            **params,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        )
    except (ImportError, OSError):
        return ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=(
                None if int(params["max_depth"]) < 0 else int(params["max_depth"])
            ),
            min_samples_leaf=max(4, int(params["min_child_samples"]) // 2),
            max_features=float(params["colsample_bytree"]),
            random_state=random_seed,
            n_jobs=1,
        )


def _intraday_base_regressor(
    name: str,
    random_seed: int,
    model_params: dict[str, float | int] | None = None,
):
    if name == "lightgbm":
        return _intraday_regressor(random_seed, model_params)
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=240,
            max_depth=12,
            min_samples_leaf=18,
            max_features=0.7,
            random_state=random_seed,
            n_jobs=1,
        )
    if name == "hist_gradient":
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=220,
            max_leaf_nodes=23,
            l2_regularization=1.0,
            random_state=random_seed,
        )
    raise ValueError(f"Unknown intraday ensemble model: {name}")


class WeightedIntradayRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        model_params: dict[str, float | int] | None = None,
        weights: dict[str, float] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model_params = model_params
        self.weights = weights
        self.random_seed = random_seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> WeightedIntradayRegressor:
        values = np.asarray(x)
        target = np.asarray(y)
        weights = self.weights or {
            name: 1 / 3 for name in INTRADAY_ENSEMBLE_MODELS
        }
        self.estimators_: dict[str, object] = {}
        importance = np.zeros(values.shape[1], dtype=float)
        for name in INTRADAY_ENSEMBLE_MODELS:
            estimator = _intraday_base_regressor(
                name, self.random_seed, self.model_params
            )
            estimator.fit(values, target)
            self.estimators_[name] = estimator
            raw_importance = getattr(estimator, "feature_importances_", None)
            if raw_importance is not None:
                normalized = np.asarray(raw_importance, dtype=float)
                normalized /= normalized.sum() or 1
                importance += float(weights[name]) * normalized
        self.feature_importances_ = importance
        self.n_features_in_ = values.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x)
        weights = self.weights or {
            name: 1 / 3 for name in INTRADAY_ENSEMBLE_MODELS
        }
        return sum(
            float(weights[name]) * estimator.predict(values)
            for name, estimator in self.estimators_.items()
        )


def _model_pipeline(
    random_seed: int,
    model_params: dict[str, float | int] | None = None,
    regressor_name: str = "lightgbm",
    ensemble_weights: dict[str, float] | None = None,
) -> Pipeline:
    regressor = (
        WeightedIntradayRegressor(model_params, ensemble_weights, random_seed)
        if ensemble_weights is not None
        else _intraday_base_regressor(regressor_name, random_seed, model_params)
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", regressor),
        ]
    )


def _quantile_pipeline(alpha: float, random_seed: int) -> Pipeline:
    try:
        from lightgbm import LGBMRegressor

        regressor = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=140,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=6,
            min_child_samples=24,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.2,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        )
    except (ImportError, OSError):
        regressor = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=140,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=16,
            random_state=random_seed,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", regressor),
        ]
    )


def _add_decomposed_targets(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    labeled = result["target_return"].notna()
    labeled_rows = result.loc[labeled]
    grouped = labeled_rows.groupby("session_date")["target_return"]
    target_low = grouped.transform("min")
    target_high = grouped.transform("max")
    target_range = (target_high - target_low).clip(lower=1e-6)
    result.loc[labeled, "target_shape"] = (
        labeled_rows["target_return"].to_numpy() - target_low.to_numpy()
    ) / target_range.to_numpy()
    result.loc[labeled, "target_close_return"] = grouped.transform("last").to_numpy()
    result.loc[labeled, "target_range_return"] = target_range.to_numpy()
    return result


def _feature_names(features: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features.columns
        if column != "session_date" and column not in INTRADAY_TARGET_COLUMNS
    ]


def _session_feature_frame(
    dataset: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    grouped = dataset.groupby("session_date", sort=True)[feature_names]
    means = grouped.mean().add_prefix("mean__")
    standard_deviations = grouped.std().fillna(0).add_prefix("std__")
    minimums = grouped.min().add_prefix("min__")
    maximums = grouped.max().add_prefix("max__")
    session_features = pd.concat(
        [means, standard_deviations, minimums, maximums], axis=1
    ).reset_index()
    return session_features


def _fit_quantile_models(
    dataset: pd.DataFrame,
    feature_names: list[str],
    random_seed: int,
) -> tuple[dict[str, dict[float, Pipeline]], list[str]]:
    session_features = _session_feature_frame(dataset, feature_names)
    targets = (
        dataset.sort_values(["session_date", "slot"])
        .groupby("session_date", as_index=False)
        .agg(
            target_close_return=("target_close_return", "last"),
            target_range_return=("target_range_return", "last"),
        )
    )
    training = session_features.merge(targets, on="session_date", how="inner")
    session_feature_names = [
        column
        for column in session_features.columns
        if column != "session_date"
    ]
    models: dict[str, dict[float, Pipeline]] = {}
    for target in ("target_close_return", "target_range_return"):
        models[target] = {}
        for quantile_index, quantile in enumerate(INTRADAY_QUANTILES):
            model = _quantile_pipeline(quantile, random_seed + quantile_index)
            model.fit(training[session_feature_names], training[target])
            models[target][quantile] = model
    return models, session_feature_names


def _predict_session_quantiles(
    models: dict[str, dict[float, Pipeline]],
    dataset: pd.DataFrame,
    feature_names: list[str],
    session_feature_names: list[str],
) -> pd.DataFrame:
    result = _session_feature_frame(dataset, feature_names)
    for target, target_models in models.items():
        stem = "close" if target == "target_close_return" else "range"
        predictions = np.column_stack(
            [
                target_models[quantile].predict(result[session_feature_names])
                for quantile in INTRADAY_QUANTILES
            ]
        )
        predictions.sort(axis=1)
        if stem == "range":
            predictions = np.clip(predictions, 0.0005, None)
        for index, quantile in enumerate(INTRADAY_QUANTILES):
            result[f"{stem}_q{int(quantile * 100)}"] = predictions[:, index]
    return result[
        [
            "session_date",
            "close_q10",
            "close_q50",
            "close_q90",
            "range_q10",
            "range_q50",
            "range_q90",
        ]
    ]


def _normalize_shape(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0, 1)
    span = float(np.nanmax(clipped) - np.nanmin(clipped))
    if not np.isfinite(span) or span < 1e-4:
        return np.linspace(0.45, 0.55, len(clipped))
    return (clipped - np.nanmin(clipped)) / span


def _calibrate_paths(
    dataset: pd.DataFrame,
    raw_shape: np.ndarray,
    session_quantiles: pd.DataFrame,
) -> pd.DataFrame:
    result = dataset.copy()
    result["raw_predicted_shape"] = raw_shape
    result = result.merge(session_quantiles, on="session_date", how="left")
    calibrated_parts: list[pd.DataFrame] = []
    for _, frame in result.groupby("session_date", sort=False):
        ordered = frame.sort_values("slot").copy()
        shape = _normalize_shape(ordered["raw_predicted_shape"].to_numpy())
        paths: dict[int, np.ndarray] = {}
        for quantile in (10, 50, 90):
            close_return = float(ordered[f"close_q{quantile}"].iloc[0])
            range_return = float(ordered[f"range_q{quantile}"].iloc[0])
            paths[quantile] = close_return + (shape - shape[-1]) * range_return
        lower = np.minimum.reduce([paths[10], paths[50], paths[90]])
        upper = np.maximum.reduce([paths[10], paths[50], paths[90]])
        ordered["predicted_return_q10"] = lower
        ordered["predicted_return"] = paths[50]
        ordered["predicted_return_q90"] = upper
        calibrated_parts.append(ordered)
    return pd.concat(calibrated_parts, ignore_index=True)


def _probability_up(q10: float, q50: float, q90: float) -> float:
    if q10 >= 0:
        return 0.95
    if q90 <= 0:
        return 0.05
    if q50 >= 0:
        cdf_zero = 0.1 + 0.4 * (0 - q10) / max(q50 - q10, 1e-9)
    else:
        cdf_zero = 0.5 + 0.4 * (0 - q50) / max(q90 - q50, 1e-9)
    return float(np.clip(1 - cdf_zero, 0.05, 0.95))


def roll_intraday_forecast(
    forecast: pd.DataFrame,
    observed_bars: pd.DataFrame,
    previous_close: float,
) -> pd.DataFrame:
    """Anchor the remaining forecast path to observed 5-minute bars."""
    required_forecast = {
        "slot",
        "predicted_return_q10",
        "predicted_return",
        "predicted_return_q90",
    }
    missing = required_forecast.difference(forecast.columns)
    if missing:
        raise ValueError(f"Forecast is missing columns: {', '.join(sorted(missing))}")
    if previous_close <= 0:
        raise ValueError("previous_close must be positive")
    if observed_bars.empty:
        result = forecast.copy()
        result["observed"] = False
        result["forecast_mode"] = "premarket"
        return result
    if not {"slot", "close"}.issubset(observed_bars.columns):
        raise ValueError("Observed bars require slot and close columns")

    result = forecast.sort_values("slot").copy().reset_index(drop=True)
    observed = (
        observed_bars.loc[observed_bars["slot"].isin(result["slot"]), ["slot", "close"]]
        .dropna()
        .drop_duplicates("slot", keep="last")
        .sort_values("slot")
    )
    if observed.empty:
        result["observed"] = False
        result["forecast_mode"] = "premarket"
        return result

    cutoff_slot = int(observed["slot"].max())
    observed_returns = observed.set_index("slot")["close"] / previous_close - 1
    base = result.set_index("slot")
    common_slots = observed_returns.index.intersection(base.index)
    expected_prices = previous_close * (1 + base.loc[common_slots, "predicted_return"])
    realized_step_returns = observed.set_index("slot").loc[common_slots, "close"].pct_change()
    expected_step_returns = expected_prices.pct_change()
    if realized_step_returns.notna().sum() >= 4 and expected_step_returns.std() > 1e-8:
        volatility_scale = float(
            np.clip(
                realized_step_returns.std() / expected_step_returns.std(),
                0.5,
                3.0,
            )
        )
    else:
        volatility_scale = 1.0

    anchor_return = float(observed_returns.loc[cutoff_slot])
    future_mask = result["slot"] > cutoff_slot
    observed_mask = result["slot"].isin(observed_returns.index)
    for suffix in ("q10", "", "q90"):
        column = f"predicted_return_{suffix}" if suffix else "predicted_return"
        base_anchor = float(base.loc[cutoff_slot, column])
        result.loc[future_mask, column] = anchor_return + volatility_scale * (
            result.loc[future_mask, column] - base_anchor
        )
        result.loc[observed_mask, column] = result.loc[
            observed_mask, "slot"
        ].map(observed_returns)

    lower = result[
        ["predicted_return_q10", "predicted_return", "predicted_return_q90"]
    ].min(axis=1)
    upper = result[
        ["predicted_return_q10", "predicted_return", "predicted_return_q90"]
    ].max(axis=1)
    result["predicted_return_q10"] = lower
    result["predicted_return_q90"] = upper
    for suffix in ("q10", "", "q90"):
        return_column = f"predicted_return_{suffix}" if suffix else "predicted_return"
        price_column = f"predicted_price_{suffix}" if suffix else "predicted_price"
        result[price_column] = previous_close * (1 + result[return_column])
    result["observed"] = observed_mask
    result["forecast_mode"] = "rolling"
    result["cutoff_slot"] = cutoff_slot
    result["volatility_scale"] = volatility_scale
    return result


def _bounded_ensemble_weights(
    raw_weights: dict[str, float], maximum: float = 0.6
) -> dict[str, float]:
    total = sum(raw_weights.values()) or 1
    weights = {name: value / total for name, value in raw_weights.items()}
    leader = max(weights, key=weights.get)
    if weights[leader] <= maximum:
        return weights
    remainder = 1 - weights[leader]
    weights[leader] = maximum
    scale = (1 - maximum) / remainder
    for name in weights:
        if name != leader:
            weights[name] *= scale
    return weights


def _intraday_candidates(
    random_seed: int, trials: int
) -> list[dict[str, float | int]]:
    candidates = [dict(INTRADAY_DEFAULT_PARAMS)]
    random = np.random.default_rng(random_seed)
    while len(candidates) < max(1, trials):
        params: dict[str, float | int] = {
            "n_estimators": int(random.choice([140, 180, 220, 280, 340])),
            "learning_rate": float(random.choice([0.02, 0.03, 0.035, 0.05])),
            "num_leaves": int(random.choice([15, 23, 31, 47])),
            "max_depth": int(random.choice([5, 7, 9, -1])),
            "min_child_samples": int(random.choice([20, 40, 60, 90])),
            "subsample": float(random.choice([0.7, 0.85, 1.0])),
            "subsample_freq": int(random.choice([0, 1])),
            "colsample_bytree": float(random.choice([0.65, 0.75, 0.85, 1.0])),
            "reg_alpha": float(random.choice([0.0, 0.2, 0.5, 1.0])),
            "reg_lambda": float(random.choice([0.3, 0.8, 1.2, 2.5, 4.0])),
        }
        if params not in candidates:
            candidates.append(params)
    return candidates


def _session_matrices(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
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
    if "is_observed" in frame:
        observed = frame.pivot_table(
            index="session_date", columns="slot", values="is_observed", aggfunc="last"
        ).reindex(index=closes.index, columns=range(INTRADAY_SLOTS), fill_value=False)
    else:
        observed = pd.DataFrame(True, index=closes.index, columns=range(INTRADAY_SLOTS))
    return closes, returns, volumes, ranges, observed.astype(float)


def build_intraday_features(
    bars: pd.DataFrame,
    include_forecast_session: bool = True,
    enhanced: bool = True,
) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    closes, returns, volumes, ranges, observed = _session_matrices(bars)
    if len(closes) < 260:
        raise ValueError(f"At least 260 complete intraday sessions are required: {len(closes)}")

    forecast_date = next_xnys_session(pd.Timestamp(closes.index.max()))
    all_dates = closes.index
    if include_forecast_session:
        all_dates = all_dates.append(pd.DatetimeIndex([forecast_date]))
    returns = returns.reindex(all_dates)
    volumes = volumes.reindex(all_dates)
    observed = observed.reindex(all_dates)

    lag1 = returns.shift(1)
    lag2 = returns.shift(2)
    lag5 = returns.shift(5)
    lag1_volume = np.log1p(volumes.shift(1))
    lag1_missing = 1 - observed.shift(1)
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
    session_open = closes[0]
    session_close = closes[INTRADAY_SLOTS - 1]
    previous_close = session_close.shift(1)
    open_gap = session_open / previous_close - 1
    intraday_return = session_close / session_open - 1
    opening_30m_return = closes[5] / session_open - 1
    morning_return = closes[29] / session_open - 1
    afternoon_return = session_close / closes[30] - 1
    intraday_returns = closes.pct_change(axis=1)
    session_realized_vol = intraday_returns.std(axis=1)
    path_efficiency = (session_close - session_open).abs() / closes.diff(axis=1).abs().sum(axis=1)
    session_vwap = closes.mul(volumes).sum(axis=1) / total_volume.replace(0, np.nan)
    close_vs_vwap = session_close / session_vwap - 1
    opening_volume_share = volumes.loc[:, :11].sum(axis=1) / total_volume.replace(0, np.nan)
    closing_volume_share = volumes.loc[:, 66:].sum(axis=1) / total_volume.replace(0, np.nan)
    missing_fraction = 1 - observed.mean(axis=1)
    enhanced_daily = {
        "prior_open_gap": open_gap.shift(1).reindex(all_dates),
        "prior_intraday_return": intraday_return.shift(1).reindex(all_dates),
        "prior_opening_30m_return": opening_30m_return.shift(1).reindex(all_dates),
        "prior_morning_return": morning_return.shift(1).reindex(all_dates),
        "prior_afternoon_return": afternoon_return.shift(1).reindex(all_dates),
        "prior_session_realized_vol": session_realized_vol.shift(1).reindex(all_dates),
        "prior_path_efficiency": path_efficiency.shift(1).reindex(all_dates),
        "prior_close_vs_vwap": close_vs_vwap.shift(1).reindex(all_dates),
        "prior_opening_volume_share": opening_volume_share.shift(1).reindex(all_dates),
        "prior_closing_volume_share": closing_volume_share.shift(1).reindex(all_dates),
        "prior_missing_fraction": missing_fraction.shift(1).reindex(all_dates),
        "daily_trend_20d": (
            session_close / session_close.rolling(20).mean() - 1
        ).shift(1).reindex(all_dates),
        "daily_volatility_regime": (
            session_close.pct_change().rolling(5).std()
            / session_close.pct_change().rolling(20).std().replace(0, np.nan)
        ).shift(1).reindex(all_dates),
    }

    records: list[pd.DataFrame] = []
    for session_date in all_dates:
        slots = np.arange(INTRADAY_SLOTS)
        feature_values: dict[str, object] = {
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
                "lag1_same_slot_missing": lag1_missing.loc[session_date].to_numpy(),
                "prior_close_return_1d": close_return_1d.loc[session_date],
                "prior_close_return_5d": close_return_5d.loc[session_date],
                "prior_realized_vol_20d": realized_vol_20d.loc[session_date],
                "prior_session_range": prior_range.loc[session_date],
                "prior_volume_z_20d": prior_volume_z.loc[session_date],
                "target_return": returns.loc[session_date].to_numpy(),
        }
        if enhanced:
            feature_values.update(
                {
                    name: values.loc[session_date]
                    for name, values in enhanced_daily.items()
                }
            )
        frame = pd.DataFrame(feature_values)
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
    model_params: dict[str, float | int] | None = None,
    ensemble_weights: dict[str, float] | None = None,
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
        shape_model = _model_pipeline(
            random_seed, model_params, ensemble_weights=ensemble_weights
        )
        shape_model.fit(train[feature_names], train["target_shape"])
        quantile_models, session_feature_names = _fit_quantile_models(
            train, feature_names, random_seed
        )
        session_quantiles = _predict_session_quantiles(
            quantile_models,
            validation,
            feature_names,
            session_feature_names,
        )
        validation = _calibrate_paths(
            validation,
            shape_model.predict(validation[feature_names]),
            session_quantiles,
        )
        validation["train_end_date"] = train["session_date"].max()
        validation["fold"] = fold_number
        predictions.append(validation)
    return pd.concat(predictions, ignore_index=True)


def tune_intraday_symbol(
    bars: pd.DataFrame,
    symbol: str,
    random_seed: int = 42,
    enhanced_features: bool = True,
    trials: int = 3,
    folds: int = 2,
) -> IntradayTuningResult:
    symbol = _validate_symbol(symbol)
    features, _, _ = build_intraday_features(bars, enhanced=enhanced_features)
    feature_names = [
        column
        for column in features.columns
        if column not in {"session_date", "target_return"}
    ]
    labeled = features.dropna(subset=["target_return"]).copy()
    dates = pd.DatetimeIndex(sorted(labeled["session_date"].unique()))
    split_index = min(max(int(len(dates) * 0.8), 200), len(dates) - 40)
    tuning_dates = dates[:split_index]
    validation_start = max(120, int(len(tuning_dates) * 0.6))
    validation_dates = tuning_dates[validation_start:]
    fold_dates = [part for part in np.array_split(validation_dates, folds) if len(part)]
    if not fold_dates:
        raise ValueError(f"Not enough sessions for rolling tuning: {symbol}")

    trial_rows: list[dict[str, object]] = []
    for trial_number, params in enumerate(
        _intraday_candidates(random_seed, trials), start=1
    ):
        accuracies: list[float] = []
        path_rmses: list[float] = []
        for validation_fold in fold_dates:
            fold_start = pd.Timestamp(validation_fold[0])
            train = labeled.loc[labeled["session_date"] < fold_start]
            validation = labeled.loc[
                labeled["session_date"].isin(validation_fold)
            ].copy()
            model = _model_pipeline(random_seed, params)
            model.fit(train[feature_names], train["target_return"])
            validation["predicted_return"] = model.predict(validation[feature_names])
            close_rows = validation.sort_values(["session_date", "slot"]).groupby(
                "session_date", as_index=False
            ).tail(1)
            direction_correct = close_rows["target_return"].ge(0) == close_rows[
                "predicted_return"
            ].ge(0)
            accuracies.append(float(direction_correct.mean()))
            path_rmses.append(
                float(
                    np.sqrt(
                        mean_squared_error(
                            validation["target_return"], validation["predicted_return"]
                        )
                    )
                )
            )
        mean_accuracy = float(np.mean(accuracies))
        accuracy_std = float(np.std(accuracies))
        mean_path_rmse = float(np.mean(path_rmses))
        score = mean_accuracy - 8 * mean_path_rmse - 0.1 * accuracy_std
        trial_rows.append(
            {
                "trial": trial_number,
                "params": params,
                "mean_direction_accuracy": mean_accuracy,
                "accuracy_std": accuracy_std,
                "mean_path_return_rmse": mean_path_rmse,
                "score": score,
            }
        )
    best = max(trial_rows, key=lambda row: float(row["score"]))
    return IntradayTuningResult(
        symbol=symbol,
        selected_params=dict(best["params"]),
        inner_start=pd.Timestamp(tuning_dates.min()).date().isoformat(),
        inner_end=pd.Timestamp(tuning_dates.max()).date().isoformat(),
        folds=len(fold_dates),
        trials=trial_rows,
    )


def derive_intraday_ensemble_plan(
    bars: pd.DataFrame,
    symbol: str,
    random_seed: int = 42,
    enhanced_features: bool = True,
    model_params: dict[str, float | int] | None = None,
    folds: int = 3,
) -> IntradayEnsemblePlan:
    symbol = _validate_symbol(symbol)
    features, _, _ = build_intraday_features(bars, enhanced=enhanced_features)
    feature_names = [
        column
        for column in features.columns
        if column not in {"session_date", "target_return"}
    ]
    labeled = features.dropna(subset=["target_return"]).copy()
    dates = pd.DatetimeIndex(sorted(labeled["session_date"].unique()))
    split_index = min(max(int(len(dates) * 0.8), 200), len(dates) - 40)
    tuning_dates = dates[:split_index]
    validation_start = max(120, int(len(tuning_dates) * 0.6))
    fold_dates = [
        part
        for part in np.array_split(tuning_dates[validation_start:], folds)
        if len(part)
    ]
    actual_parts: list[np.ndarray] = []
    session_parts: list[np.ndarray] = []
    slot_parts: list[np.ndarray] = []
    predictions = {name: [] for name in INTRADAY_ENSEMBLE_MODELS}
    for validation_fold in fold_dates:
        fold_start = pd.Timestamp(validation_fold[0])
        train = labeled.loc[labeled["session_date"] < fold_start]
        validation = labeled.loc[labeled["session_date"].isin(validation_fold)]
        actual_parts.append(validation["target_return"].to_numpy())
        session_parts.append(validation["session_date"].to_numpy())
        slot_parts.append(validation["slot"].to_numpy())
        for name in INTRADAY_ENSEMBLE_MODELS:
            model = _model_pipeline(
                random_seed,
                model_params,
                regressor_name=name,
            )
            model.fit(train[feature_names], train["target_return"])
            predictions[name].append(model.predict(validation[feature_names]))
    actual = np.concatenate(actual_parts)
    session_dates = np.concatenate(session_parts)
    slots = np.concatenate(slot_parts)
    component_rows: list[dict[str, object]] = []
    raw_weights: dict[str, float] = {}
    joined_predictions: dict[str, np.ndarray] = {}
    for name in INTRADAY_ENSEMBLE_MODELS:
        predicted = np.concatenate(predictions[name])
        joined_predictions[name] = predicted
        evaluation = pd.DataFrame(
            {
                "session_date": session_dates,
                "slot": slots,
                "actual": actual,
                "predicted": predicted,
            }
        )
        close_rows = evaluation.sort_values(["session_date", "slot"]).groupby(
            "session_date", as_index=False
        ).tail(1)
        direction_accuracy = float(
            (close_rows["actual"].ge(0) == close_rows["predicted"].ge(0)).mean()
        )
        path_rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        quality = max(0.01, direction_accuracy - 8 * path_rmse)
        raw_weights[name] = quality
        component_rows.append(
            {
                "model": name,
                "direction_accuracy": direction_accuracy,
                "path_return_rmse": path_rmse,
                "quality_score": quality,
            }
        )
    weights = _bounded_ensemble_weights(raw_weights)
    return IntradayEnsemblePlan(
        symbol=symbol,
        weights=weights,
        inner_start=pd.Timestamp(tuning_dates.min()).date().isoformat(),
        inner_end=pd.Timestamp(tuning_dates.max()).date().isoformat(),
        folds=len(fold_dates),
        component_metrics=component_rows,
    )


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
    enhanced_features: bool = True,
    model_params: dict[str, float | int] | None = None,
    ensemble_weights: dict[str, float] | None = None,
) -> IntradayResult:
    symbol = _validate_symbol(symbol)
    features, forecast_date, latest_close = build_intraday_features(
        bars, enhanced=enhanced_features
    )
    features = _add_decomposed_targets(features)
    feature_names = _feature_names(features)
    labeled = features.dropna(subset=["target_return"]).copy()
    backtest = _walk_forward_backtest(
        labeled,
        feature_names,
        random_seed,
        model_params=model_params,
        ensemble_weights=ensemble_weights,
    )
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
    backtest["predicted_price_q10"] = backtest["previous_close"] * (
        1 + backtest["predicted_return_q10"]
    )
    backtest["predicted_price_q90"] = backtest["previous_close"] * (
        1 + backtest["predicted_return_q90"]
    )
    backtest["timestamp"] = _timestamp_for_slots(
        backtest["session_date"], backtest["slot"]
    )
    backtest_sessions = _session_backtest(backtest)

    shape_model = _model_pipeline(
        random_seed, model_params, ensemble_weights=ensemble_weights
    )
    shape_model.fit(labeled[feature_names], labeled["target_shape"])
    quantile_models, session_feature_names = _fit_quantile_models(
        labeled, feature_names, random_seed
    )
    model = DecomposedIntradayModel(
        shape_model=shape_model,
        quantile_models=quantile_models,
        feature_names=feature_names,
        session_feature_names=session_feature_names,
    )
    future = features.loc[features["session_date"] == forecast_date].copy()
    future_quantiles = _predict_session_quantiles(
        quantile_models,
        future,
        feature_names,
        session_feature_names,
    )
    future = _calibrate_paths(
        future,
        shape_model.predict(future[feature_names]),
        future_quantiles,
    )
    future["predicted_price"] = latest_close * (1 + future["predicted_return"])
    future["predicted_price_q10"] = latest_close * (
        1 + future["predicted_return_q10"]
    )
    future["predicted_price_q90"] = latest_close * (
        1 + future["predicted_return_q90"]
    )
    future["timestamp"] = _timestamp_for_slots(future["session_date"], future["slot"])
    forecast = future[
        [
            "timestamp",
            "session_date",
            "slot",
            "predicted_return_q10",
            "predicted_return",
            "predicted_return_q90",
            "predicted_price_q10",
            "predicted_price",
            "predicted_price_q90",
        ]
    ].reset_index(drop=True)

    close_actual = backtest_sessions["actual_close_return"]
    close_predicted = backtest_sessions["predicted_close_return"]
    metrics: dict[str, float | int | str] = {
        "symbol": symbol,
        "model": (
            "DecomposedQuantile+"
            f"{shape_model.named_steps['regressor'].__class__.__name__}"
        ),
        "target_design": "normalized_shape+close_level+session_range",
        "quantiles": "10/50/90",
        "feature_count": len(feature_names),
        "feature_set": "enhanced" if enhanced_features else "baseline",
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
        "interval_80_coverage": float(
            (
                backtest["target_return"].ge(backtest["predicted_return_q10"])
                & backtest["target_return"].le(backtest["predicted_return_q90"])
            ).mean()
        ),
        "median_interval_width_pct": float(
            (backtest["predicted_return_q90"] - backtest["predicted_return_q10"]).median()
        ),
        "forecast_probability_up": _probability_up(
            float(forecast["predicted_return_q10"].iloc[-1]),
            float(forecast["predicted_return"].iloc[-1]),
            float(forecast["predicted_return_q90"].iloc[-1]),
        ),
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


def train_intraday_symbols(
    paths: IntradayPaths,
    random_seed: int = 42,
    feed: str = "iex",
    tuning_trials: int = 3,
    tuning_folds: int = 2,
) -> list[IntradayResult]:
    prepare_intraday_symbols(paths, feed)
    results: list[IntradayResult] = []
    ablation_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    for index, symbol in enumerate(INTRADAY_SYMBOLS):
        bars_path = paths.cleaned_bars(symbol)
        bars = pd.read_parquet(bars_path)
        baseline = train_intraday_symbol(
            bars, symbol, random_seed + index, enhanced_features=False
        )
        enhanced = train_intraday_symbol(
            bars, symbol, random_seed + index, enhanced_features=True
        )
        accuracy_delta = float(enhanced.metrics["direction_accuracy"]) - float(
            baseline.metrics["direction_accuracy"]
        )
        rmse_delta = float(enhanced.metrics["path_price_rmse_pct"]) - float(
            baseline.metrics["path_price_rmse_pct"]
        )
        accepted = accuracy_delta >= 0.015 or rmse_delta <= -0.001
        selected_feature_set = "enhanced" if accepted else "baseline"
        fixed_result = enhanced if accepted else baseline
        tuning = tune_intraday_symbol(
            bars,
            symbol,
            random_seed + index,
            enhanced_features=accepted,
            trials=tuning_trials,
            folds=tuning_folds,
        )
        tuned_result = train_intraday_symbol(
            bars,
            symbol,
            random_seed + index,
            enhanced_features=accepted,
            model_params=tuning.selected_params,
        )
        tuning_accuracy_delta = float(tuned_result.metrics["direction_accuracy"]) - float(
            fixed_result.metrics["direction_accuracy"]
        )
        tuning_rmse_delta = float(tuned_result.metrics["path_price_rmse_pct"]) - float(
            fixed_result.metrics["path_price_rmse_pct"]
        )
        tuning_accepted = (
            tuning_accuracy_delta >= 0.01 or tuning_rmse_delta <= -0.0005
        ) and tuning_accuracy_delta >= -0.01
        single_result = tuned_result if tuning_accepted else fixed_result
        selected_params = (
            tuning.selected_params if tuning_accepted else INTRADAY_DEFAULT_PARAMS
        )
        ensemble_plan = derive_intraday_ensemble_plan(
            bars,
            symbol,
            random_seed + index,
            enhanced_features=accepted,
            model_params=selected_params,
            folds=tuning_folds,
        )
        ensemble_result = train_intraday_symbol(
            bars,
            symbol,
            random_seed + index,
            enhanced_features=accepted,
            model_params=selected_params,
            ensemble_weights=ensemble_plan.weights,
        )
        ensemble_accuracy_delta = float(
            ensemble_result.metrics["direction_accuracy"]
        ) - float(single_result.metrics["direction_accuracy"])
        ensemble_rmse_delta = float(
            ensemble_result.metrics["path_price_rmse_pct"]
        ) - float(single_result.metrics["path_price_rmse_pct"])
        ensemble_accepted = (
            ensemble_accuracy_delta >= 0.01 or ensemble_rmse_delta <= -0.0005
        ) and ensemble_accuracy_delta >= -0.01
        result = ensemble_result if ensemble_accepted else single_result
        result.metrics["parameter_set"] = "tuned" if tuning_accepted else "fixed"
        result.metrics["model_strategy"] = "ensemble" if ensemble_accepted else "single"
        save_intraday_result(result, paths)
        results.append(result)
        ablation_rows.append(
            {
                "symbol": symbol,
                "backtest_start": enhanced.metrics["backtest_start"],
                "backtest_end": enhanced.metrics["backtest_end"],
                "backtest_sessions": enhanced.metrics["backtest_sessions"],
                "baseline_feature_count": baseline.feature_count,
                "enhanced_feature_count": enhanced.feature_count,
                "baseline_direction_accuracy": baseline.metrics["direction_accuracy"],
                "enhanced_direction_accuracy": enhanced.metrics["direction_accuracy"],
                "direction_accuracy_delta": accuracy_delta,
                "baseline_path_rmse_pct": baseline.metrics["path_price_rmse_pct"],
                "enhanced_path_rmse_pct": enhanced.metrics["path_price_rmse_pct"],
                "path_rmse_delta": rmse_delta,
                "accepted": accepted,
                "selected_feature_set": selected_feature_set,
            }
        )
        tuning_rows.append(
            {
                "symbol": symbol,
                "feature_set": selected_feature_set,
                "inner_start": tuning.inner_start,
                "inner_end": tuning.inner_end,
                "folds": tuning.folds,
                "trial_count": len(tuning.trials),
                "fixed_direction_accuracy": fixed_result.metrics["direction_accuracy"],
                "tuned_direction_accuracy": tuned_result.metrics["direction_accuracy"],
                "direction_accuracy_delta": tuning_accuracy_delta,
                "fixed_path_rmse_pct": fixed_result.metrics["path_price_rmse_pct"],
                "tuned_path_rmse_pct": tuned_result.metrics["path_price_rmse_pct"],
                "path_rmse_delta": tuning_rmse_delta,
                "accepted": tuning_accepted,
                "selected_parameter_set": "tuned" if tuning_accepted else "fixed",
                "selected_params": tuning.selected_params,
                "trials": tuning.trials,
            }
        )
        ensemble_rows.append(
            {
                "symbol": symbol,
                "feature_set": selected_feature_set,
                "parameter_set": "tuned" if tuning_accepted else "fixed",
                "inner_start": ensemble_plan.inner_start,
                "inner_end": ensemble_plan.inner_end,
                "folds": ensemble_plan.folds,
                "single_direction_accuracy": single_result.metrics[
                    "direction_accuracy"
                ],
                "ensemble_direction_accuracy": ensemble_result.metrics[
                    "direction_accuracy"
                ],
                "direction_accuracy_delta": ensemble_accuracy_delta,
                "single_path_rmse_pct": single_result.metrics["path_price_rmse_pct"],
                "ensemble_path_rmse_pct": ensemble_result.metrics[
                    "path_price_rmse_pct"
                ],
                "path_rmse_delta": ensemble_rmse_delta,
                "accepted": ensemble_accepted,
                "selected_strategy": "ensemble" if ensemble_accepted else "single",
                "weights": ensemble_plan.weights,
                "components": ensemble_plan.component_metrics,
            }
        )
    feature_ablation_path = paths.artifacts_dir.parent / "feature_ablation" / "intraday.json"
    feature_ablation_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_ablation_path.open("w", encoding="utf-8") as ablation_file:
        json.dump(ablation_rows, ablation_file, ensure_ascii=False, indent=2)
    rolling_tuning_path = paths.artifacts_dir.parent / "rolling_tuning" / "intraday.json"
    rolling_tuning_path.parent.mkdir(parents=True, exist_ok=True)
    with rolling_tuning_path.open("w", encoding="utf-8") as tuning_file:
        json.dump(tuning_rows, tuning_file, ensure_ascii=False, indent=2)
    ensemble_path = paths.artifacts_dir.parent / "ensemble" / "intraday.json"
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    with ensemble_path.open("w", encoding="utf-8") as ensemble_file:
        json.dump(ensemble_rows, ensemble_file, ensure_ascii=False, indent=2)
    return results


def run_intraday_pipeline(
    paths: IntradayPaths,
    start: str = "2018-01-01",
    end: str | None = None,
    feed: str = "iex",
    random_seed: int = 42,
    tuning_trials: int = 3,
    tuning_folds: int = 2,
) -> list[IntradayResult]:
    fetch_intraday_symbols(paths, start=start, end=end, feed=feed)
    return train_intraday_symbols(
        paths,
        random_seed=random_seed,
        feed=feed,
        tuning_trials=tuning_trials,
        tuning_folds=tuning_folds,
    )
