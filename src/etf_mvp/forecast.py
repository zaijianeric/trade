from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from etf_mvp.features import build_price_features, feature_groups

TARGET_COLUMNS = (
    "next_open_return",
    "next_high_return",
    "next_low_return",
    "next_close_return",
)


@dataclass(frozen=True)
class NextSessionForecast:
    symbol: str
    as_of_date: pd.Timestamp
    forecast_date: pd.Timestamp
    last_close: float
    predicted_open: float
    predicted_high: float
    predicted_low: float
    predicted_close: float
    expected_return: float
    close_lower: float
    close_upper: float
    daily_volatility: float
    open_return_mae: float
    high_return_mae: float
    low_return_mae: float
    close_return_mae: float
    direction_accuracy: float
    model_name: str
    feature_count: int
    feature_kinds: tuple[str, ...]
    intraday: pd.DataFrame
    backtest: pd.DataFrame
    feature_importance: pd.DataFrame


def build_next_session_targets(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    symbol_prices = (
        prices.loc[
            prices["symbol"] == symbol,
            ["date", "open", "high", "low", "close"],
        ]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    current_close = symbol_prices["close"]
    targets = pd.DataFrame(index=symbol_prices.index)
    targets["forecast_date"] = symbol_prices.index.to_series().shift(-1).to_numpy()
    targets["next_open_return"] = symbol_prices["open"].shift(-1) / current_close - 1
    targets["next_high_return"] = symbol_prices["high"].shift(-1) / current_close - 1
    targets["next_low_return"] = symbol_prices["low"].shift(-1) / current_close - 1
    targets["next_close_return"] = symbol_prices["close"].shift(-1) / current_close - 1
    targets.index.name = "date"
    return targets.reset_index()


def _regressor(random_seed: int):
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        )
    except (ImportError, OSError):
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=random_seed,
            n_jobs=1,
        )


def _model_pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", _regressor(random_seed)),
        ]
    )


def _stable_seed(symbol: str, forecast_date: pd.Timestamp, random_seed: int) -> int:
    payload = f"{symbol}:{forecast_date.date().isoformat()}:{random_seed}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], byteorder="big")


def simulate_intraday_candles(
    symbol: str,
    forecast_date: pd.Timestamp,
    predicted_open: float,
    predicted_high: float,
    predicted_low: float,
    predicted_close: float,
    daily_volatility: float,
    random_seed: int,
    intervals: int = 79,
) -> pd.DataFrame:
    seed = _stable_seed(symbol, forecast_date, random_seed)
    random = np.random.default_rng(seed)
    step_volatility = daily_volatility / np.sqrt(intervals)

    shocks = random.normal(0, step_volatility, size=intervals)
    cumulative = np.cumsum(shocks)
    bridge = cumulative - np.linspace(1 / intervals, 1, intervals) * cumulative[-1]
    target_log_return = np.log(predicted_close / predicted_open)
    log_closes = (
        np.log(predicted_open)
        + np.linspace(1 / intervals, 1, intervals) * target_log_return
        + bridge * 0.55
    )
    closes = np.exp(log_closes)
    price_range = max(predicted_high - predicted_low, predicted_open * 0.001)
    close_floor = predicted_low + price_range * 0.08
    close_ceiling = predicted_high - price_range * 0.08
    closes = np.clip(closes, close_floor, close_ceiling)
    candidate_indexes = np.arange(4, intervals - 4)
    high_index, low_index = random.choice(candidate_indexes, size=2, replace=False)
    while abs(high_index - low_index) < intervals // 6:
        high_index, low_index = random.choice(candidate_indexes, size=2, replace=False)
    closes[0] = predicted_open
    closes[-1] = predicted_close
    opens = np.concatenate(([predicted_open], closes[:-1]))

    wick_scale = np.maximum(
        np.abs(random.normal(0, step_volatility * 0.35, intervals)), 0.0002
    )
    highs = np.maximum(opens, closes) * (1 + wick_scale)
    lows = np.minimum(opens, closes) * (1 - wick_scale)
    highs = np.minimum(highs, predicted_high)
    lows = np.maximum(lows, predicted_low)

    highs[high_index] = predicted_high
    lows[low_index] = predicted_low
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    session_start = forecast_date.normalize() + pd.Timedelta(hours=9, minutes=30)
    timestamps = pd.date_range(session_start, periods=intervals, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "scenario": "forecast",
        }
    )


def _walk_forward_backtest(
    labeled: pd.DataFrame,
    feature_names: list[str],
    random_seed: int,
    test_fraction: float,
    folds: int = 5,
) -> pd.DataFrame:
    split_index = int(len(labeled) * (1 - test_fraction))
    split_index = min(max(split_index, 200), len(labeled) - 40)
    validation_indexes = np.arange(split_index, len(labeled))
    fold_indexes = [fold for fold in np.array_split(validation_indexes, folds) if len(fold)]
    predictions: list[pd.DataFrame] = []

    for fold_number, indexes in enumerate(fold_indexes, start=1):
        train = labeled.iloc[: indexes[0]]
        validation = labeled.iloc[indexes]
        fold = validation[["date", "forecast_date", *TARGET_COLUMNS]].copy()
        fold["train_end_date"] = pd.Timestamp(train.iloc[-1]["date"])
        fold["fold"] = fold_number
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            model = _model_pipeline(random_seed + target_index)
            model.fit(train[feature_names], train[target_name])
            fold[f"predicted_{target_name}"] = model.predict(
                validation[feature_names]
            )
        predictions.append(fold)

    return pd.concat(predictions, ignore_index=True)


def forecast_next_session(
    prices: pd.DataFrame,
    symbol: str,
    random_seed: int = 42,
    test_fraction: float = 0.2,
    feature_table: pd.DataFrame | None = None,
) -> NextSessionForecast:
    normalized_prices = prices.copy()
    normalized_prices["date"] = pd.to_datetime(normalized_prices["date"])
    symbols = tuple(sorted(normalized_prices["symbol"].unique()))
    if symbol not in symbols:
        raise ValueError(f"No market data for symbol: {symbol}")

    if feature_table is None:
        features = build_price_features(normalized_prices, symbols)
    else:
        features = feature_table.copy()
        features["date"] = pd.to_datetime(features["date"])
        features = features.sort_values("date").drop_duplicates("date", keep="last")
    targets = build_next_session_targets(normalized_prices, symbol)
    dataset = features.merge(targets, on="date", how="left").sort_values("date")
    feature_names = [column for column in features.columns if column != "date"]
    feature_kinds = tuple(sorted(set(feature_groups(feature_names).values())))
    labeled = dataset.dropna(subset=list(TARGET_COLUMNS)).copy()
    if len(labeled) < 260:
        raise ValueError(f"Not enough rows to forecast {symbol}: {len(labeled)}")

    backtest = _walk_forward_backtest(
        labeled,
        feature_names,
        random_seed,
        test_fraction,
    )
    latest_features = dataset.loc[[dataset.index[-1]], feature_names]

    predicted_returns: dict[str, float] = {}
    close_feature_importance = pd.DataFrame(
        columns=["feature", "importance", "local_contribution", "local_share"]
    )
    model_name = ""
    for index, target_name in enumerate(TARGET_COLUMNS):
        model = _model_pipeline(random_seed + index)
        model.fit(labeled[feature_names], labeled[target_name])
        predicted_returns[target_name] = float(model.predict(latest_features)[0])
        model_name = model.named_steps["regressor"].__class__.__name__
        if target_name == "next_close_return":
            regressor = model.named_steps["regressor"]
            importance_values = getattr(regressor, "feature_importances_", None)
            if importance_values is not None:
                local_contributions = np.full(len(feature_names), np.nan, dtype=float)
                local_shares = np.full(len(feature_names), np.nan, dtype=float)
                if model_name == "LGBMRegressor":
                    imputed_latest = model.named_steps["imputer"].transform(
                        latest_features
                    )
                    contribution_values = regressor.predict(
                        imputed_latest, pred_contrib=True
                    )[0]
                    local_contributions = np.asarray(
                        contribution_values[: len(feature_names)], dtype=float
                    )
                    contribution_total = float(np.abs(local_contributions).sum())
                    local_shares = (
                        np.abs(local_contributions) / contribution_total
                        if contribution_total
                        else np.zeros(len(feature_names), dtype=float)
                    )
                close_feature_importance = (
                    pd.DataFrame(
                        {
                            "feature": feature_names,
                            "importance": importance_values,
                            "local_contribution": local_contributions,
                            "local_share": local_shares,
                        }
                    )
                    .sort_values(
                        ["local_share", "importance"], ascending=[False, False]
                    )
                    .reset_index(drop=True)
                )

    close_actual = backtest["next_close_return"].to_numpy()
    close_predicted = backtest["predicted_next_close_return"].to_numpy()
    return_mae = {
        target_name: float(
            mean_absolute_error(
                backtest[target_name],
                backtest[f"predicted_{target_name}"],
            )
        )
        for target_name in TARGET_COLUMNS
    }
    close_return_mae = float(mean_absolute_error(close_actual, close_predicted))
    direction_accuracy = float(np.mean((close_actual >= 0) == (close_predicted >= 0)))
    residuals = close_actual - close_predicted

    symbol_prices = normalized_prices.loc[
        normalized_prices["symbol"] == symbol
    ].sort_values("date")
    latest_price = symbol_prices.iloc[-1]
    backtest = backtest.merge(
        symbol_prices[["date", "close"]].rename(columns={"close": "last_close"}),
        on="date",
        how="left",
    )
    backtest["actual_close"] = backtest["last_close"] * (
        1 + backtest["next_close_return"]
    )
    backtest["predicted_close"] = backtest["last_close"] * (
        1 + backtest["predicted_next_close_return"]
    )
    backtest["direction_correct"] = (
        backtest["next_close_return"].ge(0)
        == backtest["predicted_next_close_return"].ge(0)
    )
    last_close = float(latest_price["close"])
    log_returns = np.log(symbol_prices["close"]).diff()
    daily_volatility = float(log_returns.tail(20).std())
    daily_volatility = float(np.clip(daily_volatility, 0.003, 0.08))

    open_return = float(
        np.clip(
            predicted_returns["next_open_return"],
            -3 * daily_volatility,
            3 * daily_volatility,
        )
    )
    close_return = float(
        np.clip(
            predicted_returns["next_close_return"],
            -3 * daily_volatility,
            3 * daily_volatility,
        )
    )
    high_return = float(
        np.clip(
            predicted_returns["next_high_return"],
            -3 * daily_volatility,
            4 * daily_volatility,
        )
    )
    low_return = float(
        np.clip(
            predicted_returns["next_low_return"],
            -4 * daily_volatility,
            3 * daily_volatility,
        )
    )

    predicted_open = last_close * (1 + open_return)
    predicted_close = last_close * (1 + close_return)
    predicted_high = max(last_close * (1 + high_return), predicted_open, predicted_close)
    predicted_low = min(last_close * (1 + low_return), predicted_open, predicted_close)
    forecast_date = pd.Timestamp(latest_price["date"]) + BDay(1)

    close_lower_return = close_return + float(np.quantile(residuals, 0.1))
    close_upper_return = close_return + float(np.quantile(residuals, 0.9))
    close_lower = last_close * (1 + min(close_lower_return, close_upper_return))
    close_upper = last_close * (1 + max(close_lower_return, close_upper_return))

    intraday = simulate_intraday_candles(
        symbol=symbol,
        forecast_date=forecast_date,
        predicted_open=predicted_open,
        predicted_high=predicted_high,
        predicted_low=predicted_low,
        predicted_close=predicted_close,
        daily_volatility=daily_volatility,
        random_seed=random_seed,
    )
    return NextSessionForecast(
        symbol=symbol,
        as_of_date=pd.Timestamp(latest_price["date"]),
        forecast_date=forecast_date,
        last_close=last_close,
        predicted_open=predicted_open,
        predicted_high=predicted_high,
        predicted_low=predicted_low,
        predicted_close=predicted_close,
        expected_return=close_return,
        close_lower=close_lower,
        close_upper=close_upper,
        daily_volatility=daily_volatility,
        open_return_mae=return_mae["next_open_return"],
        high_return_mae=return_mae["next_high_return"],
        low_return_mae=return_mae["next_low_return"],
        close_return_mae=close_return_mae,
        direction_accuracy=direction_accuracy,
        model_name=model_name,
        feature_count=len(feature_names),
        feature_kinds=feature_kinds,
        intraday=intraday,
        backtest=backtest,
        feature_importance=close_feature_importance,
    )
