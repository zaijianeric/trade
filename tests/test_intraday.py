import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_mvp.intraday import (
    INTRADAY_SLOTS,
    IntradayPaths,
    _active_xnys_session,
    build_intraday_features,
    derive_intraday_ensemble_plan,
    parse_alpaca_bars,
    refresh_live_intraday_forecasts,
    roll_intraday_forecast,
    train_intraday_symbol,
    tune_intraday_symbol,
)


def _synthetic_intraday(symbol: str = "QQQ", sessions: int = 300) -> pd.DataFrame:
    random = np.random.default_rng(21)
    dates = pd.bdate_range("2022-01-03", periods=sessions)
    rows: list[dict[str, object]] = []
    previous_close = 300.0 if symbol == "QQQ" else 200.0
    for day_number, session_date in enumerate(dates):
        daily_return = random.normal(0.0004, 0.012)
        close_target = previous_close * (1 + daily_return)
        slots = np.arange(INTRADAY_SLOTS)
        profile = np.linspace(previous_close, close_target, INTRADAY_SLOTS)
        profile *= 1 + 0.002 * np.sin(2 * np.pi * slots / INTRADAY_SLOTS + day_number)
        profile *= np.exp(np.cumsum(random.normal(0, 0.00035, INTRADAY_SLOTS)))
        profile *= close_target / profile[-1]
        session_start = session_date + pd.Timedelta(hours=9, minutes=30)
        timestamps = pd.date_range(
            session_start.tz_localize("America/New_York"),
            periods=INTRADAY_SLOTS,
            freq="5min",
        ).tz_convert("UTC")
        opens = np.concatenate(([previous_close], profile[:-1]))
        for slot, (timestamp, open_price, close_price) in enumerate(
            zip(timestamps, opens, profile, strict=True)
        ):
            wick = abs(random.normal(0.0003, 0.0001))
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "session_date": session_date,
                    "slot": slot,
                    "open": open_price,
                    "high": max(open_price, close_price) * (1 + wick),
                    "low": min(open_price, close_price) * (1 - wick),
                    "close": close_price,
                    "volume": int(random.integers(20_000, 200_000)),
                    "trade_count": int(random.integers(100, 2_000)),
                    "vwap": (open_price + close_price) / 2,
                }
            )
        previous_close = close_target
    return pd.DataFrame(rows)


def test_parse_alpaca_bars_filters_extended_hours() -> None:
    rows = [
        {
            "t": "2026-10-20T13:30:00Z",
            "o": 100,
            "h": 101,
            "l": 99,
            "c": 100.5,
            "v": 1000,
            "n": 20,
            "vw": 100.2,
        },
        {
            "t": "2026-10-20T12:00:00Z",
            "o": 99,
            "h": 100,
            "l": 98,
            "c": 99.5,
            "v": 500,
            "n": 10,
            "vw": 99.2,
        },
    ]

    frame = parse_alpaca_bars("QQQ", rows)

    assert len(frame) == 1
    assert frame.iloc[0]["slot"] == 0
    assert frame.iloc[0]["symbol"] == "QQQ"


def test_intraday_pipeline_is_restricted_to_qqq_and_soxx() -> None:
    with pytest.raises(ValueError, match="QQQ, SOXX"):
        parse_alpaca_bars("SPY", [])


def test_active_xnys_session_only_returns_regular_market_hours() -> None:
    assert _active_xnys_session(pd.Timestamp("2026-10-26 14:30:00Z")) == pd.Timestamp(
        "2026-10-26"
    )
    assert _active_xnys_session(pd.Timestamp("2026-10-26 13:29:00Z")) is None
    assert _active_xnys_session(pd.Timestamp("2026-10-24 15:00:00Z")) is None


def test_live_refresh_persists_rolling_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = IntradayPaths(tmp_path / "data", tmp_path / "artifacts")
    session_date = pd.Timestamp("2026-10-26")
    slots = np.arange(INTRADAY_SLOTS)
    timestamps = pd.date_range("2026-10-26 13:30:00Z", periods=INTRADAY_SLOTS, freq="5min")
    for symbol in ("QQQ", "SOXX"):
        artifact_dir = paths.artifact_dir(symbol)
        artifact_dir.mkdir(parents=True)
        forecast = pd.DataFrame(
            {
                "timestamp": timestamps,
                "session_date": session_date,
                "slot": slots,
                "predicted_return_q10": np.linspace(-0.01, -0.005, INTRADAY_SLOTS),
                "predicted_return": np.linspace(0, 0.01, INTRADAY_SLOTS),
                "predicted_return_q90": np.linspace(0.01, 0.02, INTRADAY_SLOTS),
            }
        )
        forecast.to_csv(artifact_dir / "forecast.csv", index=False)
        (artifact_dir / "metrics.json").write_text(
            json.dumps({"forecast_date": "2026-10-26", "last_close": 100.0}),
            encoding="utf-8",
        )

    def fake_fetch(symbol: str, start: str, **_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": timestamps[:2],
                "session_date": session_date,
                "slot": [0, 1],
                "open": [100.0, 100.2],
                "high": [100.3, 100.5],
                "low": [99.9, 100.1],
                "close": [100.2, 100.4],
                "volume": [100, 100],
                "trade_count": [10, 10],
                "vwap": [100.1, 100.3],
            }
        )

    monkeypatch.setattr("etf_mvp.intraday.fetch_alpaca_bars", fake_fetch)
    results = refresh_live_intraday_forecasts(
        paths, now=pd.Timestamp("2026-10-26 14:00:00Z")
    )

    assert results["QQQ"]["status"] == "updated"
    assert results["QQQ"]["cutoff_slot"] == 1
    live = pd.read_csv(paths.live_forecast("QQQ"))
    assert live.loc[live["slot"] <= 1, "observed"].all()
    assert paths.live_status("SOXX").exists()


@pytest.mark.parametrize("symbol", ["QQQ", "SOXX"])
def test_intraday_features_and_walk_forward_backtest(symbol: str) -> None:
    bars = _synthetic_intraday(symbol)
    features, forecast_date, latest_close = build_intraday_features(bars)
    baseline_features, _, _ = build_intraday_features(bars, enhanced=False)

    assert len(features.loc[features["session_date"] == forecast_date]) == INTRADAY_SLOTS
    assert latest_close > 0
    assert "prior_open_gap" in features
    assert "prior_close_vs_vwap" in features
    assert features.shape[1] > baseline_features.shape[1]

    result = train_intraday_symbol(bars, symbol, random_seed=9)

    assert result.symbol == symbol
    assert len(result.forecast) == INTRADAY_SLOTS
    assert result.feature_count >= 15
    assert result.metrics["backtest_sessions"] >= 40
    assert 0 <= result.metrics["direction_accuracy"] <= 1
    assert 0 <= result.metrics["interval_80_coverage"] <= 1
    assert result.metrics["target_design"] == "normalized_shape+close_level+session_range"
    assert (
        result.forecast["predicted_return_q10"]
        <= result.forecast["predicted_return"]
    ).all()
    assert (
        result.forecast["predicted_return"]
        <= result.forecast["predicted_return_q90"]
    ).all()
    assert result.forecast["predicted_price"].max() > result.forecast[
        "predicted_price"
    ].min()
    assert (
        result.backtest_sessions["train_end_date"]
        < result.backtest_sessions["session_date"]
    ).all()
    assert set(result.backtest_sessions["fold"]) == {1, 2, 3, 4, 5}


def test_intraday_rolling_tuning_uses_prior_sessions() -> None:
    result = tune_intraday_symbol(
        _synthetic_intraday(sessions=300),
        "QQQ",
        random_seed=9,
        trials=2,
        folds=2,
    )

    assert result.folds == 2
    assert len(result.trials) == 2
    assert result.inner_start < result.inner_end
    assert result.selected_params["n_estimators"] > 0


def test_intraday_ensemble_weights_use_prior_sessions() -> None:
    result = derive_intraday_ensemble_plan(
        _synthetic_intraday(sessions=300),
        "QQQ",
        random_seed=9,
        folds=2,
    )

    assert result.folds == 2
    assert len(result.component_metrics) == 3
    assert sum(result.weights.values()) == pytest.approx(1)
    assert max(result.weights.values()) <= 0.6


def test_rolling_forecast_uses_only_observed_prefix() -> None:
    result = train_intraday_symbol(_synthetic_intraday(sessions=300), "SOXX", random_seed=9)
    forecast = result.forecast.copy()
    observed = pd.DataFrame(
        {
            "slot": np.arange(12),
            "close": np.linspace(
                float(result.metrics["last_close"]) * 1.002,
                float(result.metrics["last_close"]) * 1.009,
                12,
            ),
        }
    )

    rolled = roll_intraday_forecast(
        forecast,
        observed,
        float(result.metrics["last_close"]),
    )

    assert rolled.loc[rolled["slot"] <= 11, "observed"].all()
    assert not rolled.loc[rolled["slot"] > 11, "observed"].any()
    assert rolled.loc[rolled["slot"] == 11, "predicted_price"].iloc[0] == pytest.approx(
        observed["close"].iloc[-1]
    )
    assert rolled.loc[rolled["slot"] > 11, "predicted_price"].notna().all()
    assert (rolled["predicted_return_q10"] <= rolled["predicted_return"]).all()
    assert (rolled["predicted_return"] <= rolled["predicted_return_q90"]).all()
