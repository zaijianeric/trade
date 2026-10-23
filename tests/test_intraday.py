import numpy as np
import pandas as pd
import pytest

from etf_mvp.intraday import (
    INTRADAY_SLOTS,
    build_intraday_features,
    parse_alpaca_bars,
    train_intraday_symbol,
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


@pytest.mark.parametrize("symbol", ["QQQ", "SOXX"])
def test_intraday_features_and_walk_forward_backtest(symbol: str) -> None:
    bars = _synthetic_intraday(symbol)
    features, forecast_date, latest_close = build_intraday_features(bars)

    assert len(features.loc[features["session_date"] == forecast_date]) == INTRADAY_SLOTS
    assert latest_close > 0

    result = train_intraday_symbol(bars, symbol, random_seed=9)

    assert result.symbol == symbol
    assert len(result.forecast) == INTRADAY_SLOTS
    assert result.feature_count >= 15
    assert result.metrics["backtest_sessions"] >= 40
    assert 0 <= result.metrics["direction_accuracy"] <= 1
    assert (
        result.backtest_sessions["train_end_date"]
        < result.backtest_sessions["session_date"]
    ).all()
    assert set(result.backtest_sessions["fold"]) == {1, 2, 3, 4, 5}
