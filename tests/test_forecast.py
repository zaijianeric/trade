import pandas as pd

from etf_mvp.data.base import FetchRequest
from etf_mvp.data.sources.market import SyntheticMarketSource
from etf_mvp.forecast import (
    build_next_session_targets,
    compare_backtests_on_common_dates,
    forecast_next_session,
)


def _synthetic_prices() -> pd.DataFrame:
    source = SyntheticMarketSource(random_seed=11)
    return source.fetch(
        FetchRequest(
            symbols=("SPY", "QQQ", "TLT"),
            start="2018-01-01",
            end="2022-12-31",
        )
    )


def test_next_session_targets_use_following_row() -> None:
    prices = _synthetic_prices()
    targets = build_next_session_targets(prices, "SPY")
    spy = prices.loc[prices["symbol"] == "SPY"].sort_values("date").reset_index(drop=True)

    expected_close_return = spy.loc[1, "close"] / spy.loc[0, "close"] - 1

    assert targets.loc[0, "next_close_return"] == expected_close_return
    assert pd.isna(targets.iloc[-1]["next_close_return"])


def test_forecast_has_consistent_ohlc_and_intraday_path() -> None:
    forecast = forecast_next_session(_synthetic_prices(), "SPY", random_seed=17)

    assert forecast.forecast_date > forecast.as_of_date
    assert forecast.predicted_low <= min(forecast.predicted_open, forecast.predicted_close)
    assert forecast.predicted_high >= max(forecast.predicted_open, forecast.predicted_close)
    assert forecast.close_lower <= forecast.close_upper
    assert 0 <= forecast.direction_accuracy <= 1
    assert len(forecast.intraday) == 79
    assert forecast.intraday.iloc[0]["timestamp"].strftime("%H:%M") == "09:30"
    assert forecast.intraday.iloc[-1]["timestamp"].strftime("%H:%M") == "16:00"
    assert forecast.intraday.iloc[0]["close"] == forecast.predicted_open
    assert forecast.intraday.iloc[-1]["close"] == forecast.predicted_close
    assert forecast.intraday["high"].max() == forecast.predicted_high
    assert forecast.intraday["low"].min() == forecast.predicted_low
    assert forecast.intraday["close"].max() == forecast.predicted_high
    assert forecast.intraday["close"].min() == forecast.predicted_low
    assert not forecast.backtest.empty
    assert (
        forecast.backtest["train_end_date"] < forecast.backtest["date"]
    ).all()
    assert set(forecast.backtest["fold"]) == {1, 2, 3, 4, 5}
    assert forecast.backtest["direction_correct"].dtype == bool
    assert set(
        ["feature", "importance", "local_contribution", "local_share"]
    ).issubset(forecast.feature_importance.columns)
    assert forecast.feature_importance["local_share"].between(0, 1).all()
    assert forecast.feature_count == 79
    assert forecast.feature_kinds == ("market",)


def test_forecast_accepts_point_in_time_feature_table() -> None:
    prices = _synthetic_prices()
    dates = sorted(prices["date"].unique())
    feature_table = pd.DataFrame(
        {
            "date": dates,
            "SPY__return_1d": range(len(dates)),
            "news__economic_sentiment__level": 0.1,
            "geo__gpr_level": 100.0,
        }
    )

    forecast = forecast_next_session(
        prices,
        "SPY",
        random_seed=17,
        feature_table=feature_table,
    )

    assert forecast.feature_count == 3
    assert forecast.feature_kinds == ("geopolitical", "market", "news")


def test_backtest_comparison_uses_only_common_dates() -> None:
    daily = pd.DataFrame(
        {
            "forecast_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "next_open_return": [0.01, 0.02],
            "next_high_return": [0.03, 0.04],
            "next_low_return": [-0.01, 0.0],
            "next_close_return": [0.02, -0.01],
            "predicted_next_open_return": [0.0, 0.01],
            "predicted_next_high_return": [0.02, 0.03],
            "predicted_next_low_return": [-0.02, -0.01],
            "predicted_next_close_return": [0.01, 0.02],
        }
    )
    minute = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "previous_close": [100.0, 100.0],
            "actual_open": [102.0, 101.0],
            "predicted_open": [101.0, 101.0],
            "actual_high": [104.0, 102.0],
            "predicted_high": [103.0, 102.0],
            "actual_low": [100.0, 99.0],
            "predicted_low": [99.0, 99.0],
            "actual_close": [99.0, 101.0],
            "predicted_close": [102.0, 101.0],
            "direction_correct": [False, True],
        }
    )

    comparison = compare_backtests_on_common_dates(daily, minute)

    assert comparison["sessions"].tolist() == [1, 1]
    assert comparison["start"].eq(pd.Timestamp("2026-01-05")).all()
    assert comparison.loc[0, "direction_accuracy"] == 0.0
    assert comparison.loc[1, "close_mae"] == 0.03
