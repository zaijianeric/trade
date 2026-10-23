import pandas as pd

from etf_mvp.data.base import FetchRequest
from etf_mvp.data.sources.market import SyntheticMarketSource
from etf_mvp.forecast import build_next_session_targets, forecast_next_session


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
    assert forecast.intraday["close"].max() < forecast.predicted_high
    assert forecast.intraday["close"].min() > forecast.predicted_low
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
    assert forecast.feature_count == 45
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
