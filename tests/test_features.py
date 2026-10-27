from etf_mvp.data.base import FetchRequest
from etf_mvp.data.sources.market import SyntheticMarketSource
from etf_mvp.features import build_features, build_target


def test_features_and_forward_target_are_aligned() -> None:
    prices = SyntheticMarketSource(random_seed=7).fetch(
        FetchRequest(start="2020-01-01", end="2022-12-31", symbols=("SPY", "QQQ"))
    )
    features = build_features(prices, ["SPY", "QQQ"])
    target = build_target(prices, "SPY", horizon_days=5)

    assert features["date"].is_monotonic_increasing
    assert "SPY__return_5d" in features.columns
    assert "SPY__overnight_gap" in features.columns
    assert "QQQ__relative_spy_20d" in features.columns
    assert "market__breadth_positive_1d" in features.columns
    assert target["target"].tail(5).isna().all()

    spy = prices.loc[prices["symbol"] == "SPY"].sort_values("date").reset_index(drop=True)
    expected = spy.loc[5, "close"] / spy.loc[0, "close"] - 1
    assert target.loc[0, "forward_return"] == expected
