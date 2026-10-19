"""The feature layer must never use information before it was available."""

from pathlib import Path

import numpy as np
import pandas as pd

from etf_mvp.config import AppConfig, DataConfig, ModelConfig, OutputConfig, UniverseConfig
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.service import collect, load_bundle
from etf_mvp.data.store import DataStore
from etf_mvp.features.asof import asof_values
from etf_mvp.features.central_bank import build_central_bank_features
from etf_mvp.features.news import build_news_features


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        universe=UniverseConfig(symbols=("SPY", "QQQ"), targets=("SPY",)),
        data=DataConfig(
            provider="synthetic",
            start="2021-01-01",
            end="2022-12-31",
            prices_path=tmp_path / "market" / "prices.parquet",
            features_path=tmp_path / "features.parquet",
            warehouse_path=tmp_path / "warehouse.duckdb",
        ),
        model=ModelConfig(
            horizon_days=5, test_fraction=0.2, probability_threshold=0.5, random_seed=1
        ),
        output=OutputConfig(artifacts_dir=tmp_path / "artifacts"),
    )


def test_asof_never_reads_ahead() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-10")
    events = pd.DataFrame(
        {"available_date": pd.to_datetime(["2024-01-03", "2024-01-08"]), "value": [1.0, 2.0]}
    )
    values = asof_values(dates, events, ["value"])["value"]
    assert values.loc["2024-01-02":"2024-01-02"].isna().all()
    assert (values.loc["2024-01-03":"2024-01-05"] == 1.0).all()
    assert (values.loc["2024-01-08":] == 2.0).all()


def test_fomc_decision_appears_only_from_meeting_day() -> None:
    dates = pd.bdate_range("2024-01-25", "2024-02-05")
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2023-12-13", "2024-01-31", "2024-03-20"]),
            "available_date": pd.to_datetime(["2023-12-13", "2024-01-31", "2024-03-20"]),
            "event_type": "fomc_meeting",
            "is_scheduled": True,
            "rate_lower": [5.25, 5.0, np.nan],
            "rate_upper": [5.5, 5.25, np.nan],
            "hawkish_score": [0.1, -0.3, np.nan],
            "statement_text": "",
            "source": "test",
        }
    )
    features = build_central_bank_features(events, dates)
    assert features.loc["2024-01-30", "cb__rate_upper"] == 5.5
    assert features.loc["2024-01-31", "cb__rate_upper"] == 5.25
    assert features.loc["2024-01-31", "cb__last_rate_change"] == -0.25
    assert features.loc["2024-01-31", "cb__is_meeting_day"] == 1
    assert features.loc["2024-01-29", "cb__days_to_next_meeting"] == 2


def test_news_before_coverage_is_unknown_not_zero() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    news = pd.DataFrame(
        {
            "article_id": ["a", "b"],
            "available_date": pd.to_datetime(["2024-01-15", "2024-01-16"]),
            "sentiment": [0.5, -0.5],
            "topics": ["economy_macro", "economy_macro"],
        }
    )
    features = build_news_features(news, dates, ["economy_macro"])
    assert features.loc[:"2024-01-12", "news__all__sentiment_5d"].isna().all()
    assert features.loc["2024-01-15", "news__all__sentiment_5d"] == 0.5
    assert features.loc["2024-01-16", "news__economy_macro__sentiment_5d"] == 0.0


def test_collect_writes_each_kind_separately(tmp_path: Path) -> None:
    config = _config(tmp_path)
    collect(config)
    store = DataStore(config.data.raw_root, config.data.prices_path, config.data.warehouse_path)
    for kind in DataKind:
        assert store.exists(kind, "synthetic"), kind
    bundle = load_bundle(config)
    assert set(bundle.frames) == set(DataKind)
    macro = bundle.get(DataKind.MACRO)
    assert (macro["available_date"] >= macro["observation_date"]).all()


def test_incremental_news_merges_instead_of_replacing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = collect(config, [DataKind.NEWS]).get(DataKind.NEWS)
    second = collect(config, [DataKind.NEWS]).get(DataKind.NEWS)
    assert len(second) == len(first)
    assert second["article_id"].is_unique


def test_provider_overrides() -> None:
    config = _config(Path("/tmp")).with_providers({"market": "tiingo", "macro": "fred"})
    assert config.data.provider == "tiingo"
    assert config.source(DataKind.MACRO).provider == "fred"
    assert config.source(DataKind.NEWS).provider == "synthetic"


def test_synthetic_alt_data_is_not_mixed_with_real_prices() -> None:
    from etf_mvp.data.service import enabled_kinds

    config = _config(Path("/tmp")).with_providers({"market": "tiingo", "macro": "fred"})
    assert enabled_kinds(config) == [DataKind.MARKET, DataKind.MACRO]
