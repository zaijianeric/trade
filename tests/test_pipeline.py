from pathlib import Path

from etf_mvp.config import AppConfig, DataConfig, ModelConfig, OutputConfig, UniverseConfig
from etf_mvp.pipeline import run_pipeline


def test_end_to_end_synthetic_pipeline(tmp_path: Path) -> None:
    config = AppConfig(
        universe=UniverseConfig(
            symbols=("SPY", "QQQ", "TLT"),
            targets=("SPY", "QQQ"),
        ),
        data=DataConfig(
            provider="synthetic",
            start="2018-01-01",
            end="2022-12-31",
            prices_path=tmp_path / "prices.parquet",
            features_path=tmp_path / "features.parquet",
            warehouse_path=tmp_path / "warehouse.duckdb",
        ),
        model=ModelConfig(
            horizon_days=5,
            test_fraction=0.2,
            probability_threshold=0.5,
            random_seed=42,
        ),
        output=OutputConfig(artifacts_dir=tmp_path / "artifacts"),
    )

    predictions = run_pipeline(config)

    assert set(predictions["symbol"]) == {"SPY", "QQQ"}
    assert predictions["probability_up"].between(0, 1).all()
    assert (tmp_path / "warehouse.duckdb").exists()
    assert (tmp_path / "artifacts" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "models" / "SPY.joblib").exists()


def test_pipeline_reports_importance_by_kind(tmp_path: Path) -> None:
    import pandas as pd

    config = AppConfig(
        universe=UniverseConfig(symbols=("SPY", "QQQ"), targets=("SPY",)),
        data=DataConfig(
            provider="synthetic",
            start="2018-01-01",
            end="2022-12-31",
            prices_path=tmp_path / "prices.parquet",
            features_path=tmp_path / "features.parquet",
            warehouse_path=tmp_path / "warehouse.duckdb",
        ),
        model=ModelConfig(
            horizon_days=5, test_fraction=0.2, probability_threshold=0.5, random_seed=42
        ),
        output=OutputConfig(artifacts_dir=tmp_path / "artifacts"),
    )
    run_pipeline(config)
    by_kind = pd.read_csv(tmp_path / "artifacts" / "importance_by_kind.csv")
    assert {
        "market",
        "macro",
        "fundamental",
        "news",
        "geopolitical",
        "central_bank",
    } <= set(by_kind["kind"])
    features = pd.read_parquet(tmp_path / "features.parquet")
    assert any(c.startswith("macro__") for c in features.columns)
    assert any(c.startswith("cb__") for c in features.columns)
