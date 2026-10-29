import json
from pathlib import Path

import pandas as pd

from etf_mvp import pipeline
from etf_mvp.config import AppConfig, DataConfig, ModelConfig, OutputConfig, UniverseConfig
from etf_mvp.pipeline import DAILY_FORECAST_HORIZONS, run_pipeline, train_models


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
    assert (tmp_path / "artifacts" / "rolling_tuning" / "daily.json").exists()
    assert (tmp_path / "artifacts" / "feature_ablation" / "news.json").exists()
    assert (tmp_path / "artifacts" / "data_quality" / "news.json").exists()


def test_pipeline_reports_importance_by_kind(tmp_path: Path) -> None:
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


def test_multi_horizon_training_combines_isolated_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    config = AppConfig(
        universe=UniverseConfig(symbols=("SPY",), targets=("SPY",)),
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

    def fake_train(single_config: AppConfig, bundle: object) -> pd.DataFrame:
        del bundle
        horizon = single_config.model.horizon_days
        artifacts_dir = single_config.output.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "metrics.json").write_text(
            json.dumps([{"symbol": "SPY", "horizon_days": horizon}]),
            encoding="utf-8",
        )
        pd.DataFrame(
            [{"symbol": "SPY", "feature": "market__return", "importance": 1.0, "kind": "market"}]
        ).to_csv(artifacts_dir / "feature_importance.csv", index=False)
        pd.DataFrame(
            [{"symbol": "SPY", "kind": "market", "importance": 1.0}]
        ).to_csv(artifacts_dir / "importance_by_kind.csv", index=False)
        return pd.DataFrame(
            [
                {
                    "prediction_date": "2026-10-29",
                    "symbol": "SPY",
                    "horizon_days": horizon,
                    "probability_up": 0.5,
                    "predicted_direction": "UP",
                    "model": "TestModel",
                }
            ]
        )

    monkeypatch.setattr(pipeline, "_train_models_single", fake_train)
    predictions = train_models(config, bundle=object(), horizons=DAILY_FORECAST_HORIZONS)

    assert set(predictions["horizon_days"]) == set(DAILY_FORECAST_HORIZONS)
    combined_importance = pd.read_csv(config.output.artifacts_dir / "feature_importance.csv")
    assert set(combined_importance["horizon_days"]) == set(DAILY_FORECAST_HORIZONS)
