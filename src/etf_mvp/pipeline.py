from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from etf_mvp.config import AppConfig
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.service import DataBundle, collect, load_bundle, store_for
from etf_mvp.data.store import write_parquet
from etf_mvp.features import build_feature_table, feature_groups
from etf_mvp.modeling import save_model_result, train_symbol_model, write_json


def _with_market_provider(config: AppConfig, provider_name: str | None) -> AppConfig:
    if not provider_name:
        return config
    return replace(config, data=replace(config.data, provider=provider_name))


def fetch_data(
    config: AppConfig, kinds: list[DataKind] | None = None, provider_name: str | None = None
) -> DataBundle:
    """Collect and persist the given kinds (default: every enabled kind)."""
    return collect(_with_market_provider(config, provider_name), kinds)


def fetch_prices(config: AppConfig, provider_name: str | None = None) -> pd.DataFrame:
    return fetch_data(config, [DataKind.MARKET], provider_name).prices


def train_models(config: AppConfig, bundle: DataBundle | None = None) -> pd.DataFrame:
    bundle = bundle or load_bundle(config)
    market_prices = bundle.prices
    features = build_feature_table(bundle, config)
    write_parquet(features, config.data.features_path)
    store_for(config).refresh_warehouse(bundle.frames, features)

    predictions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    for symbol in config.universe.targets:
        result = train_symbol_model(
            features=features,
            prices=market_prices,
            symbol=symbol,
            horizon_days=config.model.horizon_days,
            test_fraction=config.model.test_fraction,
            threshold=config.model.probability_threshold,
            random_seed=config.model.random_seed,
        )
        save_model_result(result, config.output.artifacts_dir)
        predictions.append(
            {
                "prediction_date": result.prediction_date,
                "symbol": symbol,
                "horizon_days": config.model.horizon_days,
                "probability_up": result.probability_up,
                "predicted_direction": result.predicted_direction,
                "model": result.model.named_steps["classifier"].__class__.__name__,
            }
        )
        metrics.append(result.metrics)
        importance_frames.append(result.feature_importance)

    artifacts_dir = config.output.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions_frame = pd.DataFrame(predictions)
    predictions_frame.to_csv(artifacts_dir / "predictions.csv", index=False)

    importance = pd.concat(importance_frames, ignore_index=True)
    importance["kind"] = importance["feature"].map(feature_groups(list(importance["feature"])))
    importance.groupby("symbol", group_keys=False).head(25).to_csv(
        artifacts_dir / "feature_importance.csv", index=False
    )
    importance.groupby(["symbol", "kind"], as_index=False)["importance"].sum().to_csv(
        artifacts_dir / "importance_by_kind.csv", index=False
    )
    sanitized_metrics = [
        {
            key: (None if isinstance(value, float) and np.isnan(value) else value)
            for key, value in row.items()
        }
        for row in metrics
    ]
    write_json(sanitized_metrics, artifacts_dir / "metrics.json")
    return predictions_frame


def run_pipeline(config: AppConfig, provider_name: str | None = None) -> pd.DataFrame:
    config = _with_market_provider(config, provider_name)
    bundle = collect(config)
    return train_models(config, bundle)
