from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from etf_mvp.config import AppConfig
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.service import DataBundle, collect, load_bundle, store_for
from etf_mvp.data.store import write_parquet
from etf_mvp.features import build_feature_table, feature_groups
from etf_mvp.modeling import (
    DAILY_DEFAULT_PARAMS,
    derive_daily_ensemble_plan,
    save_model_result,
    train_symbol_model,
    tune_symbol_model,
    write_json,
)
from etf_mvp.news_quality import summarize_news_quality


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
    baseline_features = build_feature_table(bundle, config, enhanced_price=False)
    features = build_feature_table(bundle, config, enhanced_price=True)
    write_parquet(features, config.data.features_path)
    store_for(config).refresh_warehouse(bundle.frames, features)

    predictions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    news_ablation_rows: list[dict[str, object]] = []
    for symbol in config.universe.targets:
        baseline_result = train_symbol_model(
            features=baseline_features,
            prices=market_prices,
            symbol=symbol,
            horizon_days=config.model.horizon_days,
            test_fraction=config.model.test_fraction,
            threshold=config.model.probability_threshold,
            random_seed=config.model.random_seed,
        )
        enhanced_result = train_symbol_model(
            features=features,
            prices=market_prices,
            symbol=symbol,
            horizon_days=config.model.horizon_days,
            test_fraction=config.model.test_fraction,
            threshold=config.model.probability_threshold,
            random_seed=config.model.random_seed,
        )
        accuracy_delta = float(enhanced_result.metrics["accuracy"]) - float(
            baseline_result.metrics["accuracy"]
        )
        auc_delta = float(enhanced_result.metrics["roc_auc"]) - float(
            baseline_result.metrics["roc_auc"]
        )
        brier_delta = float(enhanced_result.metrics["brier_score"]) - float(
            baseline_result.metrics["brier_score"]
        )
        accepted = (
            accuracy_delta >= 0.015
            or auc_delta >= 0.015
            or brier_delta <= -0.005
        )
        selected_feature_set = "enhanced" if accepted else "baseline"
        selected_features = features if accepted else baseline_features
        fixed_result = enhanced_result if accepted else baseline_result
        tuning = tune_symbol_model(
            features=selected_features,
            prices=market_prices,
            symbol=symbol,
            horizon_days=config.model.horizon_days,
            test_fraction=config.model.test_fraction,
            threshold=config.model.probability_threshold,
            random_seed=config.model.random_seed,
            trials=config.model.tuning_trials,
            folds=config.model.tuning_folds,
        )
        tuned_result = train_symbol_model(
            features=selected_features,
            prices=market_prices,
            symbol=symbol,
            horizon_days=config.model.horizon_days,
            test_fraction=config.model.test_fraction,
            threshold=tuning.selected_threshold,
            random_seed=config.model.random_seed,
            model_params=tuning.selected_params,
        )
        tuning_accuracy_delta = float(tuned_result.metrics["accuracy"]) - float(
            fixed_result.metrics["accuracy"]
        )
        tuning_auc_delta = float(tuned_result.metrics["roc_auc"]) - float(
            fixed_result.metrics["roc_auc"]
        )
        tuning_brier_delta = float(tuned_result.metrics["brier_score"]) - float(
            fixed_result.metrics["brier_score"]
        )
        tuning_accepted = (
            tuning_accuracy_delta >= 0.01
            or tuning_auc_delta >= 0.01
            or tuning_brier_delta <= -0.003
        ) and tuning_accuracy_delta >= -0.01
        single_result = tuned_result if tuning_accepted else fixed_result
        selected_params = tuning.selected_params if tuning_accepted else DAILY_DEFAULT_PARAMS
        selected_threshold = (
            tuning.selected_threshold
            if tuning_accepted
            else config.model.probability_threshold
        )
        ensemble_plan = derive_daily_ensemble_plan(
            features=selected_features,
            prices=market_prices,
            symbol=symbol,
            horizon_days=config.model.horizon_days,
            test_fraction=config.model.test_fraction,
            random_seed=config.model.random_seed,
            model_params=selected_params,
            folds=config.model.tuning_folds,
        )
        ensemble_result = train_symbol_model(
            features=selected_features,
            prices=market_prices,
            symbol=symbol,
            horizon_days=config.model.horizon_days,
            test_fraction=config.model.test_fraction,
            threshold=ensemble_plan.threshold,
            random_seed=config.model.random_seed,
            model_params=selected_params,
            ensemble_weights=ensemble_plan.weights,
        )
        ensemble_accuracy_delta = float(ensemble_result.metrics["accuracy"]) - float(
            single_result.metrics["accuracy"]
        )
        ensemble_auc_delta = float(ensemble_result.metrics["roc_auc"]) - float(
            single_result.metrics["roc_auc"]
        )
        ensemble_brier_delta = float(ensemble_result.metrics["brier_score"]) - float(
            single_result.metrics["brier_score"]
        )
        ensemble_accepted = (
            ensemble_accuracy_delta >= 0.01
            or ensemble_auc_delta >= 0.01
            or ensemble_brier_delta <= -0.003
        ) and ensemble_accuracy_delta >= -0.01
        result = ensemble_result if ensemble_accepted else single_result
        news_columns = [
            column for column in selected_features.columns if column.startswith("news__")
        ]
        if news_columns:
            features_without_news = selected_features.drop(columns=news_columns)
            no_news_result = train_symbol_model(
                features=features_without_news,
                prices=market_prices,
                symbol=symbol,
                horizon_days=config.model.horizon_days,
                test_fraction=config.model.test_fraction,
                threshold=(
                    ensemble_plan.threshold if ensemble_accepted else selected_threshold
                ),
                random_seed=config.model.random_seed,
                model_params=selected_params,
                ensemble_weights=(
                    ensemble_plan.weights if ensemble_accepted else None
                ),
            )
            news_accuracy_delta = float(result.metrics["accuracy"]) - float(
                no_news_result.metrics["accuracy"]
            )
            news_auc_delta = float(result.metrics["roc_auc"]) - float(
                no_news_result.metrics["roc_auc"]
            )
            news_brier_delta = float(result.metrics["brier_score"]) - float(
                no_news_result.metrics["brier_score"]
            )
            news_accepted = (
                news_accuracy_delta >= 0.01
                or news_auc_delta >= 0.01
                or news_brier_delta <= -0.003
            ) and news_accuracy_delta >= -0.01
            if not news_accepted:
                result = no_news_result
            news_ablation_rows.append(
                {
                    "symbol": symbol,
                    "news_feature_count": len(news_columns),
                    "without_news_accuracy": no_news_result.metrics["accuracy"],
                    "with_news_accuracy": (
                        ensemble_result.metrics["accuracy"]
                        if ensemble_accepted
                        else single_result.metrics["accuracy"]
                    ),
                    "accuracy_delta": news_accuracy_delta,
                    "without_news_roc_auc": no_news_result.metrics["roc_auc"],
                    "with_news_roc_auc": (
                        ensemble_result.metrics["roc_auc"]
                        if ensemble_accepted
                        else single_result.metrics["roc_auc"]
                    ),
                    "roc_auc_delta": news_auc_delta,
                    "without_news_brier": no_news_result.metrics["brier_score"],
                    "with_news_brier": (
                        ensemble_result.metrics["brier_score"]
                        if ensemble_accepted
                        else single_result.metrics["brier_score"]
                    ),
                    "brier_delta": news_brier_delta,
                    "accepted": news_accepted,
                    "selected_news_features": "included" if news_accepted else "excluded",
                }
            )
            result.metrics["news_features"] = (
                "included" if news_accepted else "excluded"
            )
        else:
            result.metrics["news_features"] = "unavailable"
        result.metrics["feature_set"] = selected_feature_set
        result.metrics["parameter_set"] = "tuned" if tuning_accepted else "fixed"
        result.metrics["model_strategy"] = "ensemble" if ensemble_accepted else "single"
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
        ablation_rows.append(
            {
                "symbol": symbol,
                "test_start": enhanced_result.metrics["test_start"],
                "test_end": enhanced_result.metrics["test_end"],
                "test_rows": enhanced_result.metrics["test_rows"],
                "baseline_feature_count": len(baseline_result.feature_importance),
                "enhanced_feature_count": len(enhanced_result.feature_importance),
                "baseline_accuracy": baseline_result.metrics["accuracy"],
                "enhanced_accuracy": enhanced_result.metrics["accuracy"],
                "accuracy_delta": accuracy_delta,
                "baseline_roc_auc": baseline_result.metrics["roc_auc"],
                "enhanced_roc_auc": enhanced_result.metrics["roc_auc"],
                "roc_auc_delta": auc_delta,
                "baseline_brier": baseline_result.metrics["brier_score"],
                "enhanced_brier": enhanced_result.metrics["brier_score"],
                "brier_delta": brier_delta,
                "accepted": accepted,
                "selected_feature_set": selected_feature_set,
            }
        )
        tuning_rows.append(
            {
                "symbol": symbol,
                "feature_set": selected_feature_set,
                "inner_start": tuning.inner_start,
                "inner_end": tuning.inner_end,
                "folds": tuning.folds,
                "trial_count": len(tuning.trials),
                "fixed_accuracy": fixed_result.metrics["accuracy"],
                "tuned_accuracy": tuned_result.metrics["accuracy"],
                "accuracy_delta": tuning_accuracy_delta,
                "fixed_roc_auc": fixed_result.metrics["roc_auc"],
                "tuned_roc_auc": tuned_result.metrics["roc_auc"],
                "roc_auc_delta": tuning_auc_delta,
                "fixed_brier": fixed_result.metrics["brier_score"],
                "tuned_brier": tuned_result.metrics["brier_score"],
                "brier_delta": tuning_brier_delta,
                "accepted": tuning_accepted,
                "selected_parameter_set": "tuned" if tuning_accepted else "fixed",
                "selected_threshold": tuning.selected_threshold,
                "selected_params": tuning.selected_params,
                "trials": tuning.trials,
            }
        )
        ensemble_rows.append(
            {
                "symbol": symbol,
                "feature_set": selected_feature_set,
                "parameter_set": "tuned" if tuning_accepted else "fixed",
                "inner_start": ensemble_plan.inner_start,
                "inner_end": ensemble_plan.inner_end,
                "folds": ensemble_plan.folds,
                "single_accuracy": single_result.metrics["accuracy"],
                "ensemble_accuracy": ensemble_result.metrics["accuracy"],
                "accuracy_delta": ensemble_accuracy_delta,
                "single_roc_auc": single_result.metrics["roc_auc"],
                "ensemble_roc_auc": ensemble_result.metrics["roc_auc"],
                "roc_auc_delta": ensemble_auc_delta,
                "single_brier": single_result.metrics["brier_score"],
                "ensemble_brier": ensemble_result.metrics["brier_score"],
                "brier_delta": ensemble_brier_delta,
                "accepted": ensemble_accepted,
                "selected_strategy": "ensemble" if ensemble_accepted else "single",
                "single_threshold": selected_threshold,
                "ensemble_threshold": ensemble_plan.threshold,
                "weights": ensemble_plan.weights,
                "components": ensemble_plan.component_metrics,
            }
        )

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
    write_json(ablation_rows, artifacts_dir / "feature_ablation" / "daily.json")
    write_json(tuning_rows, artifacts_dir / "rolling_tuning" / "daily.json")
    write_json(ensemble_rows, artifacts_dir / "ensemble" / "daily.json")
    write_json(news_ablation_rows, artifacts_dir / "feature_ablation" / "news.json")
    if DataKind.NEWS in bundle.frames:
        write_json(
            summarize_news_quality(bundle.get(DataKind.NEWS), market_prices),
            artifacts_dir / "data_quality" / "news.json",
        )
    return predictions_frame


def run_pipeline(config: AppConfig, provider_name: str | None = None) -> pd.DataFrame:
    config = _with_market_provider(config, provider_name)
    bundle = collect(config)
    return train_models(config, bundle)
