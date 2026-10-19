from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from etf_mvp.features import build_target


@dataclass(frozen=True)
class TrainedModel:
    symbol: str
    prediction_date: str
    probability_up: float
    predicted_direction: str
    metrics: dict[str, float | int | str]
    model: Pipeline
    feature_importance: pd.DataFrame


def _classifier(random_seed: int):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        )
    except (ImportError, OSError):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=1,
        )


def _safe_auc(y_true: pd.Series, probability: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probability)) if y_true.nunique() > 1 else float("nan")


def train_symbol_model(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    symbol: str,
    horizon_days: int,
    test_fraction: float,
    threshold: float,
    random_seed: int,
) -> TrainedModel:
    target = build_target(prices, symbol, horizon_days)
    dataset = features.merge(target, on="date", how="left").sort_values("date")
    feature_names = [column for column in features.columns if column != "date"]
    labeled = dataset.dropna(subset=["target"]).copy()
    if len(labeled) < 260:
        raise ValueError(f"Not enough labeled rows for {symbol}: {len(labeled)}")

    split_index = int(len(labeled) * (1 - test_fraction))
    split_index = min(max(split_index, 200), len(labeled) - 40)
    train = labeled.iloc[: split_index - horizon_days]
    test = labeled.iloc[split_index:]
    x_train = train[feature_names]
    y_train = train["target"].astype(int)
    x_test = test[feature_names]
    y_test = test["target"].astype(int)

    pipeline = Pipeline(
        [
            (
                "imputer",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            # Keep all-NaN columns (e.g. news before coverage) so that
                            # feature names and importances stay aligned.
                            SimpleImputer(strategy="median", keep_empty_features=True),
                            feature_names,
                        )
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("classifier", _classifier(random_seed)),
        ]
    )
    pipeline.fit(x_train, y_train)
    test_probability = pipeline.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= threshold).astype(int)

    metrics: dict[str, float | int | str] = {
        "symbol": symbol,
        "horizon_days": horizon_days,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": pd.Timestamp(test["date"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["date"].max()).date().isoformat(),
        "accuracy": float(accuracy_score(y_test, test_prediction)),
        "roc_auc": _safe_auc(y_test, test_probability),
        "brier_score": float(brier_score_loss(y_test, test_probability)),
        "log_loss": float(log_loss(y_test, test_probability, labels=[0, 1])),
        "positive_rate": float(y_test.mean()),
    }

    full_x = labeled[feature_names]
    full_y = labeled["target"].astype(int)
    pipeline.fit(full_x, full_y)
    latest = dataset.iloc[-1]
    latest_x = dataset.loc[[dataset.index[-1]], feature_names]
    probability_up = float(pipeline.predict_proba(latest_x)[:, 1][0])

    classifier = pipeline.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", np.zeros(len(feature_names)))
    feature_importance = pd.DataFrame(
        {"symbol": symbol, "feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    return TrainedModel(
        symbol=symbol,
        prediction_date=pd.Timestamp(latest["date"]).date().isoformat(),
        probability_up=probability_up,
        predicted_direction="UP" if probability_up >= threshold else "DOWN",
        metrics=metrics,
        model=pipeline,
        feature_importance=feature_importance,
    )


def save_model_result(result: TrainedModel, artifacts_dir: Path) -> None:
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_dir / f"{result.symbol}.joblib")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2, allow_nan=False)
