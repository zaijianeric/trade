from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


@dataclass(frozen=True)
class DailyTuningResult:
    symbol: str
    selected_params: dict[str, float | int]
    selected_threshold: float
    inner_start: str
    inner_end: str
    folds: int
    trials: list[dict[str, object]]


@dataclass(frozen=True)
class DailyEnsemblePlan:
    symbol: str
    weights: dict[str, float]
    threshold: float
    inner_start: str
    inner_end: str
    folds: int
    component_metrics: list[dict[str, object]]


DAILY_DEFAULT_PARAMS: dict[str, float | int] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 0,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
}

DAILY_ENSEMBLE_MODELS = ("lightgbm", "extra_trees", "logistic")


def _classifier(
    random_seed: int, model_params: dict[str, float | int] | None = None
):
    params = {**DAILY_DEFAULT_PARAMS, **(model_params or {})}
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            **params,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        )
    except (ImportError, OSError):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=(
                None if int(params["max_depth"]) < 0 else int(params["max_depth"])
            ),
            min_samples_leaf=max(2, int(params["min_child_samples"]) // 4),
            max_features=float(params["colsample_bytree"]),
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=1,
        )


def _daily_base_classifier(
    name: str,
    random_seed: int,
    model_params: dict[str, float | int] | None = None,
):
    if name == "lightgbm":
        return _classifier(random_seed, model_params)
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=320,
            max_depth=10,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=1,
        )
    if name == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.35,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown daily ensemble model: {name}")


class CalibratedWeightedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model_params: dict[str, float | int] | None = None,
        weights: dict[str, float] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model_params = model_params
        self.weights = weights
        self.random_seed = random_seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> CalibratedWeightedClassifier:
        values = np.asarray(x)
        target = np.asarray(y).astype(int)
        calibration_start = min(max(int(len(target) * 0.8), 100), len(target) - 40)
        train_x = values[:calibration_start]
        train_y = target[:calibration_start]
        calibration_x = values[calibration_start:]
        calibration_y = target[calibration_start:]
        self.estimators_: dict[str, object] = {}
        self.calibrators_: dict[str, LogisticRegression | None] = {}
        importance = np.zeros(values.shape[1], dtype=float)
        weights = self.weights or {name: 1 / 3 for name in DAILY_ENSEMBLE_MODELS}
        for name in DAILY_ENSEMBLE_MODELS:
            calibration_model = _daily_base_classifier(
                name, self.random_seed, self.model_params
            )
            calibration_model.fit(train_x, train_y)
            raw_probability = calibration_model.predict_proba(calibration_x)[:, 1]
            calibrator: LogisticRegression | None = None
            if np.unique(calibration_y).size > 1:
                calibrator = LogisticRegression(max_iter=500, random_state=self.random_seed)
                calibrator.fit(raw_probability.reshape(-1, 1), calibration_y)
            estimator = _daily_base_classifier(name, self.random_seed, self.model_params)
            estimator.fit(values, target)
            self.estimators_[name] = estimator
            self.calibrators_[name] = calibrator
            raw_importance = getattr(estimator, "feature_importances_", None)
            if name == "logistic":
                raw_importance = np.abs(estimator.named_steps["classifier"].coef_[0])
            if raw_importance is None and hasattr(estimator, "coef_"):
                raw_importance = np.abs(estimator.coef_[0])
            if raw_importance is not None:
                normalized = np.asarray(raw_importance, dtype=float)
                normalized /= normalized.sum() or 1
                importance += float(weights[name]) * normalized
        self.feature_importances_ = importance
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = values.shape[1]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x)
        weights = self.weights or {name: 1 / 3 for name in DAILY_ENSEMBLE_MODELS}
        combined = np.zeros(len(values), dtype=float)
        for name, estimator in self.estimators_.items():
            probability = estimator.predict_proba(values)[:, 1]
            calibrator = self.calibrators_[name]
            if calibrator is not None:
                probability = calibrator.predict_proba(probability.reshape(-1, 1))[:, 1]
            combined += float(weights[name]) * probability
        return np.column_stack([1 - combined, combined])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def _safe_auc(y_true: pd.Series, probability: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probability)) if y_true.nunique() > 1 else float("nan")


def _model_pipeline(
    feature_names: list[str],
    random_seed: int,
    model_params: dict[str, float | int] | None = None,
    classifier_name: str = "lightgbm",
    ensemble_weights: dict[str, float] | None = None,
) -> Pipeline:
    classifier = (
        CalibratedWeightedClassifier(model_params, ensemble_weights, random_seed)
        if ensemble_weights is not None
        else _daily_base_classifier(classifier_name, random_seed, model_params)
    )
    return Pipeline(
        [
            (
                "imputer",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                            feature_names,
                        )
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def _bounded_weights(raw_weights: dict[str, float], maximum: float = 0.6) -> dict[str, float]:
    total = sum(raw_weights.values()) or 1
    weights = {name: value / total for name, value in raw_weights.items()}
    leader = max(weights, key=weights.get)
    if weights[leader] <= maximum:
        return weights
    remainder = 1 - weights[leader]
    weights[leader] = maximum
    scale = (1 - maximum) / remainder
    for name in weights:
        if name != leader:
            weights[name] *= scale
    return weights


def derive_daily_ensemble_plan(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    symbol: str,
    horizon_days: int,
    test_fraction: float,
    random_seed: int,
    model_params: dict[str, float | int] | None = None,
    folds: int = 4,
) -> DailyEnsemblePlan:
    target = build_target(prices, symbol, horizon_days)
    dataset = features.merge(target, on="date", how="left").sort_values("date")
    feature_names = [column for column in features.columns if column != "date"]
    labeled = dataset.dropna(subset=["target"]).reset_index(drop=True)
    split_index = min(max(int(len(labeled) * (1 - test_fraction)), 200), len(labeled) - 40)
    tuning = labeled.iloc[: split_index - horizon_days].reset_index(drop=True)
    validation_start = max(120 + horizon_days, int(len(tuning) * 0.6))
    fold_indices = [
        part
        for part in np.array_split(np.arange(validation_start, len(tuning)), folds)
        if len(part)
    ]
    actual_parts: list[np.ndarray] = []
    probabilities = {name: [] for name in DAILY_ENSEMBLE_MODELS}
    for validation_index in fold_indices:
        train_stop = int(validation_index[0]) - horizon_days
        train = tuning.iloc[:train_stop]
        validation = tuning.iloc[validation_index]
        actual_parts.append(validation["target"].astype(int).to_numpy())
        for name in DAILY_ENSEMBLE_MODELS:
            model = _model_pipeline(
                feature_names,
                random_seed,
                model_params,
                classifier_name=name,
            )
            model.fit(train[feature_names], train["target"].astype(int))
            probabilities[name].append(model.predict_proba(validation[feature_names])[:, 1])
    actual = np.concatenate(actual_parts)
    component_rows: list[dict[str, object]] = []
    raw_weights: dict[str, float] = {}
    joined_probabilities: dict[str, np.ndarray] = {}
    actual_series = pd.Series(actual)
    for name in DAILY_ENSEMBLE_MODELS:
        probability = np.concatenate(probabilities[name])
        joined_probabilities[name] = probability
        accuracy = float(accuracy_score(actual, probability >= 0.5))
        auc = _safe_auc(actual_series, probability)
        auc = 0.5 if np.isnan(auc) else auc
        brier = float(brier_score_loss(actual, probability))
        quality = max(0.01, accuracy + 0.25 * (auc - 0.5) - 0.25 * brier)
        raw_weights[name] = quality
        component_rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "roc_auc": auc,
                "brier": brier,
                "quality_score": quality,
            }
        )
    weights = _bounded_weights(raw_weights)
    combined = sum(weights[name] * joined_probabilities[name] for name in weights)
    threshold_candidates = np.arange(0.46, 0.541, 0.01)
    threshold = float(
        max(
            threshold_candidates,
            key=lambda value: (
                accuracy_score(actual, combined >= value),
                -abs(float(value) - 0.5),
            ),
        )
    )
    return DailyEnsemblePlan(
        symbol=symbol,
        weights=weights,
        threshold=threshold,
        inner_start=pd.Timestamp(tuning.iloc[0]["date"]).date().isoformat(),
        inner_end=pd.Timestamp(tuning.iloc[-1]["date"]).date().isoformat(),
        folds=len(fold_indices),
        component_metrics=component_rows,
    )


def _daily_candidates(
    random_seed: int, trials: int, base_threshold: float
) -> list[tuple[dict[str, float | int], float]]:
    candidates = [(dict(DAILY_DEFAULT_PARAMS), base_threshold)]
    random = np.random.default_rng(random_seed)
    while len(candidates) < max(1, trials):
        params: dict[str, float | int] = {
            "n_estimators": int(random.choice([180, 240, 300, 380, 460])),
            "learning_rate": float(random.choice([0.015, 0.025, 0.03, 0.04, 0.055])),
            "num_leaves": int(random.choice([7, 15, 23, 31, 47])),
            "max_depth": int(random.choice([3, 5, 7, 9, -1])),
            "min_child_samples": int(random.choice([10, 20, 35, 50, 80])),
            "subsample": float(random.choice([0.7, 0.85, 1.0])),
            "subsample_freq": int(random.choice([0, 1])),
            "colsample_bytree": float(random.choice([0.6, 0.75, 0.9, 1.0])),
            "reg_alpha": float(random.choice([0.0, 0.1, 0.3, 0.7, 1.5])),
            "reg_lambda": float(random.choice([0.2, 0.7, 1.0, 2.0, 4.0])),
        }
        threshold = float(
            np.clip(base_threshold + random.choice([-0.04, -0.02, 0, 0.02, 0.04]), 0.4, 0.6)
        )
        signature = (tuple(params.items()), threshold)
        if all((tuple(item.items()), value) != signature for item, value in candidates):
            candidates.append((params, threshold))
    return candidates


def tune_symbol_model(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    symbol: str,
    horizon_days: int,
    test_fraction: float,
    threshold: float,
    random_seed: int,
    trials: int = 4,
    folds: int = 2,
) -> DailyTuningResult:
    target = build_target(prices, symbol, horizon_days)
    dataset = features.merge(target, on="date", how="left").sort_values("date")
    feature_names = [column for column in features.columns if column != "date"]
    labeled = dataset.dropna(subset=["target"]).reset_index(drop=True)
    split_index = min(max(int(len(labeled) * (1 - test_fraction)), 200), len(labeled) - 40)
    tuning = labeled.iloc[: split_index - horizon_days].reset_index(drop=True)
    validation_start = max(120 + horizon_days, int(len(tuning) * 0.6))
    validation_indices = np.arange(validation_start, len(tuning))
    fold_indices = [part for part in np.array_split(validation_indices, folds) if len(part)]
    if not fold_indices:
        raise ValueError(f"Not enough rows for rolling tuning: {symbol}")

    trial_rows: list[dict[str, object]] = []
    for trial_number, (params, candidate_threshold) in enumerate(
        _daily_candidates(random_seed, trials, threshold), start=1
    ):
        accuracies: list[float] = []
        aucs: list[float] = []
        briers: list[float] = []
        for validation_index in fold_indices:
            train_stop = int(validation_index[0]) - horizon_days
            train = tuning.iloc[:train_stop]
            validation = tuning.iloc[validation_index]
            model = _model_pipeline(feature_names, random_seed, params)
            model.fit(train[feature_names], train["target"].astype(int))
            probability = model.predict_proba(validation[feature_names])[:, 1]
            actual = validation["target"].astype(int)
            prediction = (probability >= candidate_threshold).astype(int)
            accuracies.append(float(accuracy_score(actual, prediction)))
            auc = _safe_auc(actual, probability)
            aucs.append(0.5 if np.isnan(auc) else auc)
            briers.append(float(brier_score_loss(actual, probability)))
        mean_accuracy = float(np.mean(accuracies))
        mean_auc = float(np.mean(aucs))
        mean_brier = float(np.mean(briers))
        accuracy_std = float(np.std(accuracies))
        score = mean_accuracy + 0.15 * (mean_auc - 0.5) - 0.15 * mean_brier - 0.1 * accuracy_std
        trial_rows.append(
            {
                "trial": trial_number,
                "params": params,
                "threshold": candidate_threshold,
                "mean_accuracy": mean_accuracy,
                "accuracy_std": accuracy_std,
                "mean_roc_auc": mean_auc,
                "mean_brier": mean_brier,
                "score": score,
            }
        )
    best = max(trial_rows, key=lambda row: float(row["score"]))
    return DailyTuningResult(
        symbol=symbol,
        selected_params=dict(best["params"]),
        selected_threshold=float(best["threshold"]),
        inner_start=pd.Timestamp(tuning.iloc[0]["date"]).date().isoformat(),
        inner_end=pd.Timestamp(tuning.iloc[-1]["date"]).date().isoformat(),
        folds=len(fold_indices),
        trials=trial_rows,
    )


def train_symbol_model(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    symbol: str,
    horizon_days: int,
    test_fraction: float,
    threshold: float,
    random_seed: int,
    model_params: dict[str, float | int] | None = None,
    ensemble_weights: dict[str, float] | None = None,
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

    pipeline = _model_pipeline(
        feature_names,
        random_seed,
        model_params,
        ensemble_weights=ensemble_weights,
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
        "probability_threshold": threshold,
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
