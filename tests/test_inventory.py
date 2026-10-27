from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from etf_mvp.inventory import (
    format_bytes,
    scan_data_sources,
    scan_ensemble,
    scan_intraday_quality,
    scan_models,
    scan_news_quality,
    scan_rolling_tuning,
)


def test_format_bytes() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"


def test_scan_models(tmp_path: Path) -> None:
    model_path = tmp_path / "artifacts" / "models" / "QQQ.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump(DummyClassifier(), model_path)

    models = scan_models(tmp_path)

    assert len(models) == 1
    assert models[0]["模型"] == "QQQ"
    assert models[0]["算法"] == "DummyClassifier"
    assert models[0]["用途"] == "日线方向分类"


def test_scan_data_sources_reports_available_and_missing(tmp_path: Path) -> None:
    price_path = tmp_path / "data" / "market" / "prices.parquet"
    price_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-22", "2026-10-23"]),
            "symbol": ["QQQ", "QQQ"],
        }
    ).to_parquet(price_path)

    sources = scan_data_sources(tmp_path)
    daily = next(source for source in sources if source["数据集"] == "ETF 日线行情")
    intraday = next(source for source in sources if source["数据集"] == "QQQ 5分钟行情")

    assert daily["状态"] == "可用"
    assert daily["最新数据"] == "2026-10-23"
    assert daily["记录数"] == 2
    assert intraday["状态"] == "未获取"


def test_scan_intraday_quality_ignores_comparison_files(tmp_path: Path) -> None:
    quality_dir = tmp_path / "artifacts" / "data_quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "common_window_comparison.json").write_text("[]")

    assert scan_intraday_quality(tmp_path) == []


def test_scan_rolling_tuning(tmp_path: Path) -> None:
    tuning_dir = tmp_path / "artifacts" / "rolling_tuning"
    tuning_dir.mkdir(parents=True)
    (tuning_dir / "daily.json").write_text(
        """[{"symbol":"QQQ","feature_set":"baseline","inner_start":"2020-01-01",\
"inner_end":"2024-01-01","folds":4,"trial_count":24,"fixed_accuracy":0.53,\
"tuned_accuracy":0.55,"accuracy_delta":0.02,"brier_delta":-0.004,\
"accepted":true,"selected_parameter_set":"tuned"}]""",
        encoding="utf-8",
    )

    rows = scan_rolling_tuning(tmp_path)

    assert rows[0]["标的"] == "QQQ"
    assert rows[0]["最终参数组"] == "tuned"
    assert rows[0]["试验数"] == 24


def test_scan_ensemble(tmp_path: Path) -> None:
    ensemble_dir = tmp_path / "artifacts" / "ensemble"
    ensemble_dir.mkdir(parents=True)
    (ensemble_dir / "daily.json").write_text(
        """[{"symbol":"SOXX","feature_set":"enhanced","parameter_set":"tuned",\
"single_accuracy":0.57,"ensemble_accuracy":0.59,"accuracy_delta":0.02,\
"brier_delta":-0.004,"weights":{"lightgbm":0.4,"extra_trees":0.35,\
"logistic":0.25},"accepted":true,"selected_strategy":"ensemble"}]""",
        encoding="utf-8",
    )

    rows = scan_ensemble(tmp_path)

    assert rows[0]["标的"] == "SOXX"
    assert rows[0]["最终策略"] == "ensemble"
    assert "lightgbm 40%" in rows[0]["权重"]


def test_scan_news_quality(tmp_path: Path) -> None:
    quality_dir = tmp_path / "artifacts" / "data_quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "news.json").write_text(
        """{"status":"available","stale_calendar_days":1,"sources":[\
{"source":"gdelt","rows":200,"coverage_start":"2026-08-01",\
"latest_available":"2026-10-23","active_days":60}]}""",
        encoding="utf-8",
    )

    rows = scan_news_quality(tmp_path)

    assert rows[0]["来源"] == "gdelt"
    assert rows[0]["记录数"] == 200
