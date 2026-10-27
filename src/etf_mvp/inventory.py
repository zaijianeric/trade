from __future__ import annotations

import importlib.metadata
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as parquet

PROVIDER_LABELS = {
    "alpaca": "Alpaca Market Data",
    "fred": "Federal Reserve Economic Data (FRED)",
    "gpr": "Geopolitical Risk Index (GPR)",
    "hybrid": "SF Fed + GDELT + Alpha Vantage（可选）",
    "nasdaq": "Nasdaq",
    "sec_edgar": "SEC EDGAR",
    "sf_fed": "San Francisco Fed",
    "synthetic": "Synthetic（演示数据）",
}

DATASETS = (
    {
        "category": "标的历史行情",
        "name": "ETF 日线行情",
        "path": "data/market/prices.parquet",
        "provider": "nasdaq",
        "date_columns": ("date",),
    },
    {
        "category": "标的历史行情",
        "name": "QQQ 5分钟行情",
        "path": "data/intraday/QQQ_5min.parquet",
        "provider": "alpaca",
        "date_columns": ("timestamp", "session_date"),
    },
    {
        "category": "标的历史行情",
        "name": "SOXX 5分钟行情",
        "path": "data/intraday/SOXX_5min.parquet",
        "provider": "alpaca",
        "date_columns": ("timestamp", "session_date"),
    },
    {
        "category": "经济数据",
        "name": "宏观经济指标",
        "path": "data/raw/macro/fred.parquet",
        "provider": "fred",
        "date_columns": ("observation_date", "available_date"),
    },
    {
        "category": "经济数据",
        "name": "美联储事件",
        "path": "data/raw/central_bank/fred.parquet",
        "provider": "fred",
        "date_columns": ("event_date", "available_date"),
    },
    {
        "category": "经济数据",
        "name": "地缘风险指数",
        "path": "data/raw/geopolitical/gpr.parquet",
        "provider": "gpr",
        "date_columns": ("observation_date", "available_date"),
    },
    {
        "category": "新闻",
        "name": "每日新闻情绪",
        "path": "data/raw/news/hybrid.parquet",
        "provider": "hybrid",
        "date_columns": ("published_at", "available_date"),
    },
    {
        "category": "基本面",
        "name": "ETF 成分股财务",
        "path": "data/raw/fundamental/sec_edgar.parquet",
        "provider": "sec_edgar",
        "date_columns": ("period_end", "available_date"),
    },
)


def format_bytes(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _format_datetime(value: Any, include_time: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York").tz_localize(None)
    return timestamp.strftime("%Y-%m-%d %H:%M") if include_time else timestamp.strftime("%Y-%m-%d")


def _file_updated_at(path: Path) -> str:
    updated = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    return updated.strftime("%Y-%m-%d %H:%M")


def _estimator_details(model: Any) -> tuple[str, str]:
    estimator = model
    if hasattr(model, "steps") and model.steps:
        estimator = model.steps[-1][1]
    algorithm = type(estimator).__name__
    module = type(estimator).__module__.split(".", 1)[0]
    distribution = {"lightgbm": "lightgbm", "sklearn": "scikit-learn"}.get(module, module)
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "未知"
    return algorithm, version


def scan_models(project_root: Path) -> list[dict[str, Any]]:
    artifacts_dir = project_root / "artifacts"
    rows: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.rglob("*.joblib")):
        try:
            model = joblib.load(path)
            algorithm, version = _estimator_details(model)
            relative = path.relative_to(project_root)
            if path.parent.name == "models":
                scope = "日线方向分类"
                target = path.stem
            else:
                scope = "5分钟序列回归"
                target = path.parent.name
            rows.append(
                {
                    "模型": target,
                    "用途": scope,
                    "算法": algorithm,
                    "版本": version,
                    "大小": format_bytes(path.stat().st_size),
                    "更新时间": _file_updated_at(path),
                    "文件": str(relative),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "模型": path.stem,
                    "用途": "无法读取",
                    "算法": type(error).__name__,
                    "版本": "—",
                    "大小": format_bytes(path.stat().st_size),
                    "更新时间": _file_updated_at(path),
                    "文件": str(path.relative_to(project_root)),
                }
            )
    return rows


def _latest_data_value(path: Path, date_columns: tuple[str, ...]) -> str:
    schema_columns = set(parquet.ParquetFile(path).schema.names)
    selected = next((column for column in date_columns if column in schema_columns), None)
    if not selected:
        return "—"
    values = pd.read_parquet(path, columns=[selected])[selected]
    return _format_datetime(pd.to_datetime(values, errors="coerce").max())


def scan_data_sources(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = project_root / str(dataset["path"])
        if not path.exists():
            rows.append(
                {
                    "分类": dataset["category"],
                    "数据集": dataset["name"],
                    "提供方": PROVIDER_LABELS[str(dataset["provider"])],
                    "状态": "未获取",
                    "最新数据": "—",
                    "文件更新": "—",
                    "记录数": 0,
                    "大小": "—",
                }
            )
            continue
        metadata = parquet.ParquetFile(path).metadata
        rows.append(
            {
                "分类": dataset["category"],
                "数据集": dataset["name"],
                "提供方": PROVIDER_LABELS[str(dataset["provider"])],
                "状态": "可用",
                "最新数据": _latest_data_value(path, dataset["date_columns"]),
                "文件更新": _file_updated_at(path),
                "记录数": metadata.num_rows,
                "大小": format_bytes(path.stat().st_size),
            }
        )
    return rows


def scan_dependencies(project_root: Path) -> list[dict[str, str]]:
    with (project_root / "pyproject.toml").open("rb") as project_file:
        dependencies = tomllib.load(project_file)["project"]["dependencies"]
    rows = [{"依赖": "Python", "版本": platform_python_version(), "约束": ">=3.11,<3.13"}]
    for specification in dependencies:
        name = specification.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip()
        constraint = specification[len(name) :]
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "未安装"
        rows.append({"依赖": name, "版本": installed, "约束": constraint})
    return rows


def scan_intraday_quality(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quality_dir = project_root / "artifacts" / "data_quality"
    for path in (quality_dir / "QQQ.json", quality_dir / "SOXX.json"):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as quality_file:
            summary = json.load(quality_file)
        rows.append(
            {
                "标的": summary["symbol"],
                "Feed": str(summary.get("feed", "—")).upper(),
                "质量评分": float(summary["quality_score"]),
                "K线完整率": float(summary["completeness"]),
                "可训练交易日": int(summary["eligible_sessions"]),
                "缺失交易日": int(summary["missing_sessions"]),
                "异常记录": int(summary["invalid_rows"]),
                "重复记录": int(summary["duplicate_rows"]),
                "清洗后记录": int(summary["clean_rows"]),
                "报告时间": _format_datetime(summary["generated_at"], include_time=True),
            }
        )
    return rows


def scan_intraday_quality_comparison(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "artifacts" / "data_quality" / "common_window_comparison.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as comparison_file:
        comparisons = json.load(comparison_file)
    return [
        {
            "标的": row["symbol"],
            "共同回测交易日": int(row["common_sessions"]),
            "回测开始": row["backtest_start"],
            "回测结束": row["backtest_end"],
            "清洗前准确率": float(row["direction_accuracy_before"]),
            "清洗后准确率": float(row["direction_accuracy_after"]),
            "变化": float(row["direction_accuracy_delta"]),
        }
        for row in comparisons
    ]


def scan_feature_ablation(project_root: Path) -> list[dict[str, Any]]:
    experiment_dir = project_root / "artifacts" / "feature_ablation"
    rows: list[dict[str, Any]] = []
    daily_path = experiment_dir / "daily.json"
    if daily_path.exists():
        with daily_path.open(encoding="utf-8") as daily_file:
            for row in json.load(daily_file):
                rows.append(
                    {
                        "任务": "日线方向",
                        "标的": row["symbol"],
                        "基线特征": int(row["baseline_feature_count"]),
                        "增强特征": int(row["enhanced_feature_count"]),
                        "基线准确率": float(row["baseline_accuracy"]),
                        "增强准确率": float(row["enhanced_accuracy"]),
                        "准确率变化": float(row["accuracy_delta"]),
                        "辅助指标变化": float(row["brier_delta"]),
                        "辅助指标": "Brier（越低越好）",
                        "是否达标": "是" if row["accepted"] else "否",
                        "最终采用": row.get(
                            "selected_feature_set",
                            "enhanced" if row["accepted"] else "baseline",
                        ),
                    }
                )
    intraday_path = experiment_dir / "intraday.json"
    if intraday_path.exists():
        with intraday_path.open(encoding="utf-8") as intraday_file:
            for row in json.load(intraday_file):
                rows.append(
                    {
                        "任务": "5分钟路径",
                        "标的": row["symbol"],
                        "基线特征": int(row["baseline_feature_count"]),
                        "增强特征": int(row["enhanced_feature_count"]),
                        "基线准确率": float(row["baseline_direction_accuracy"]),
                        "增强准确率": float(row["enhanced_direction_accuracy"]),
                        "准确率变化": float(row["direction_accuracy_delta"]),
                        "辅助指标变化": float(row["path_rmse_delta"]),
                        "辅助指标": "路径RMSE（越低越好）",
                        "是否达标": "是" if row["accepted"] else "否",
                        "最终采用": row.get(
                            "selected_feature_set",
                            "enhanced" if row["accepted"] else "baseline",
                        ),
                    }
                )
    news_path = experiment_dir / "news.json"
    if news_path.exists():
        with news_path.open(encoding="utf-8") as news_file:
            for row in json.load(news_file):
                rows.append(
                    {
                        "任务": "新闻因子",
                        "标的": row["symbol"],
                        "基线特征": 0,
                        "增强特征": int(row["news_feature_count"]),
                        "基线准确率": float(row["without_news_accuracy"]),
                        "增强准确率": float(row["with_news_accuracy"]),
                        "准确率变化": float(row["accuracy_delta"]),
                        "辅助指标变化": float(row["brier_delta"]),
                        "辅助指标": "Brier（越低越好）",
                        "是否达标": "是" if row["accepted"] else "否",
                        "最终采用": row["selected_news_features"],
                    }
                )
    return rows


def scan_news_quality(project_root: Path) -> list[dict[str, Any]]:
    quality_path = project_root / "artifacts" / "data_quality" / "news.json"
    if not quality_path.exists():
        return []
    with quality_path.open(encoding="utf-8") as quality_file:
        quality = json.load(quality_file)
    if quality.get("status") != "available":
        return []
    return [
        {
            "来源": source["source"],
            "记录数": int(source["rows"]),
            "覆盖开始": source["coverage_start"],
            "最新可用": source["latest_available"],
            "活跃日期": int(source["active_days"]),
            "距行情截止": int(quality["stale_calendar_days"]),
        }
        for source in quality["sources"]
    ]


def scan_rolling_tuning(project_root: Path) -> list[dict[str, Any]]:
    experiment_dir = project_root / "artifacts" / "rolling_tuning"
    rows: list[dict[str, Any]] = []
    daily_path = experiment_dir / "daily.json"
    if daily_path.exists():
        with daily_path.open(encoding="utf-8") as daily_file:
            for row in json.load(daily_file):
                rows.append(
                    {
                        "任务": "日线方向",
                        "标的": row["symbol"],
                        "特征集": row["feature_set"],
                        "内层窗口": f"{row['inner_start']} → {row['inner_end']}",
                        "折数": int(row["folds"]),
                        "试验数": int(row["trial_count"]),
                        "固定参数准确率": float(row["fixed_accuracy"]),
                        "调参后准确率": float(row["tuned_accuracy"]),
                        "准确率变化": float(row["accuracy_delta"]),
                        "辅助指标变化": float(row["brier_delta"]),
                        "辅助指标": "Brier（越低越好）",
                        "是否采用": "是" if row["accepted"] else "否",
                        "最终参数组": row["selected_parameter_set"],
                    }
                )
    intraday_path = experiment_dir / "intraday.json"
    if intraday_path.exists():
        with intraday_path.open(encoding="utf-8") as intraday_file:
            for row in json.load(intraday_file):
                rows.append(
                    {
                        "任务": "5分钟路径",
                        "标的": row["symbol"],
                        "特征集": row["feature_set"],
                        "内层窗口": f"{row['inner_start']} → {row['inner_end']}",
                        "折数": int(row["folds"]),
                        "试验数": int(row["trial_count"]),
                        "固定参数准确率": float(row["fixed_direction_accuracy"]),
                        "调参后准确率": float(row["tuned_direction_accuracy"]),
                        "准确率变化": float(row["direction_accuracy_delta"]),
                        "辅助指标变化": float(row["path_rmse_delta"]),
                        "辅助指标": "路径RMSE（越低越好）",
                        "是否采用": "是" if row["accepted"] else "否",
                        "最终参数组": row["selected_parameter_set"],
                    }
                )
    return rows


def scan_ensemble(project_root: Path) -> list[dict[str, Any]]:
    experiment_dir = project_root / "artifacts" / "ensemble"
    rows: list[dict[str, Any]] = []
    daily_path = experiment_dir / "daily.json"
    if daily_path.exists():
        with daily_path.open(encoding="utf-8") as daily_file:
            for row in json.load(daily_file):
                weights = " · ".join(
                    f"{name} {float(weight):.0%}"
                    for name, weight in row["weights"].items()
                )
                rows.append(
                    {
                        "任务": "日线方向",
                        "标的": row["symbol"],
                        "特征/参数": f"{row['feature_set']} / {row['parameter_set']}",
                        "单模型准确率": float(row["single_accuracy"]),
                        "集成准确率": float(row["ensemble_accuracy"]),
                        "准确率变化": float(row["accuracy_delta"]),
                        "辅助指标变化": float(row["brier_delta"]),
                        "辅助指标": "Brier（越低越好）",
                        "权重": weights,
                        "是否采用": "是" if row["accepted"] else "否",
                        "最终策略": row["selected_strategy"],
                    }
                )
    intraday_path = experiment_dir / "intraday.json"
    if intraday_path.exists():
        with intraday_path.open(encoding="utf-8") as intraday_file:
            for row in json.load(intraday_file):
                weights = " · ".join(
                    f"{name} {float(weight):.0%}"
                    for name, weight in row["weights"].items()
                )
                rows.append(
                    {
                        "任务": "5分钟路径",
                        "标的": row["symbol"],
                        "特征/参数": f"{row['feature_set']} / {row['parameter_set']}",
                        "单模型准确率": float(row["single_direction_accuracy"]),
                        "集成准确率": float(row["ensemble_direction_accuracy"]),
                        "准确率变化": float(row["direction_accuracy_delta"]),
                        "辅助指标变化": float(row["path_rmse_delta"]),
                        "辅助指标": "路径RMSE（越低越好）",
                        "权重": weights,
                        "是否采用": "是" if row["accepted"] else "否",
                        "最终策略": row["selected_strategy"],
                    }
                )
    return rows


def platform_python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])
