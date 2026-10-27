from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as exchange_calendars
import numpy as np
import pandas as pd

NEW_YORK = "America/New_York"
BAR_MINUTES = 5


@dataclass(frozen=True)
class IntradayQualityResult:
    symbol: str
    cleaned: pd.DataFrame
    sessions: pd.DataFrame
    summary: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def write_intraday_source_metadata(
    path: Path,
    symbol: str,
    feed: str,
    requested_start: str,
    requested_end: str | None,
    rows: int,
) -> None:
    write_json(
        {
            "symbol": symbol,
            "provider": "alpaca",
            "feed": feed,
            "timeframe": "5Min",
            "adjustment": "all",
            "requested_start": requested_start,
            "requested_end": requested_end,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "rows": rows,
            "sha256": file_sha256(path),
        },
        path.with_suffix(".metadata.json"),
    )


def next_xnys_session(value: pd.Timestamp) -> pd.Timestamp:
    calendar = exchange_calendars.get_calendar("XNYS")
    session = calendar.date_to_session(pd.Timestamp(value).normalize(), direction="next")
    if session <= pd.Timestamp(value).normalize():
        session = calendar.next_session(session)
    return pd.Timestamp(session).tz_localize(None)


def _invalid_price_rows(frame: pd.DataFrame) -> pd.Series:
    price_columns = ["open", "high", "low", "close", "vwap"]
    missing_price = frame[price_columns].isna().any(axis=1)
    nonpositive_price = frame[price_columns].le(0).any(axis=1)
    invalid_ohlc = (
        frame["high"].lt(frame[["open", "close"]].max(axis=1))
        | frame["low"].gt(frame[["open", "close"]].min(axis=1))
        | frame["high"].lt(frame["low"])
    )
    invalid_activity = (
        frame["volume"].isna()
        | frame["trade_count"].isna()
        | frame["volume"].lt(0)
        | frame["trade_count"].lt(0)
    )
    return missing_price | nonpositive_price | invalid_ohlc | invalid_activity


def _max_missing_run(observed: np.ndarray) -> int:
    longest = current = 0
    for is_observed in observed:
        current = 0 if is_observed else current + 1
        longest = max(longest, current)
    return longest


def _clean_session(
    expected_timestamps: pd.DatetimeIndex,
    valid: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    grid = pd.DataFrame({"timestamp": expected_timestamps})
    source_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    cleaned = grid.merge(valid[source_columns], on="timestamp", how="left")
    cleaned["is_observed"] = cleaned["close"].notna()
    interpolated_close = cleaned["close"].interpolate(limit=3, limit_direction="both")
    imputed = ~cleaned["is_observed"]
    for column in ("open", "high", "low", "close", "vwap"):
        cleaned.loc[imputed, column] = interpolated_close.loc[imputed]
    cleaned[["volume", "trade_count"]] = cleaned[["volume", "trade_count"]].fillna(0)
    local_time = cleaned["timestamp"].dt.tz_convert(NEW_YORK)
    cleaned["symbol"] = symbol
    cleaned["session_date"] = local_time.dt.tz_localize(None).dt.normalize()
    cleaned["slot"] = np.arange(len(cleaned))
    cleaned["is_imputed"] = imputed
    return cleaned[
        [
            "symbol",
            "timestamp",
            "session_date",
            "slot",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
            "is_observed",
            "is_imputed",
        ]
    ]


def audit_intraday_bars(
    bars: pd.DataFrame,
    symbol: str,
    min_completeness: float = 0.90,
    max_consecutive_missing: int = 3,
) -> IntradayQualityResult:
    if bars.empty:
        raise ValueError(f"No intraday bars available for {symbol}")
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    for column in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate_rows = int(frame.duplicated("timestamp", keep="last").sum())
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    local_date = frame["timestamp"].dt.tz_convert(NEW_YORK).dt.tz_localize(None).dt.normalize()
    frame["derived_session_date"] = local_date
    frame["invalid_values"] = _invalid_price_rows(frame)

    calendar = exchange_calendars.get_calendar("XNYS")
    start = local_date.min()
    end = local_date.max()
    expected_sessions = calendar.sessions_in_range(start, end)
    expected_session_dates = pd.DatetimeIndex(expected_sessions).tz_localize(None)
    off_calendar_rows = int((~local_date.isin(expected_session_dates)).sum())

    session_rows: list[dict[str, Any]] = []
    cleaned_sessions: list[pd.DataFrame] = []
    for session in expected_sessions:
        session_date = pd.Timestamp(session).tz_localize(None)
        session_open = calendar.session_open(session)
        session_close = calendar.session_close(session)
        expected_timestamps = pd.date_range(
            session_open,
            session_close,
            freq=f"{BAR_MINUTES}min",
            inclusive="left",
        )
        expected_bars = len(expected_timestamps)
        day = frame.loc[frame["derived_session_date"] == session_date].copy()
        on_grid = day["timestamp"].isin(expected_timestamps)
        valid = day.loc[on_grid & ~day["invalid_values"]].copy()
        valid = valid.drop_duplicates("timestamp", keep="last")
        observed = expected_timestamps.isin(valid["timestamp"])
        observed_bars = int(observed.sum())
        completeness = observed_bars / expected_bars if expected_bars else 0.0
        longest_gap = _max_missing_run(observed)
        is_early_close = expected_bars < 78
        training_eligible = (
            not is_early_close
            and completeness >= min_completeness
            and longest_gap <= max_consecutive_missing
        )
        session_rows.append(
            {
                "session_date": session_date,
                "expected_bars": expected_bars,
                "observed_bars": observed_bars,
                "missing_bars": expected_bars - observed_bars,
                "completeness": completeness,
                "max_consecutive_missing": longest_gap,
                "invalid_rows": int((~on_grid | day["invalid_values"]).sum()),
                "is_early_close": is_early_close,
                "training_eligible": training_eligible,
            }
        )
        if training_eligible:
            cleaned_sessions.append(_clean_session(expected_timestamps, valid, symbol))

    sessions = pd.DataFrame(session_rows)
    cleaned = pd.concat(cleaned_sessions, ignore_index=True) if cleaned_sessions else pd.DataFrame()
    total_expected = int(sessions["expected_bars"].sum())
    total_observed = int(sessions["observed_bars"].sum())
    quality_score = 100 * total_observed / total_expected if total_expected else 0.0
    summary: dict[str, Any] = {
        "symbol": symbol,
        "provider": "alpaca",
        "generated_at": datetime.now(UTC).isoformat(),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "raw_rows": len(bars),
        "clean_rows": len(cleaned),
        "expected_sessions": len(sessions),
        "observed_sessions": int(sessions["observed_bars"].gt(0).sum()),
        "missing_sessions": int(sessions["observed_bars"].eq(0).sum()),
        "eligible_sessions": int(sessions["training_eligible"].sum()),
        "early_close_sessions": int(sessions["is_early_close"].sum()),
        "duplicate_rows": duplicate_rows,
        "invalid_rows": int(sessions["invalid_rows"].sum()) + off_calendar_rows,
        "off_calendar_rows": off_calendar_rows,
        "expected_bars": total_expected,
        "observed_bars": total_observed,
        "completeness": total_observed / total_expected if total_expected else 0.0,
        "quality_score": quality_score,
        "min_completeness": min_completeness,
        "max_consecutive_missing": max_consecutive_missing,
    }
    return IntradayQualityResult(symbol, cleaned, sessions, summary)


def prepare_intraday_file(
    raw_path: Path,
    clean_path: Path,
    sessions_path: Path,
    summary_path: Path,
    symbol: str,
    feed: str = "iex",
) -> IntradayQualityResult:
    bars = pd.read_parquet(raw_path)
    metadata_path = raw_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        write_json(
            {
                "symbol": symbol,
                "provider": "alpaca",
                "feed": feed,
                "timeframe": "5Min",
                "adjustment": "all",
                "retrieved_at": datetime.fromtimestamp(
                    raw_path.stat().st_mtime, tz=UTC
                ).isoformat(),
                "metadata_origin": "backfilled_from_file",
                "rows": len(bars),
                "sha256": file_sha256(raw_path),
            },
            metadata_path,
        )
    result = audit_intraday_bars(bars, symbol)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    result.cleaned.to_parquet(clean_path, index=False)
    result.sessions.to_parquet(sessions_path, index=False)
    summary = {
        **result.summary,
        "feed": feed,
        "raw_file": str(raw_path),
        "clean_file": str(clean_path),
        "source_metadata": str(metadata_path),
        "raw_sha256": file_sha256(raw_path),
    }
    write_json(summary, summary_path)
    return IntradayQualityResult(symbol, result.cleaned, result.sessions, summary)
