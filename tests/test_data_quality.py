from __future__ import annotations

import exchange_calendars as exchange_calendars
import pandas as pd

from etf_mvp.data_quality import audit_intraday_bars, next_xnys_session


def _session_bars(session: str, symbol: str = "QQQ") -> pd.DataFrame:
    calendar = exchange_calendars.get_calendar("XNYS")
    timestamps = pd.date_range(
        calendar.session_open(session),
        calendar.session_close(session),
        freq="5min",
        inclusive="left",
    )
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "trade_count": 50,
            "vwap": 100.2,
        }
    )


def test_intraday_quality_fills_only_short_gaps() -> None:
    bars = _session_bars("2026-10-20")
    duplicate = bars.iloc[[0]].copy()
    bars = pd.concat([bars.drop(index=[10, 11]), duplicate], ignore_index=True)

    result = audit_intraday_bars(bars, "QQQ")

    assert result.summary["duplicate_rows"] == 1
    assert result.summary["eligible_sessions"] == 1
    assert len(result.cleaned) == 78
    assert int(result.cleaned["is_imputed"].sum()) == 2
    assert result.sessions.iloc[0]["max_consecutive_missing"] == 2


def test_intraday_quality_rejects_long_gaps() -> None:
    bars = _session_bars("2026-10-20").drop(index=[10, 11, 12, 13])

    result = audit_intraday_bars(bars, "QQQ")

    assert result.summary["eligible_sessions"] == 0
    assert result.cleaned.empty


def test_early_close_is_recognized_but_not_used_for_fixed_shape_model() -> None:
    result = audit_intraday_bars(_session_bars("2026-11-27"), "QQQ")

    assert result.sessions.iloc[0]["expected_bars"] == 42
    assert bool(result.sessions.iloc[0]["is_early_close"])
    assert result.summary["eligible_sessions"] == 0


def test_next_xnys_session_skips_holiday() -> None:
    assert next_xnys_session(pd.Timestamp("2026-11-25")) == pd.Timestamp("2026-11-27")
