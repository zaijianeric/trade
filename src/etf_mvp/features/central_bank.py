from __future__ import annotations

import numpy as np
import pandas as pd

from etf_mvp.features.asof import asof_values

PRE_MEETING_DAYS = 7


def build_central_bank_features(events: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """FOMC calendar position (known in advance) and the latest decision and tone."""
    meetings = events.loc[events["event_type"] == "fomc_meeting"]
    scheduled = np.sort(
        pd.to_datetime(meetings.loc[meetings["is_scheduled"].astype(bool), "event_date"])
        .to_numpy()
        .astype("datetime64[D]")
    )
    day_values = dates.to_numpy().astype("datetime64[D]")
    features = pd.DataFrame(index=dates)

    if len(scheduled):
        next_index = np.searchsorted(scheduled, day_values, side="left")
        has_next = next_index < len(scheduled)
        next_meeting = scheduled[np.minimum(next_index, len(scheduled) - 1)]
        to_next = np.busday_count(day_values, next_meeting).astype(float)
        to_next[~has_next] = np.nan
        features["cb__days_to_next_meeting"] = to_next
        features["cb__is_meeting_day"] = (to_next == 0).astype(float)
        features["cb__pre_meeting_window"] = (
            (to_next >= 0) & (to_next <= PRE_MEETING_DAYS)
        ).astype(float)

    decided = meetings.dropna(subset=["rate_upper"]).sort_values("event_date").copy()
    if not decided.empty:
        decided["rate_change"] = decided["rate_upper"].diff()
        decided["hawkish_change"] = decided["hawkish_score"].diff()
        decided["decision_date"] = decided["available_date"]
        latest = asof_values(
            dates,
            decided,
            ["rate_upper", "rate_change", "hawkish_score", "hawkish_change", "decision_date"],
        )
        features["cb__rate_upper"] = latest["rate_upper"]
        features["cb__last_rate_change"] = latest["rate_change"]
        features["cb__hawkish_score"] = latest["hawkish_score"]
        features["cb__hawkish_change"] = latest["hawkish_change"]
        since = np.full(len(dates), np.nan)
        known = latest["decision_date"].notna().to_numpy()
        since[known] = np.busday_count(
            latest["decision_date"].to_numpy()[known].astype("datetime64[D]"), day_values[known]
        )
        features["cb__days_since_decision"] = since
    return features
