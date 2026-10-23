from __future__ import annotations

import pandas as pd

from etf_mvp.features.asof import asof_values, rolling_zscore


def build_geopolitical_features(
    observations: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(index=dates)
    levels = asof_values(dates, observations, ["gpr", "threats", "acts"])
    risk = levels["gpr"]
    threats = levels["threats"]
    acts = levels["acts"]
    columns = {
        "geo__gpr_level": risk,
        "geo__gpr_chg_5d": risk.pct_change(5),
        "geo__gpr_chg_20d": risk.pct_change(20),
        "geo__gpr_z_60d": rolling_zscore(risk, 60),
        "geo__gpr_z_252d": rolling_zscore(risk, 252),
        "geo__threats_z_252d": rolling_zscore(threats, 252),
        "geo__acts_z_252d": rolling_zscore(acts, 252),
    }
    return pd.DataFrame(columns, index=dates)
