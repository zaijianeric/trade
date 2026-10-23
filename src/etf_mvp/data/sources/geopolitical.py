"""Daily geopolitical-risk indexes from Caldara and Iacoviello."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from etf_mvp.data.base import DataSource, FetchRequest
from etf_mvp.data.http import HttpClient
from etf_mvp.data.kinds import DataKind

GEOPOLITICAL_COLUMNS = [
    "observation_date",
    "available_date",
    "gpr",
    "threats",
    "acts",
    "source",
]


def parse_gpr_daily(content: bytes, start: str, end: str) -> pd.DataFrame:
    raw = pd.read_stata(BytesIO(content))
    observation_date = pd.to_datetime(raw["date"])
    frame = pd.DataFrame(
        {
            "observation_date": observation_date,
            "available_date": observation_date + pd.Timedelta(days=7),
            "gpr": pd.to_numeric(raw["GPRD"], errors="coerce"),
            "threats": pd.to_numeric(raw["GPRD_THREAT"], errors="coerce"),
            "acts": pd.to_numeric(raw["GPRD_ACT"], errors="coerce"),
            "source": "gpr",
        }
    )
    mask = frame["observation_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return frame.loc[mask, GEOPOLITICAL_COLUMNS].reset_index(drop=True)


class GprGeopoliticalSource(DataSource):
    kind = DataKind.GEOPOLITICAL
    name = "gpr"
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta"

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient(min_interval_seconds=0.5)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        response = self.client.get(self.url)
        return parse_gpr_daily(response.content, request.start, request.end)


class SyntheticGeopoliticalSource(DataSource):
    kind = DataKind.GEOPOLITICAL
    name = "synthetic"
    requires_network = False

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        dates = pd.date_range(request.start, request.end, freq="D")
        random = np.random.default_rng(self.random_seed + 404)
        base = np.maximum(0, 100 + np.cumsum(random.normal(0, 2, len(dates))))
        threats = np.maximum(0, base * 0.65 + random.normal(0, 8, len(dates)))
        acts = np.maximum(0, base * 0.35 + random.normal(0, 5, len(dates)))
        return pd.DataFrame(
            {
                "observation_date": dates,
                "available_date": dates + pd.Timedelta(days=7),
                "gpr": base,
                "threats": threats,
                "acts": acts,
                "source": self.name,
            }
        )[GEOPOLITICAL_COLUMNS]
