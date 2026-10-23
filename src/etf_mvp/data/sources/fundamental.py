"""Company fundamentals from SEC EDGAR XBRL company facts.

ETFs have no financial statements of their own, so the fundamental layer collects
the reported facts of each ETF's configured holdings. Each fact keeps the date it
was first filed; later filings that repeat a period as a comparative are dropped.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from etf_mvp.data.base import DataSource, FetchRequest
from etf_mvp.data.calendar import next_business_day
from etf_mvp.data.http import HttpClient
from etf_mvp.data.kinds import DataKind

# canonical concept -> (XBRL tags in priority order, unit)
CONCEPTS: dict[str, tuple[tuple[str, ...], str]] = {
    "revenue": (
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ),
        "USD",
    ),
    "net_income": (("NetIncomeLoss",), "USD"),
    "eps_diluted": (("EarningsPerShareDiluted",), "USD/shares"),
    "equity": (("StockholdersEquity",), "USD"),
}
FLOW_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}


def holdings_from_options(request: FetchRequest) -> list[str]:
    holdings: dict[str, list[str]] = request.options.get("holdings", {})
    tickers = {ticker.upper() for members in holdings.values() for ticker in members}
    return sorted(tickers)


def period_class(start: pd.Series, end: pd.Series) -> pd.Series:
    """'Q' for ~3-month durations, 'FY' for ~12-month, 'I' for instants, else None."""
    days = (end - start).dt.days
    result = pd.Series(np.where(start.isna(), "I", None), index=end.index, dtype=object)
    result[(days >= 70) & (days <= 100)] = "Q"
    result[(days >= 350) & (days <= 380)] = "FY"
    return result


def parse_company_facts(payload: dict, ticker: str) -> pd.DataFrame:
    gaap = payload.get("facts", {}).get("us-gaap", {})
    frames: list[pd.DataFrame] = []
    for concept, (tags, unit) in CONCEPTS.items():
        for tag in tags:
            facts = gaap.get(tag, {}).get("units", {}).get(unit)
            if facts:
                frames.append(pd.DataFrame(facts).assign(concept=concept))
                break
    if not frames:
        return pd.DataFrame()

    facts = pd.concat(frames, ignore_index=True)
    facts = facts.loc[facts["form"].isin(FLOW_FORMS)].copy()
    facts["end"] = pd.to_datetime(facts["end"])
    facts["start"] = pd.to_datetime(facts["start"]) if "start" in facts else pd.NaT
    facts["filed"] = pd.to_datetime(facts["filed"])
    facts["fiscal_period"] = period_class(facts["start"], facts["end"])
    facts = facts.dropna(subset=["fiscal_period"])
    # First filing wins: later filings re-report old periods as comparatives.
    facts = facts.sort_values("filed").drop_duplicates(
        ["concept", "end", "fiscal_period"], keep="first"
    )
    return pd.DataFrame(
        {
            "ticker": ticker,
            "concept": facts["concept"],
            "period_end": facts["end"],
            "fiscal_period": facts["fiscal_period"],
            "form": facts["form"],
            # Filings often land after the close; use them from the next session.
            "available_date": next_business_day(facts["filed"]),
            "value": facts["val"],
            "source": "sec_edgar",
        }
    )


class SecEdgarFundamentalSource(DataSource):
    kind = DataKind.FUNDAMENTAL
    name = "sec_edgar"
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    facts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

    def __init__(self, user_agent: str | None = None, client: HttpClient | None = None) -> None:
        agent = user_agent or os.getenv("SEC_USER_AGENT")
        if not agent:
            raise ValueError(
                "SEC_USER_AGENT is required for fundamental provider 'sec_edgar' "
                '(SEC asks for a contact, e.g. "Your Name you@example.com")'
            )
        # SEC fair-access policy: at most 10 requests per second.
        self.client = client or HttpClient(user_agent=agent, min_interval_seconds=0.15)

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        cik_map = self._cik_map()
        frames: list[pd.DataFrame] = []
        for ticker in holdings_from_options(request):
            cik = cik_map.get(ticker.replace(".", "-"))
            if cik is None:
                raise ValueError(f"SEC has no CIK for ticker {ticker}")
            payload = self.client.get(self.facts_url.format(cik=cik)).json()
            frames.append(parse_company_facts(payload, ticker))
        facts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if facts.empty:
            return facts
        start, end = pd.Timestamp(request.start), pd.Timestamp(request.end)
        # Keep a year before start so the first year-over-year changes can be computed.
        in_range = facts["period_end"].between(start - pd.DateOffset(years=1), end)
        return facts.loc[in_range]

    def _cik_map(self) -> dict[str, int]:
        payload = self.client.get(self.tickers_url).json()
        return {row["ticker"].upper(): int(row["cik_str"]) for row in payload.values()}


class SyntheticFundamentalSource(DataSource):
    kind = DataKind.FUNDAMENTAL
    name = "synthetic"
    requires_network = False

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        random = np.random.default_rng(self.random_seed + 202)
        start = pd.Timestamp(request.start) - pd.DateOffset(years=1)
        quarter_ends = pd.date_range(start, request.end, freq="QE")
        rows: list[dict] = []
        for index, ticker in enumerate(holdings_from_options(request)):
            revenue = 1e10 * (1 + index * 0.3)
            margin = 0.1 + 0.02 * (index % 5)
            equity = revenue * 2
            for period_end in quarter_ends:
                revenue *= np.exp(random.normal(0.015, 0.04))
                margin = float(np.clip(margin + random.normal(0, 0.01), -0.05, 0.4))
                equity *= np.exp(random.normal(0.01, 0.02))
                filed = period_end + pd.Timedelta(days=int(random.integers(25, 45)))
                values = {
                    "revenue": revenue,
                    "net_income": revenue * margin,
                    "eps_diluted": revenue * margin / 1e9,
                    "equity": equity,
                }
                for concept, value in values.items():
                    rows.append(
                        {
                            "ticker": ticker,
                            "concept": concept,
                            "period_end": period_end,
                            "fiscal_period": "I" if concept == "equity" else "Q",
                            "form": "10-Q",
                            "available_date": filed + pd.offsets.BDay(1),
                            "value": value,
                            "source": self.name,
                        }
                    )
        return pd.DataFrame(rows)
