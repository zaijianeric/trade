"""Parsers for the real providers, tested against payloads shaped like the live APIs."""

from io import BytesIO

import pandas as pd
import pytest

from etf_mvp.data.kinds import DataKind, normalize
from etf_mvp.data.sources.central_bank import build_fomc_events, load_meetings
from etf_mvp.data.sources.fundamental import parse_company_facts
from etf_mvp.data.sources.geopolitical import parse_gpr_daily
from etf_mvp.data.sources.macro import MacroSeries, parse_fred_observations
from etf_mvp.data.sources.market import parse_nasdaq_prices
from etf_mvp.data.sources.news import (
    parse_news_feed,
    parse_sf_fed_sentiment,
    topic_slug,
)


def test_nasdaq_market_rows_parse_currency_and_volume() -> None:
    rows = [
        {
            "date": "10/21/2026",
            "close": "$773.09",
            "volume": "31,943,810",
            "open": "771.17",
            "high": "774.44",
            "low": "768.40",
        }
    ]

    frame = parse_nasdaq_prices(rows, "SPY")

    assert frame.loc[0, "date"] == pd.Timestamp("2026-10-21")
    assert frame.loc[0, "symbol"] == "SPY"
    assert frame.loc[0, "close"] == pytest.approx(773.09)
    assert frame.loc[0, "volume"] == 31_943_810


def test_sf_fed_sentiment_uses_next_business_day() -> None:
    text = "date,Daily news sentiment\n10/16/2026,-0.25\n10/17/2026,-0.10\n"

    frame = parse_sf_fed_sentiment(text, "2026-10-01", "2026-10-31")

    assert len(frame) == 1
    assert frame.loc[0, "sentiment"] == pytest.approx(-0.25)
    assert frame.loc[0, "available_date"] == pd.Timestamp("2026-10-23")


def test_gpr_daily_uses_conservative_publication_lag() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-10"]),
            "GPRD": [150.0],
            "GPRD_THREAT": [100.0],
            "GPRD_ACT": [50.0],
        }
    )
    buffer = BytesIO()
    raw.to_stata(buffer, write_index=False)
    frame = parse_gpr_daily(buffer.getvalue(), "2026-10-01", "2026-10-31")

    assert frame.loc[0, "available_date"] == pd.Timestamp("2026-10-17")
    assert frame.loc[0, "gpr"] == pytest.approx(150.0)


def test_fred_daily_series_uses_publication_lag() -> None:
    payload = {
        "observations": [
            {"date": "2024-01-05", "value": "4.05"},
            {"date": "2024-01-08", "value": "."},
        ]
    }
    frame = parse_fred_observations(payload, MacroSeries("DGS10", lag_days=1))
    assert frame["available_date"].iloc[0] == pd.Timestamp("2024-01-08")
    assert frame["value"].isna().iloc[1]


def test_fred_vintage_keeps_first_release() -> None:
    payload = {
        "observations": [
            {
                "date": "2024-01-01",
                "value": "3.4",
                "realtime_start": "2024-02-13",
                "realtime_end": "2024-03-11",
            },
            {
                "date": "2024-01-01",
                "value": "3.1",
                "realtime_start": "2024-03-12",
                "realtime_end": "9999-12-31",
            },
        ]
    }
    frame = parse_fred_observations(payload, MacroSeries("CPIAUCSL", vintage=True))
    assert len(frame) == 1
    assert frame["value"].iloc[0] == 3.4
    assert frame["available_date"].iloc[0] == pd.Timestamp("2024-02-13")


def test_sec_facts_keep_first_filing_and_classify_periods() -> None:
    revenue = [
        # Q1 2024, first filed in May, repeated as a comparative a year later
        {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "val": 100,
            "form": "10-Q",
            "filed": "2024-05-02",
        },
        {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "val": 101,
            "form": "10-Q",
            "filed": "2025-05-01",
        },
        # six-month year-to-date figure: neither a quarter nor a year, dropped
        {
            "start": "2024-01-01",
            "end": "2024-06-30",
            "val": 210,
            "form": "10-Q",
            "filed": "2024-08-01",
        },
        {
            "start": "2023-01-01",
            "end": "2023-12-31",
            "val": 400,
            "form": "10-K",
            "filed": "2024-02-20",
        },
    ]
    equity = [{"end": "2024-03-31", "val": 900, "form": "10-Q", "filed": "2024-05-02"}]
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": revenue}},
                "StockholdersEquity": {"units": {"USD": equity}},
            }
        }
    }
    frame = normalize(DataKind.FUNDAMENTAL, parse_company_facts(payload, "ACME"))
    quarter = frame.loc[(frame["concept"] == "revenue") & (frame["fiscal_period"] == "Q")]
    assert quarter["value"].tolist() == [100]
    assert quarter["available_date"].iloc[0] == pd.Timestamp("2024-05-03")  # next session
    assert set(frame["fiscal_period"]) == {"Q", "FY", "I"}


def test_news_feed_parsing_and_after_close_rollover() -> None:
    payload = {
        "feed": [
            {
                "title": "Fed holds",
                "url": "https://x/1",
                "source": "Wire",
                "time_published": "20240131T143000",
                "overall_sentiment_score": 0.2,
                "topics": [{"topic": "Economy - Monetary", "relevance_score": "0.9"}],
                "ticker_sentiment": [{"ticker": "SPY"}],
            },
            {
                "title": "Late story",
                "url": "https://x/2",
                "source": "Wire",
                "time_published": "20240202T183000",  # Friday after the close
                "overall_sentiment_score": -0.1,
                "topics": [{"topic": "Energy & Transportation"}],
            },
        ]
    }
    frame = parse_news_feed(payload)
    assert frame["topics"].tolist() == ["economy_monetary", "energy_transportation"]
    assert frame["available_date"].tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-05"),
    ]
    assert frame["tickers"].iloc[0] == "SPY"


def test_news_rate_limit_message_raises() -> None:
    with pytest.raises(RuntimeError):
        parse_news_feed({"Information": "rate limit reached"})


def test_topic_slug() -> None:
    assert topic_slug("Financial Markets") == "financial_markets"


def _daily(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.to_datetime(list(values)), dtype=float)


def test_fomc_decision_from_fred_target_range_and_two_year_move() -> None:
    meetings = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2025-09-17", "2025-10-29"]),
            "is_scheduled": [True, True],
        }
    )
    # Cut effective the day after the meeting; 2Y fell 3bp on the day.
    upper = _daily({"2025-09-16": 4.5, "2025-09-17": 4.5, "2025-09-18": 4.25})
    lower = _daily({"2025-09-16": 4.25, "2025-09-17": 4.25, "2025-09-18": 4.0})
    two_year = _daily({"2025-09-16": 3.55, "2025-09-17": 3.52})
    events = build_fomc_events(meetings, upper, lower, two_year)

    decided, pending = events.iloc[0], events.iloc[1]
    assert (decided["rate_lower"], decided["rate_upper"]) == (4.0, 4.25)
    assert decided["hawkish_score"] == pytest.approx(-3.0)
    assert decided["available_date"] == pd.Timestamp("2025-09-18")
    assert pd.isna(pending["rate_upper"]) and pd.isna(pending["hawkish_score"])
    assert pending["available_date"] == pd.Timestamp("2025-10-29")


def test_weekend_decision_uses_next_session_reaction() -> None:
    meetings = pd.DataFrame({"event_date": [pd.Timestamp("2020-03-15")], "is_scheduled": [False]})
    upper = _daily({"2020-03-15": 1.25, "2020-03-16": 0.25})
    lower = _daily({"2020-03-15": 1.0, "2020-03-16": 0.0})
    two_year = _daily({"2020-03-13": 0.49, "2020-03-16": 0.36})
    event = build_fomc_events(meetings, upper, lower, two_year).iloc[0]
    assert event["rate_upper"] == 0.25
    assert event["hawkish_score"] == pytest.approx(-13.0)
    assert event["available_date"] == pd.Timestamp("2020-03-17")


def test_bundled_meeting_calendar(tmp_path) -> None:
    meetings = load_meetings()
    per_year = meetings.loc[meetings["is_scheduled"]].groupby(meetings["event_date"].dt.year)
    assert per_year.size().eq(8).drop(2020).all()  # 2020's March meeting became two calls
    assert pd.Timestamp("2017-11-01") in set(meetings["event_date"])

    extra = tmp_path / "extra.csv"
    extra.write_text("event_date,is_scheduled\n2028-01-26,true\n")
    assert load_meetings(str(extra))["event_date"].max() == pd.Timestamp("2028-01-26")
