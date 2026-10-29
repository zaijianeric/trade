"""Which providers exist for each data kind, and how to build them."""

from __future__ import annotations

from collections.abc import Callable

from etf_mvp.data.base import DataSource
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.sources.central_bank import FredFomcSource, SyntheticCentralBankSource
from etf_mvp.data.sources.fundamental import (
    SecEdgarFundamentalSource,
    SyntheticFundamentalSource,
)
from etf_mvp.data.sources.geopolitical import (
    GprGeopoliticalSource,
    SyntheticGeopoliticalSource,
)
from etf_mvp.data.sources.macro import FredMacroSource, SyntheticMacroSource
from etf_mvp.data.sources.market import (
    AlpacaMarketSource,
    NasdaqMarketSource,
    SyntheticMarketSource,
    TiingoMarketSource,
)
from etf_mvp.data.sources.news import (
    AlphaVantageNewsSource,
    GdeltNewsSource,
    GoogleNewsRssSource,
    HybridNewsSource,
    SfFedNewsSource,
    SyntheticNewsSource,
)

Factory = Callable[[int], DataSource]

REGISTRY: dict[DataKind, dict[str, Factory]] = {
    DataKind.MARKET: {
        "alpaca": lambda seed: AlpacaMarketSource(),
        "nasdaq": lambda seed: NasdaqMarketSource(),
        "synthetic": lambda seed: SyntheticMarketSource(seed),
        "tiingo": lambda seed: TiingoMarketSource(),
    },
    DataKind.MACRO: {
        "synthetic": lambda seed: SyntheticMacroSource(seed),
        "fred": lambda seed: FredMacroSource(),
    },
    DataKind.FUNDAMENTAL: {
        "synthetic": lambda seed: SyntheticFundamentalSource(seed),
        "sec_edgar": lambda seed: SecEdgarFundamentalSource(),
    },
    DataKind.NEWS: {
        "synthetic": lambda seed: SyntheticNewsSource(seed),
        "alphavantage": lambda seed: AlphaVantageNewsSource(),
        "gdelt": lambda seed: GdeltNewsSource(),
        "google_news": lambda seed: GoogleNewsRssSource(),
        "hybrid": lambda seed: HybridNewsSource(),
        "sf_fed": lambda seed: SfFedNewsSource(),
    },
    DataKind.GEOPOLITICAL: {
        "synthetic": lambda seed: SyntheticGeopoliticalSource(seed),
        "gpr": lambda seed: GprGeopoliticalSource(),
    },
    DataKind.CENTRAL_BANK: {
        "synthetic": lambda seed: SyntheticCentralBankSource(seed),
        "fred": lambda seed: FredFomcSource(),
    },
}


def available_providers(kind: DataKind) -> list[str]:
    return sorted(REGISTRY[kind])


def create_source(kind: DataKind | str, provider: str, random_seed: int = 42) -> DataSource:
    kind = DataKind(kind)
    normalized = provider.strip().lower()
    try:
        factory = REGISTRY[kind][normalized]
    except KeyError:
        raise ValueError(
            f"Unknown {kind} provider '{provider}'. Choose from {available_providers(kind)}"
        ) from None
    return factory(random_seed)
