"""Collection entry points: fetch each enabled kind, persist it, load it back."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from etf_mvp.config import AppConfig
from etf_mvp.data.base import FetchRequest
from etf_mvp.data.kinds import DataKind, empty_frame
from etf_mvp.data.registry import create_source
from etf_mvp.data.store import DataStore

logger = logging.getLogger(__name__)
INCREMENTAL_OVERLAP_DAYS = 3


@dataclass
class DataBundle:
    """Everything the feature layer needs, one normalized frame per kind."""

    frames: dict[DataKind, pd.DataFrame] = field(default_factory=dict)

    def get(self, kind: DataKind) -> pd.DataFrame:
        return self.frames.get(kind, empty_frame(kind))

    @property
    def prices(self) -> pd.DataFrame:
        return self.get(DataKind.MARKET)


def store_for(config: AppConfig) -> DataStore:
    return DataStore(
        raw_dir=config.data.raw_root,
        prices_path=config.data.prices_path,
        warehouse_path=config.data.warehouse_path,
    )


def enabled_kinds(config: AppConfig) -> list[DataKind]:
    """Enabled kinds; synthetic alternative data is never mixed with real prices."""
    real_market = config.source(DataKind.MARKET).provider != "synthetic"
    kinds = []
    for kind in DataKind:
        source = config.source(kind)
        if not source.enabled:
            continue
        if real_market and kind is not DataKind.MARKET and source.provider == "synthetic":
            logger.warning("Skipping synthetic %s data because market data is real", kind)
            continue
        kinds.append(kind)
    return kinds


def _request(config: AppConfig, kind: DataKind, store: DataStore) -> FetchRequest:
    source_config = config.source(kind)
    start = config.data.start
    if source_config.incremental and store.exists(kind, source_config.provider):
        stored = store.read(kind, source_config.provider)
        if not stored.empty:
            latest = stored["available_date"].max() - pd.Timedelta(days=INCREMENTAL_OVERLAP_DAYS)
            start = max(pd.Timestamp(start), latest).date().isoformat()
    return FetchRequest(
        start=start,
        end=config.data.end,
        symbols=config.universe.symbols,
        options=dict(source_config.options),
    )


def collect(config: AppConfig, kinds: list[DataKind] | None = None) -> DataBundle:
    """Fetch the given kinds (default: all enabled), persist them and return the bundle."""
    store = store_for(config)
    selected = kinds or enabled_kinds(config)
    bundle = DataBundle()
    for kind in selected:
        source_config = config.source(kind)
        source = create_source(kind, source_config.provider, config.model.random_seed)
        logger.info("Fetching %s from %s", kind, source.name)
        frame = source.fetch(_request(config, kind, store))
        bundle.frames[kind] = store.write(
            kind, source_config.provider, frame, merge=source_config.incremental
        )
    # Kinds not fetched this time still come from storage when available.
    for kind in enabled_kinds(config):
        if kind not in bundle.frames and store.exists(kind, config.source(kind).provider):
            bundle.frames[kind] = store.read(kind, config.source(kind).provider)
    store.refresh_warehouse(bundle.frames)
    return bundle


def load_bundle(
    config: AppConfig, require: tuple[DataKind, ...] = (DataKind.MARKET,)
) -> DataBundle:
    """Read every enabled kind from storage; missing optional kinds are skipped."""
    store = store_for(config)
    bundle = DataBundle()
    for kind in enabled_kinds(config):
        provider = config.source(kind).provider
        if store.exists(kind, provider):
            bundle.frames[kind] = store.read(kind, provider)
        elif kind in require:
            raise FileNotFoundError(f"No stored {kind} data from '{provider}'; run fetch first")
        else:
            logger.warning("No stored %s data from '%s'; its features are skipped", kind, provider)
    return bundle
