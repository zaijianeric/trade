"""Data layer: sources grouped by kind, point-in-time schemas, storage and collection.

kinds.py     DataKind + normalized schema per kind (available_date = point in time)
base.py      DataSource / FetchRequest contract
sources/     one module per kind: market, macro, fundamental, news, central_bank
registry.py  provider names -> source classes
store.py     Parquet per (kind, provider) + DuckDB warehouse
service.py   collect() / load_bundle() -> DataBundle
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from etf_mvp.data.kinds import DataKind

_LAZY = {
    "DataBundle": "etf_mvp.data.service",
    "collect": "etf_mvp.data.service",
    "load_bundle": "etf_mvp.data.service",
    "available_providers": "etf_mvp.data.registry",
    "create_source": "etf_mvp.data.registry",
}

__all__ = ["DataKind", *_LAZY]


def __getattr__(name: str) -> Any:
    # Lazy so that etf_mvp.config can import DataKind without a circular import.
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(name)
