"""Persistence: one Parquet file per (kind, provider) plus a DuckDB warehouse."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from etf_mvp.data.kinds import SCHEMAS, DataKind, normalize


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)


class DataStore:
    """Layout::

    data/market/prices.parquet          normalized prices (kept for the dashboard)
    data/raw/<kind>/<provider>.parquet  every other kind, point-in-time
    data/warehouse.duckdb               one table per kind + model_features
    """

    def __init__(self, raw_dir: Path, prices_path: Path, warehouse_path: Path) -> None:
        self.raw_dir = raw_dir
        self.prices_path = prices_path
        self.warehouse_path = warehouse_path

    def path(self, kind: DataKind, provider: str) -> Path:
        if kind is DataKind.MARKET:
            return self.prices_path
        return self.raw_dir / kind.value / f"{provider}.parquet"

    def exists(self, kind: DataKind, provider: str) -> bool:
        return self.path(kind, provider).exists()

    def read(self, kind: DataKind, provider: str) -> pd.DataFrame:
        frame = read_parquet(self.path(kind, provider))
        return normalize(kind, frame) if not frame.empty else frame

    def write(
        self, kind: DataKind, provider: str, frame: pd.DataFrame, merge: bool
    ) -> pd.DataFrame:
        """Write a kind's data; with ``merge`` new rows are upserted into what is stored."""
        if merge and self.exists(kind, provider):
            frame = normalize(kind, pd.concat([self.read(kind, provider), frame]))
        write_parquet(frame, self.path(kind, provider))
        return frame

    def refresh_warehouse(
        self,
        frames: dict[DataKind, pd.DataFrame],
        features: pd.DataFrame | None = None,
    ) -> None:
        self.warehouse_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.warehouse_path)) as connection:
            for kind, frame in frames.items():
                table = SCHEMAS[kind].table
                connection.register("incoming", frame)
                connection.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM incoming")
                connection.unregister("incoming")
            if DataKind.MARKET in frames:
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS prices_symbol_date ON market_prices(symbol, date)"
                )
            if features is not None:
                connection.register("incoming", features)
                connection.execute(
                    "CREATE OR REPLACE TABLE model_features AS SELECT * FROM incoming"
                )
                connection.unregister("incoming")
