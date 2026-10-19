from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from etf_mvp.data.kinds import DataKind


@dataclass(frozen=True)
class UniverseConfig:
    symbols: tuple[str, ...]
    targets: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    provider: str  # market provider; kept here for backward compatibility
    start: str
    end: str
    prices_path: Path
    features_path: Path
    warehouse_path: Path
    raw_dir: Path | None = None

    @property
    def raw_root(self) -> Path:
        return self.raw_dir or self.prices_path.parent.parent / "raw"


@dataclass(frozen=True)
class SourceConfig:
    """One data kind: whether to use it, which provider, and kind-specific options."""

    enabled: bool = True
    provider: str = "synthetic"
    incremental: bool = False
    options: dict[str, Any] = field(default_factory=dict)


DEFAULT_MACRO_SERIES = (
    {"id": "DGS10"},
    {"id": "DGS2"},
    {"id": "T10Y2Y"},
    {"id": "VIXCLS"},
    {"id": "BAMLH0A0HYM2"},
    {"id": "DFF"},
    {"id": "CPIAUCSL", "vintage": True},
    {"id": "UNRATE", "vintage": True},
    {"id": "ICSA", "vintage": True},
)


def default_sources() -> dict[DataKind, SourceConfig]:
    return {
        DataKind.MACRO: SourceConfig(options={"series": list(DEFAULT_MACRO_SERIES)}),
        DataKind.FUNDAMENTAL: SourceConfig(
            options={"holdings": {"SPY": ["AAPL", "MSFT", "AMZN"], "QQQ": ["AAPL", "MSFT"]}}
        ),
        DataKind.NEWS: SourceConfig(
            incremental=True,
            options={"topics": ["financial_markets", "economy_monetary", "economy_macro"]},
        ),
        DataKind.CENTRAL_BANK: SourceConfig(),
    }


@dataclass(frozen=True)
class ModelConfig:
    horizon_days: int
    test_fraction: float
    probability_threshold: float
    random_seed: int


@dataclass(frozen=True)
class OutputConfig:
    artifacts_dir: Path


@dataclass(frozen=True)
class AppConfig:
    universe: UniverseConfig
    data: DataConfig
    model: ModelConfig
    output: OutputConfig
    sources: dict[DataKind, SourceConfig] = field(default_factory=default_sources)

    def source(self, kind: DataKind) -> SourceConfig:
        if kind is DataKind.MARKET:
            return SourceConfig(provider=self.data.provider)
        return self.sources.get(kind, SourceConfig(enabled=False))

    def with_providers(self, overrides: dict[str, str]) -> AppConfig:
        """Return a copy with providers replaced, e.g. {"market": "tiingo", "macro": "fred"}."""
        config = self
        for raw_kind, provider in overrides.items():
            kind = DataKind(raw_kind)
            if kind is DataKind.MARKET:
                config = replace(config, data=replace(config.data, provider=provider))
            else:
                sources = dict(config.sources)
                sources[kind] = replace(config.source(kind), provider=provider, enabled=True)
                config = replace(config, sources=sources)
        return config


def _resolve(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


PATH_OPTIONS = ("meetings_csv",)


def _load_sources(raw: dict[str, Any], project_root: Path) -> dict[DataKind, SourceConfig]:
    sources = default_sources()
    for name, section in raw.items():
        kind = DataKind(name)
        if kind is DataKind.MARKET:
            continue
        options = {
            key: value
            for key, value in section.items()
            if key not in {"enabled", "provider", "incremental"}
        }
        for key in PATH_OPTIONS:
            if key in options:
                options[key] = str(_resolve(project_root, options[key]))
        base = sources.get(kind, SourceConfig())
        sources[kind] = SourceConfig(
            enabled=bool(section.get("enabled", base.enabled)),
            provider=str(section.get("provider", base.provider)),
            incremental=bool(section.get("incremental", base.incremental)),
            options=options or base.options,
        )
    return sources


def load_config(path: str | Path = "configs/default.toml") -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    project_root = config_path.parent.parent
    data = raw["data"]
    end = data.get("end") or date.today().isoformat()
    universe = raw["universe"]
    targets = tuple(universe["targets"])
    symbols = tuple(universe["symbols"])
    unknown_targets = sorted(set(targets) - set(symbols))
    if unknown_targets:
        raise ValueError(f"Targets must be included in symbols: {unknown_targets}")

    raw_sources = raw.get("sources", {})
    market_provider = raw_sources.get("market", {}).get("provider", data.get("provider"))

    return AppConfig(
        universe=UniverseConfig(symbols=symbols, targets=targets),
        data=DataConfig(
            provider=market_provider or "synthetic",
            start=data["start"],
            end=end,
            prices_path=_resolve(project_root, data["prices_path"]),
            features_path=_resolve(project_root, data["features_path"]),
            warehouse_path=_resolve(project_root, data["warehouse_path"]),
            raw_dir=_resolve(project_root, data.get("raw_dir", "data/raw")),
        ),
        model=ModelConfig(**raw["model"]),
        output=OutputConfig(artifacts_dir=_resolve(project_root, raw["output"]["artifacts_dir"])),
        sources=_load_sources(raw_sources, project_root),
    )
