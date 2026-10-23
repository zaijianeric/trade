from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from etf_mvp.data.kinds import DataKind, normalize


@dataclass(frozen=True)
class FetchRequest:
    """What to collect. ``options`` carries kind-specific settings from config."""

    start: str
    end: str
    symbols: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """One provider of one kind of data. Subclasses set ``kind`` and ``name``."""

    kind: ClassVar[DataKind]
    name: ClassVar[str]
    requires_network: ClassVar[bool] = True

    def fetch(self, request: FetchRequest) -> pd.DataFrame:
        frame = self._fetch(request)
        normalized = normalize(self.kind, frame)
        if "source" in normalized.columns:
            normalized["source"] = self.name
        return normalized

    @abstractmethod
    def _fetch(self, request: FetchRequest) -> pd.DataFrame:
        """Return rows in (or coercible to) the kind's schema."""
