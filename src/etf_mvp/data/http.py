from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "us-etf-predictor/0.2 (research)"


class HttpClient:
    """Shared session with retries, a timeout and a minimum interval between calls."""

    def __init__(
        self,
        user_agent: str | None = DEFAULT_USER_AGENT,
        timeout_seconds: int = 30,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.min_interval_seconds = min_interval_seconds
        self._last_call = 0.0
        self.session = requests.Session()
        if user_agent:  # None keeps the requests default
            self.session.headers["User-Agent"] = user_agent
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get(self, url: str, **kwargs) -> requests.Response:
        wait = self.min_interval_seconds - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        response = self.session.get(url, timeout=self.timeout_seconds, **kwargs)
        self._last_call = time.monotonic()
        response.raise_for_status()
        return response
