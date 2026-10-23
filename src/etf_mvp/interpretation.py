from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from etf_mvp.forecast import NextSessionForecast

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
SECTION_TITLES = {
    "technical": "技术形态",
    "macro": "宏观面",
    "geopolitical": "地缘因素",
    "news": "短期新闻",
}


@dataclass(frozen=True)
class ModelFactor:
    feature: str
    importance_share: float
    local_contribution: float | None
    local_share: float | None
    direction: str


@dataclass(frozen=True)
class AttributionSection:
    key: str
    title: str
    attribution_pct: int
    impact_score: int
    direction: str
    analysis: str
    evidence: tuple[str, ...]
    model_factors: tuple[ModelFactor, ...]
    context_factors: tuple[str, ...]


@dataclass(frozen=True)
class InterpretationSource:
    title: str
    url: str


@dataclass(frozen=True)
class ForecastInterpretation:
    summary: str
    confidence: int
    sections: tuple[AttributionSection, ...]
    risk_notes: tuple[str, ...]
    sources: tuple[InterpretationSource, ...]
    model: str
    total_tokens: int | None


def load_local_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def deepseek_is_configured() -> bool:
    load_local_env()
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _finite(value: float | int | np.floating | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def build_forecast_context(
    forecast: NextSessionForecast,
    prices: pd.DataFrame,
    data_provider: str,
) -> dict[str, Any]:
    symbol_prices = (
        prices.loc[prices["symbol"] == forecast.symbol]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    close = symbol_prices["close"].astype(float)
    volume = symbol_prices["volume"].astype(float)
    change = close.pct_change()
    gains = change.clip(lower=0).rolling(14).mean()
    losses = -change.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    volume_zscore = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

    top_features = []
    if not forecast.feature_importance.empty:
        total_importance = float(forecast.feature_importance["importance"].sum())
        selected = [forecast.feature_importance.head(10)]
        for prefix in ("news__", "geo__", "macro__", "cb__"):
            selected.append(
                forecast.feature_importance.loc[
                    forecast.feature_importance["feature"].str.startswith(prefix)
                ].head(2)
            )
        context_features = (
            pd.concat(selected, ignore_index=True)
            .drop_duplicates("feature", keep="first")
            .head(18)
        )
        for row in context_features.itertuples(index=False):
            share = float(row.importance) / total_importance if total_importance else 0.0
            local_contribution = getattr(row, "local_contribution", None)
            local_share = getattr(row, "local_share", None)
            top_features.append(
                {
                    "feature": str(row.feature),
                    "importance_share": round(share, 4),
                    "local_contribution": _finite(local_contribution),
                    "local_share": _finite(local_share),
                }
            )

    return {
        "instrument": forecast.symbol,
        "data_provider": data_provider,
        "as_of_date": forecast.as_of_date.date().isoformat(),
        "forecast_date": forecast.forecast_date.date().isoformat(),
        "model_scope": {
            "feature_count": forecast.feature_count,
            "feature_kinds": list(forecast.feature_kinds),
            "description": (
                "次日OHLC回归模型直接使用已启用且按可用日期对齐的行情技术、宏观、"
                "央行、新闻情绪与地缘风险特征。"
            ),
        },
        "forecast": {
            "previous_close": round(forecast.last_close, 4),
            "open": round(forecast.predicted_open, 4),
            "high": round(forecast.predicted_high, 4),
            "low": round(forecast.predicted_low, 4),
            "close": round(forecast.predicted_close, 4),
            "expected_return": round(forecast.expected_return, 6),
            "close_interval": [
                round(forecast.close_lower, 4),
                round(forecast.close_upper, 4),
            ],
            "out_of_sample_direction_accuracy": round(
                forecast.direction_accuracy, 4
            ),
            "close_return_mae": round(forecast.close_return_mae, 6),
        },
        "technical_snapshot": {
            "return_1d": _finite(close.pct_change(1).iloc[-1]),
            "return_5d": _finite(close.pct_change(5).iloc[-1]),
            "return_20d": _finite(close.pct_change(20).iloc[-1]),
            "close_to_sma_20": _finite(close.iloc[-1] / close.tail(20).mean() - 1),
            "close_to_sma_50": _finite(close.iloc[-1] / close.tail(50).mean() - 1),
            "annualized_volatility_20d": _finite(
                np.log(close).diff().tail(20).std() * np.sqrt(252)
            ),
            "rsi_14": _finite(rsi.iloc[-1]),
            "volume_zscore_20d": _finite(volume_zscore.iloc[-1]),
        },
        "top_model_features": top_features,
    }


def _output_schema() -> dict[str, Any]:
    section_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "enum": list(SECTION_TITLES)},
            "attribution_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            "impact_score": {"type": "integer", "minimum": -100, "maximum": 100},
            "analysis": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "model_factors": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "context_factors": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
        },
        "required": [
            "key",
            "attribution_pct",
            "impact_score",
            "analysis",
            "evidence",
            "model_factors",
            "context_factors",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "sections": {
                "type": "array",
                "items": section_schema,
                "minItems": 4,
                "maxItems": 4,
            },
            "risk_notes": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["title", "url"],
                    "additionalProperties": False,
                },
                "maxItems": 8,
            },
        },
        "required": ["summary", "confidence", "sections", "risk_notes", "sources"],
        "additionalProperties": False,
    }


def _extract_output_text(response_payload: dict[str, Any]) -> str:
    text_parts = []
    for item in response_payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_parts.append(content["text"])
    if not text_parts:
        raise ValueError("DeepSeek returned no output text")
    return "".join(text_parts)


def _decode_json_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline >= 0 else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _normalize_weights(raw_weights: list[float]) -> list[int]:
    positive = [max(float(value), 0.0) for value in raw_weights]
    total = sum(positive)
    if total <= 0:
        return [25, 25, 25, 25]
    normalized = [round(value / total * 100) for value in positive]
    normalized[normalized.index(max(normalized))] += 100 - sum(normalized)
    return normalized


def _valid_source(raw_source: dict[str, Any]) -> InterpretationSource | None:
    url = str(raw_source.get("url", "")).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    title = str(raw_source.get("title", "")).strip() or parsed.netloc
    return InterpretationSource(title=title, url=url)


def _api_error(error: requests.RequestException) -> RuntimeError:
    detail = ""
    if error.response is not None:
        try:
            detail = str(error.response.json().get("error", {}).get("message", ""))
        except ValueError:
            detail = error.response.text[:300]
    return RuntimeError(f"DeepSeek API request failed: {detail or error}")


def _search_market_context(
    http_session: requests.Session,
    base_url: str,
    api_key: str,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    as_of = pd.Timestamp(context["as_of_date"])
    after = (as_of - pd.Timedelta(days=7)).date().isoformat()
    before = (as_of + pd.Timedelta(days=1)).date().isoformat()
    symbol = str(context["instrument"])
    queries = {
        "macro": (
            f"US Fed inflation Treasury yields economy after:{after} before:{before} lang:en"
        ),
        "geopolitical": (
            f"geopolitical risk global markets {symbol} after:{after} before:{before} lang:en"
        ),
        "news": f"{symbol} ETF market news after:{after} before:{before} lang:en",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    evidence = []
    for category, query in queries.items():
        try:
            response = http_session.post(
                f"{base_url}/v1/tools/search",
                headers=headers,
                json={
                    "query": query,
                    "count": 4,
                    "ori_question": f"分析{symbol}下一交易日的{category}影响",
                    "need_content": False,
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            if error.response is not None and error.response.status_code in {401, 402}:
                raise _api_error(error) from error
            continue

        for item in response.json().get("results", []):
            source = _valid_source(item)
            if not source:
                continue
            evidence.append(
                {
                    "category": category,
                    "title": source.title,
                    "url": source.url,
                    "date": item.get("date"),
                    "site_name": item.get("site_name"),
                    "authority": item.get("authority"),
                    "summary": str(item.get("summary") or item.get("snippet") or "")[:1200],
                }
            )
    return evidence


def parse_interpretation(
    payload: dict[str, Any],
    model: str,
    total_tokens: int | None = None,
    factor_metadata: dict[str, dict[str, Any]] | None = None,
) -> ForecastInterpretation:
    factor_metadata = factor_metadata or {}
    raw_by_key = {
        str(section.get("key")): section for section in payload.get("sections", [])
    }
    ordered_raw = [raw_by_key.get(key, {}) for key in SECTION_TITLES]
    weights = _normalize_weights(
        [float(section.get("attribution_pct", 0)) for section in ordered_raw]
    )
    sections = []
    for (key, title), raw_section, weight in zip(
        SECTION_TITLES.items(), ordered_raw, weights, strict=True
    ):
        impact_score = int(np.clip(raw_section.get("impact_score", 0), -100, 100))
        direction = "利多" if impact_score > 10 else "利空" if impact_score < -10 else "中性"
        evidence = tuple(
            str(item).strip()
            for item in raw_section.get("evidence", [])[:4]
            if str(item).strip()
        )
        requested_factors = [
            str(item).strip()
            for item in raw_section.get("model_factors", [])[:4]
            if str(item).strip() in factor_metadata
        ]
        if key == "technical" and not requested_factors:
            requested_factors = list(factor_metadata)[:3]
        model_factors = []
        for feature in requested_factors:
            metadata = factor_metadata[feature]
            contribution = metadata.get("local_contribution")
            local_share = metadata.get("local_share")
            factor_direction = (
                "推高预测"
                if contribution is not None and contribution > 0
                else "压低预测"
                if contribution is not None and contribution < 0
                else "方向不显著"
            )
            model_factors.append(
                ModelFactor(
                    feature=feature,
                    importance_share=float(metadata.get("importance_share") or 0),
                    local_contribution=(
                        float(contribution) if contribution is not None else None
                    ),
                    local_share=float(local_share) if local_share is not None else None,
                    direction=factor_direction,
                )
            )
        context_factors = tuple(
            str(item).strip()
            for item in raw_section.get("context_factors", [])[:4]
            if str(item).strip()
        )
        sections.append(
            AttributionSection(
                key=key,
                title=title,
                attribution_pct=weight,
                impact_score=impact_score,
                direction=direction,
                analysis=str(raw_section.get("analysis", "暂无充分证据。")),
                evidence=evidence,
                model_factors=tuple(model_factors),
                context_factors=context_factors,
            )
        )

    sources = []
    seen_urls = set()
    for raw_source in payload.get("sources", []):
        source = _valid_source(raw_source)
        if source and source.url not in seen_urls:
            seen_urls.add(source.url)
            sources.append(source)

    return ForecastInterpretation(
        summary=str(payload.get("summary", "暂无综合结论。")),
        confidence=int(np.clip(payload.get("confidence", 0), 0, 100)),
        sections=tuple(sections),
        risk_notes=tuple(
            str(item).strip()
            for item in payload.get("risk_notes", [])[:5]
            if str(item).strip()
        ),
        sources=tuple(sources),
        model=model,
        total_tokens=total_tokens,
    )


def interpret_forecast(
    context: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    session: requests.Session | None = None,
) -> ForecastInterpretation:
    load_local_env()
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_key:
        raise ValueError("DEEPSEEK_API_KEY is not configured")
    resolved_model = model or os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL).rstrip("/")
    web_search_enabled = os.getenv("DEEPSEEK_ENABLE_WEB_SEARCH", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    instructions = (
        "你是美股ETF量化研究解释器。请结合输入中的模型预测、技术指标和web_evidence，"
        "分析截至as_of_date可获得的宏观、地缘与最近7天新闻，并输出中文JSON。"
        "四类attribution_pct必须合计100，表示对当前预测的情境解释占比，不是因果或SHAP值；"
        "impact_score范围-100到100，正数利多、负数利空。"
        "必须明确区分模型直接特征与仅用于补充解释的外部信息。"
        "每个部分的model_factors只能逐字选择top_model_features中的feature，最多4个；"
        "新闻、宏观或地缘特征如果出现在top_model_features中，应明确视为直接模型特征；"
        "跨类别引用时才称为市场代理特征。"
        "context_factors列出该部分实际讨论的外部情境因子，最多4个。"
        "外部事实只能来自web_evidence，sources只能引用其中提供的URL。"
        "没有可靠证据时降低该部分占比并直说证据不足；禁止编造事件、日期、数字或来源。"
        "若data_provider为synthetic，必须提示行情与预测为演示数据。"
        "输出必须是合法JSON，字段结构严格遵循随后提供的JSON Schema。"
    )
    http_session = session or requests.Session()
    web_evidence = []
    if web_search_enabled:
        web_evidence = _search_market_context(
            http_session,
            base_url,
            resolved_key,
            context,
        )
    enriched_context = {**context, "web_evidence": web_evidence}
    request_payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{instructions}\nJSON Schema:\n"
                    f"{json.dumps(_output_schema(), ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(enriched_context, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "max_tokens": 3000,
        "stream": False,
    }
    try:
        response = http_session.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise _api_error(error) from error

    response_payload = response.json()
    choices = response_payload.get("choices", [])
    if not choices or not choices[0].get("message", {}).get("content"):
        raise ValueError("DeepSeek returned no output text")
    parsed_output = _decode_json_output(choices[0]["message"]["content"])
    usage = response_payload.get("usage", {})
    result = parse_interpretation(
        parsed_output,
        model=resolved_model,
        total_tokens=usage.get("total_tokens"),
        factor_metadata={
            item["feature"]: item for item in context.get("top_model_features", [])
        },
    )
    verified_sources = {
        item["url"]: InterpretationSource(title=item["title"], url=item["url"])
        for item in web_evidence
    }
    selected_sources = tuple(
        verified_sources[source.url]
        for source in result.sources
        if source.url in verified_sources
    )
    if not selected_sources:
        selected_sources = tuple(list(verified_sources.values())[:8])
    return replace(result, sources=selected_sources)
