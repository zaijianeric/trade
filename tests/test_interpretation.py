from etf_mvp.interpretation import _decode_json_output, parse_interpretation


def test_decode_json_output_accepts_markdown_fence() -> None:
    assert _decode_json_output('```json\n{"summary": "ok"}\n```') == {"summary": "ok"}


def test_interpretation_normalizes_attribution_and_direction() -> None:
    payload = {
        "summary": "整体偏多，但不确定性较高。",
        "confidence": 62,
        "sections": [
            {
                "key": "technical",
                "attribution_pct": 7,
                "impact_score": 35,
                "analysis": "趋势改善。",
                "evidence": ["价格站上均线"],
                "model_factors": ["SPY__return_5d", "unknown"],
                "context_factors": ["短期动量"],
            },
            {
                "key": "macro",
                "attribution_pct": 2,
                "impact_score": -20,
                "analysis": "利率构成压力。",
                "evidence": [],
                "model_factors": ["TLT__return_5d"],
                "context_factors": ["美债收益率"],
            },
            {
                "key": "geopolitical",
                "attribution_pct": 1,
                "impact_score": 3,
                "analysis": "影响有限。",
                "evidence": [],
                "model_factors": [],
                "context_factors": [],
            },
            {
                "key": "news",
                "attribution_pct": 0,
                "impact_score": 0,
                "analysis": "暂无新闻。",
                "evidence": [],
                "model_factors": [],
                "context_factors": [],
            },
        ],
        "risk_notes": ["这是研究输出。"],
        "sources": [
            {"title": "Valid", "url": "https://example.com/report"},
            {"title": "Invalid", "url": "javascript:alert(1)"},
        ],
    }

    result = parse_interpretation(
        payload,
        model="deepseek-v4-pro",
        total_tokens=321,
        factor_metadata={
            "SPY__return_5d": {
                "importance_share": 0.3,
                "local_contribution": 0.002,
                "local_share": 0.2,
            },
            "TLT__return_5d": {
                "importance_share": 0.2,
                "local_contribution": -0.001,
                "local_share": 0.1,
            },
        },
    )

    assert sum(section.attribution_pct for section in result.sections) == 100
    assert [section.direction for section in result.sections] == [
        "利多",
        "利空",
        "中性",
        "中性",
    ]
    assert len(result.sources) == 1
    assert result.total_tokens == 321
    assert result.sections[0].model_factors[0].feature == "SPY__return_5d"
    assert result.sections[0].model_factors[0].direction == "推高预测"
    assert result.sections[1].model_factors[0].direction == "压低预测"
    assert result.sections[0].context_factors == ("短期动量",)
