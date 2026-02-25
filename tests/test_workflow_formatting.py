from weekly_seo_agent.workflow import _normalize_ai_commentary_markdown


def test_normalize_ai_commentary_promotes_headings_and_bullets() -> None:
    raw = """
Detailed Supporting Analysis
Seasonality has impact.
Priority Actions for This Week
Action one
Action two
""".strip()

    normalized = _normalize_ai_commentary_markdown(raw)

    assert "### Detailed Supporting Analysis" in normalized
    assert "### Priority Actions for This Week" in normalized
    assert "- Seasonality has impact." in normalized
    assert "Action one" in normalized
    assert "Action two" in normalized


def test_normalize_ai_commentary_uses_page_name_and_tones_down_language() -> None:
    raw = """
### Segment Interpretation
- Page-template performance: Offer pages saw a dramatic increase (+560.35%).
""".strip()

    normalized = _normalize_ai_commentary_markdown(raw)

    assert "Page-template" not in normalized
    assert "page-template" not in normalized
    assert "Page Name" in normalized or "page name" in normalized
    assert "dramatic" not in normalized.lower()
    assert "previous-click baseline" in normalized
