import re

from weekly_seo_agent.reporting import enforce_manager_quality_guardrail


def _section_slice(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"{re.escape(heading)}\n(.*?)(?=\n## |\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


def test_guardrail_keeps_narrative_flow_non_empty_when_compacting() -> None:
    filler = "\n".join(
        "- Long context line for compression behavior in readability mode."
        for _ in range(120)
    )
    raw = (
        "# Weekly SEO Intelligence Report (2026-02-26 | PL)\n\n"
        "## Executive summary\n"
        f"{filler}\n\n"
        "## What is happening and why\n"
        "### Narrative Flow\n"
        "- Demand shifted between categories while CTR stayed stable.\n"
        "### Causal Chain\n"
        "- Working hypothesis: demand timing and routing explain movement. Confidence: 72/100.\n"
        "### Evidence by Source\n"
        "- KPI and segment evidence support this interpretation.\n"
        "### Priority Actions for This Week\n"
        "- [SEO Team | next run] Validate the hypothesis on refreshed segment data.\n\n"
        "## Hypothesis protocol\n"
        "- Hypothesis fields: falsifier | validation metric | validation date.\n"
    )

    guarded = enforce_manager_quality_guardrail(raw, max_words=120)
    narrative = _section_slice(guarded, "## What is happening and why")

    assert "### Narrative Flow" in narrative
    assert re.search(r"### Narrative Flow\n- .+", narrative)


def test_guardrail_inserts_fallback_when_narrative_subsection_is_empty() -> None:
    raw = (
        "# Weekly SEO Intelligence Report (2026-02-26 | PL)\n\n"
        "## What is happening and why\n"
        "### Narrative Flow\n\n"
        "### Causal Chain\n\n"
        "### Evidence by Source\n\n"
        "### Priority Actions for This Week\n\n"
        "## Hypothesis protocol\n"
        "- Hypothesis fields: falsifier | validation metric | validation date.\n"
    )

    guarded = enforce_manager_quality_guardrail(raw, max_words=600)
    narrative = _section_slice(guarded, "## What is happening and why")

    assert re.search(r"### Narrative Flow\n- .+", narrative)
    assert re.search(r"### Causal Chain\n- .+", narrative)
    assert re.search(r"### Evidence by Source\n- .+", narrative)
    assert re.search(r"### Priority Actions for This Week\n- .+", narrative)
