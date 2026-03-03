import re

from weekly_seo_agent.workflow import (
    _compose_final_report,
    _deduplicate_commentary_lines,
    _normalize_ai_commentary_markdown,
)


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


def test_normalize_ai_commentary_removes_placeholder_bullets() -> None:
    raw = """
### Narrative Flow
-

### Causal Chain
- -

### Evidence by Source
-
""".strip()

    normalized = _normalize_ai_commentary_markdown(raw)

    assert "- -" not in normalized
    assert re.search(r"### Narrative Flow\n- ", normalized)
    assert re.search(r"### Causal Chain\n- ", normalized)


def test_dedup_then_normalize_restores_required_section_content() -> None:
    raw = """
### Narrative Flow
- KPI moved with stable CTR and position.

### Causal Chain
- KPI moved with stable CTR and position. Confidence: 70/100.

### Evidence by Source
- KPI moved with stable CTR and position.
""".strip()

    deduped = _deduplicate_commentary_lines(
        raw,
        reference_sections=("- KPI moved with stable CTR and position.",),
    )
    normalized = _normalize_ai_commentary_markdown(deduped)

    assert "### Narrative Flow" in normalized
    assert "### Causal Chain" in normalized
    assert re.search(r"### Narrative Flow\n- ", normalized)
    assert re.search(r"### Causal Chain\n- ", normalized)


def test_compose_final_report_adds_data_backed_serp_lens_section() -> None:
    markdown_report = """
# Weekly SEO Intelligence Report (2026-03-02 | PL)

## Executive summary
- **What changed**: clicks 12.2M.

## What is happening and why
- SERP layout changed across result types (WoW/MoM/YoY): WoW up: Forums (+2); down: Product snippets (-277.2k).
- Impact chain on organic KPIs: listing-surface shifts can reallocate impressions and move CTR/position.

## Hypothesis protocol
| Hypothesis | Priority | Confidence | Falsifier | Validation metric | Validation date |
|---|---|---|---|---|---|
| Sample | Medium | 70/100 | - | - | 2026-03-09 |

## Governance and provenance
- Report date: 2026-03-02 | Country: PL.

## Evidence ledger
| ID | Source | Date | Note |
|---|---|---|---|
| E1 | GSC weekly window | 2026-02-23..2026-03-01 | Primary KPI baseline window. |
""".strip()

    commentary = """
### Narrative Flow
- What changed: clicks moved down.

### Causal Chain
- Working hypothesis: listing-surface reallocation can affect CTR. Confidence: 70/100.

### Evidence by Source
- Decision link: KPI evidence supports this week prioritization.

### Priority Actions for This Week
- [SEO Team | next run] Validate listing-surface deltas on refreshed data.

### Risks and Monitoring
- Risk: keep monitoring CTR/position.

### Continuity Check
- Compare with previous run.

### Further Analysis Flags
- Validate top drivers.
""".strip()

    final_report = _compose_final_report(markdown_report, commentary)

    assert "## What is happening and why" in final_report
    assert "### Data-backed SERP/CTR lens" in final_report
    assert "SERP layout changed across result types (WoW/MoM/YoY)" in final_report


def test_compose_final_report_keeps_case_study_line_in_serp_section() -> None:
    markdown_report = """
# Weekly SEO Intelligence Report (2026-03-02 | PL)

## Executive summary
- **What changed**: clicks 12.2M.

## Context snapshot
- **External case-study context (13 months)**: SERP case-study scanner (13M): recurring external patterns include CTR and feature-layout shifts.

## What is happening and why
- Search-results appearance mix changed (WoW/MoM/YoY): WoW up: Forums (+2); down: Product snippets (-277.2k).

## Hypothesis protocol
| Hypothesis | Priority | Confidence | Falsifier | Validation metric | Validation date |
|---|---|---|---|---|---|
| Sample | Medium | 70/100 | - | - | 2026-03-09 |

## Governance and provenance
- Report date: 2026-03-02 | Country: PL.

## Evidence ledger
| ID | Source | Date | Note |
|---|---|---|---|
| E1 | GSC weekly window | 2026-02-23..2026-03-01 | Primary KPI baseline window. |
""".strip()

    commentary = """
### Narrative Flow
- What changed: clicks moved down.

### Causal Chain
- Working hypothesis: listing-surface reallocation can affect CTR. Confidence: 70/100.

### Evidence by Source
- Decision link: KPI evidence supports this week prioritization.

### Priority Actions for This Week
- [SEO Team | next run] Validate listing-surface deltas on refreshed data.

### Risks and Monitoring
- Risk: keep monitoring CTR/position.

### Continuity Check
- Compare with previous run.

### Further Analysis Flags
- Validate top drivers.
""".strip()

    final_report = _compose_final_report(markdown_report, commentary)

    assert "## SERP and listings impact (WoW/MoM/YoY)" in final_report
    assert "case-study" in final_report.lower()
