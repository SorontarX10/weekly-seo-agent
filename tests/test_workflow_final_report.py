from weekly_seo_agent.workflow import (
    _compose_final_report,
    _normalize_ai_commentary_markdown,
)


def test_compose_final_report_includes_manager_sections_and_tables() -> None:
    markdown_report = """
# Weekly SEO Intelligence Report (2026-02-25 | PL)

## Executive summary
- **What changed**: clicks up WoW, down YoY.
- **Priority actions (owner | ETA)**: Seasonality: Align content timing (SEO + Merchandising | this week); Data quality: Fix missing source (SEO Ops | 24h).

## What is happening and why
**Confirmed vs hypothesis**
Confirmed facts from data:
- KPI moved with stable CTR.
Working hypotheses:
- Demand timing drives page-mix shifts.

## Hypothesis protocol
| Hypothesis | Confidence | Falsifier | Validation metric | Validation date |
|---|---|---|---|---|
| Seasonality: demand timing is primary | High (82/100) | CTR weakens with stable demand | CTR by Page Name | 2026-03-03 |

## Governance and provenance
- Report date: 2026-02-25 | Country: PL.
- Model version: gpt-5.2.

## Evidence ledger
| ID | Source | Date | Note |
|---|---|---|---|
| E1 | GSC weekly window | 2026-02-16..2026-02-22 | KPI baseline |
| E2 | Trade plan sheet | 2026-02-16 | campaign context |
""".strip()

    commentary = """
### Narrative Flow
- Pattern indicates category-level demand rotation.

### Causal Chain
- Main hypothesis: demand timing and routing explain movement. Confidence: 72/100.

### Evidence by Source
- [E1] GSC weekly window confirms baseline movement.

### Priority Actions for This Week
- Validate demand timing against next-week segment deltas.

### Risks and Monitoring
- Escalate technical SEO only if CTR/position weakens.

### Continuity Check
- Compare with previous run to confirm driver persistence.

### Further Analysis Flags
- Re-check campaign overlap by date and channel.
""".strip()

    final_report = _compose_final_report(markdown_report, commentary)

    assert "## Executive summary" in final_report
    assert "## What is happening and why" in final_report
    assert "## Confirmed vs hypothesis" in final_report
    assert "## Priority actions (owner | ETA)" in final_report
    assert "| Priority action | Owner | ETA |" in final_report
    assert "## Hypothesis protocol" in final_report
    assert "## Evidence ledger" in final_report
    assert "[SEO Team | next run]" in final_report or "SEO + Merchandising" in final_report


def test_normalize_ai_commentary_enforces_owner_eta_and_evidence_table() -> None:
    raw = """
### Narrative Flow
Traffic moved with mix effects.
### Priority Actions for This Week
- Validate demand hypothesis on next run.
""".strip()

    normalized = _normalize_ai_commentary_markdown(raw)

    assert "### Priority Actions for This Week" in normalized
    assert "[SEO Team | next run]" in normalized
    assert "### Evidence by Source" in normalized
    assert "| Source | Evidence signal | Why it matters |" in normalized
    assert "### Causal Chain" in normalized
