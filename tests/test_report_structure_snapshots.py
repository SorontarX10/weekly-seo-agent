from __future__ import annotations

from pathlib import Path

from weekly_seo_agent.weekly_reporting_agent.evaluation import compare_report_regression
from weekly_seo_agent.workflow import _compose_final_report


SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "final_report_snapshots"
COUNTRIES = ("PL", "CZ", "SK", "HU")


def _snapshot_markdown(country_code: str) -> str:
    return f"""
# Weekly SEO Intelligence Report (2026-02-24 | {country_code})

## Executive summary
- **In plain language**: category demand rotated while efficiency stayed stable.
- **What changed**: clicks +1.3% WoW and -11.1% YoY.
- **Business implication**: allocation and SERP mix matter more than technical risk this week.
- **Decision this week**: keep technical SEO escalation as secondary unless CTR weakens.
- **Data reliability**: daily coverage is complete for 7/7 days.
- **Priority actions (owner | ETA)**: validate routing mix (SEO Team | next run); align campaign timing (SEO + Merchandising | this week).

## What is happening and why
Observation [E1]: traffic moved with stable CTR and position.
Evidence [E2]: campaign overlap and market events can reallocate demand.
Evidence [E3]: weather supports demand-timing interpretation.
Evidence [E4]: trend mix changed at category level.
Evidence [E5]: SERP feature mix changed WoW and YoY.
Falsifier: if CTR and position weaken next week, demand-only thesis is invalid.
Validation metric: CTR delta and page-level click delta by day.
Validation date: 2026-03-03.

## Confirmed vs hypothesis
Confirmed: efficiency is broadly stable.
Hypothesis: routing and seasonal demand timing explain most movement.

## Hypothesis protocol
| Hypothesis | Confidence | Falsifier | Validation metric | Validation date |
|---|---|---|---|---|
| Demand timing dominates | 74/100 | CTR drop with stable demand | CTR by Page Name | 2026-03-03 |

## Evidence ledger
| ID | Source | Date | Note |
|---|---|---|---|
| E1 | GSC weekly totals | 2026-02-22 | KPI baseline |
| E2 | Trade plan sheet | 2026-02-22 | Campaign timing |
| E3 | Weather context | 2026-02-22 | Demand timing |
| E4 | Product trends sheet | 2026-02-22 | Category movement |
| E5 | GSC feature split | 2026-02-22 | SERP mix |

## Governance and provenance
- Report date: 2026-02-24.
- Country: {country_code}.
- Model version: gpt-5.2.
""".strip()


def _snapshot_commentary() -> str:
    return """
### Narrative Flow
- This week reflects demand rotation with stable efficiency.

### Causal Chain
- Main hypothesis: demand timing + routing explain movement. Confidence: 74/100.

### Evidence by Source
| Source | Evidence signal | Why it matters |
|---|---|---|
| GSC totals | Stable CTR and position | No broad technical degradation |
| Trade plan | Campaign overlap in the week | Reallocates user demand |
| Weather | Cooler and wetter week | Supports timing effects |

### Priority Actions for This Week
- Validate routing shifts by Page Name [SEO Team | next run]
- Align campaign timing with seasonal clusters [SEO + Merchandising | this week]

### Risks and Monitoring
- Escalate technical SEO only if efficiency weakens.

### Continuity Check
- Compare top drivers vs previous run before escalation.

### Further Analysis Flags
- Re-check event timeline overlap with day-level GSC anomalies.
""".strip()


def _build_snapshot_report(country_code: str) -> str:
    return _compose_final_report(
        _snapshot_markdown(country_code),
        _snapshot_commentary(),
    )


def test_country_snapshot_regression_structure_and_key_phrases() -> None:
    for country_code in COUNTRIES:
        current = _build_snapshot_report(country_code)
        expected_path = SNAPSHOT_DIR / f"{country_code.lower()}_final_report.md"
        expected = expected_path.read_text(encoding="utf-8")
        assert current == expected
        assert f"Country: {country_code}." in current
        assert "## Executive summary" in current
        assert "## Priority actions (owner | ETA)" in current
        assert "## Evidence ledger" in current


def test_deterministic_regression_delta_against_pl_baseline() -> None:
    baseline = (SNAPSHOT_DIR / "pl_final_report.md").read_text(encoding="utf-8")
    current = _build_snapshot_report("PL")
    delta = compare_report_regression(current, baseline)
    assert delta["changed_line_ratio"] == 0.0
    assert delta["score_delta"] == 0
