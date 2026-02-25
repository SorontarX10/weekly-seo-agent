from weekly_seo_agent.weekly_reporting_agent.evaluation import evaluate_report_text


def _strict_ready_report() -> str:
    return """
## Executive summary
In plain language: demand mix changed across categories. Business implication: short-term allocation focus.
Decision this week: keep technical escalation as second step. Data reliability is acceptable.
Priority actions (owner | ETA): SEO Team | next run.

## What is happening and why
Observation [E1]: clicks changed WoW and YoY.
Evidence [E2]: impressions moved with stable CTR.
Evidence [E3]: campaign timing overlaps with demand windows.
Evidence [E4]: weather shifts support timing interpretation.
Evidence [E5]: trend mix suggests category rotation.
Falsifier: if CTR and position deteriorate in the next window, demand-only thesis weakens.
Validation metric: CTR delta and position delta by page name.
Validation date: 2026-03-03.
Reference date: 2026-02-25.
YoY context: YoY demand is lower. YoY page mix differs. YoY trend themes changed.
YoY campaign overlap changed. YoY weather baseline differs. YoY platform/news pulse is mixed.

## Confirmed vs hypothesis
Confirmed: efficiency remained broadly stable.
Hypothesis: routing and demand timing explain most variance.

## Evidence ledger
- [E1] GSC summary
- [E2] GSC feature split
- [E3] Trade plan
- [E4] Weather context
- [E5] Trends context

## Governance and provenance
Source freshness reviewed and comparable windows aligned.
""".strip()


def test_evaluation_passes_stricter_profile() -> None:
    result = evaluate_report_text(_strict_ready_report())
    assert result["passed"] is True
    assert int(result["score"]) >= 85
    assert result["metrics"]["action_marker_hits"] >= 2
    assert result["metrics"]["yoy_mentions"] >= 6


def test_evaluation_flags_missing_action_and_yoy_depth() -> None:
    report = _strict_ready_report().replace("Priority actions (owner | ETA): SEO Team | next run.", "")
    report = report.replace(
        "YoY context: YoY demand is lower. YoY page mix differs. YoY trend themes changed.",
        "YoY context: one mention.",
    )
    report = report.replace(
        "YoY campaign overlap changed. YoY weather baseline differs. YoY platform/news pulse is mixed.",
        "",
    )

    result = evaluate_report_text(report)
    issues = " ".join(result.get("issues", []))
    assert "Action framing is incomplete" in issues
    assert "YoY coverage is too limited" in issues
    assert result["metrics"]["action_marker_hits"] < 2
    assert result["metrics"]["yoy_mentions"] < 6
