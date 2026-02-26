from weekly_seo_agent.weekly_reporting_agent.evaluation import (
    compare_report_regression,
    evaluate_duplication_verbosity,
    evaluate_evidence_alignment,
    evaluate_readability,
    evaluate_report_text,
)


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
    assert int(result["score"]) >= 88
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


def test_readability_evaluator_flags_jargon_and_sentence_length() -> None:
    noisy = (
        "In plain language: this is hard to read. "
        + "SERP indexation canonical volatility merchant_listings product_snippets p52w query cluster algorithm context. "
        + "This sentence is intentionally very long and keeps adding non-essential details so it exceeds the target sentence length for non-SEO audiences and decreases clarity significantly. "
        + "Business implication: unclear."
    )
    result = evaluate_readability(noisy)
    assert int(result["score"]) < 90
    assert result["metrics"]["jargon_hits"] > 0
    assert result["metrics"]["avg_sentence_words"] >= 10.0


def test_duplication_and_verbosity_evaluator_flags_repetition() -> None:
    repeated = "\n".join(
        [
            "## Executive summary",
            "Priority actions (owner | ETA): SEO Team | next run.",
            "Same sentence repeated for noise.",
            "Same sentence repeated for noise.",
            "Same sentence repeated for noise.",
            "Same sentence repeated for noise.",
            "Same sentence repeated for noise.",
            "Confidence: 70/100. Confidence: 70/100. Confidence: 70/100.",
        ]
    )
    result = evaluate_duplication_verbosity(repeated)
    assert int(result["score"]) < 90
    assert result["metrics"]["duplicate_line_ratio"] > 0.10


def test_evidence_alignment_flags_unsupported_claims() -> None:
    report = """
## Executive summary
In plain language: demand changed.

## What is happening and why
Clicks are down YoY because routing changed and campaign overlap intensified.
This implies broad category demand pressure with no technical SEO decline.
Validation metric: CTR delta.
Validation date: 2026-03-03.
Falsifier: if CTR drops, hypothesis fails.

## Confirmed vs hypothesis
Confirmed: movement exists.
Hypothesis: demand rotation.

## Evidence ledger
- [E1] GSC totals

## Governance and provenance
Report date: 2026-02-26.
""".strip()
    result = evaluate_evidence_alignment(report)
    assert int(result["score"]) < 90
    assert result["metrics"]["claim_count"] >= 2
    assert result["metrics"]["claim_coverage_ratio"] < 0.8


def test_regression_comparison_reports_score_delta_and_line_changes() -> None:
    baseline = _strict_ready_report()
    candidate = baseline.replace("Decision this week", "Decision this week updated")
    diff = compare_report_regression(candidate, baseline)
    assert diff["added_lines_count"] >= 0
    assert diff["removed_lines_count"] >= 0
    assert isinstance(diff["score_delta"], int)
    assert diff["changed_line_ratio"] >= 0.0
