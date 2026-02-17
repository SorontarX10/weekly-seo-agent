from __future__ import annotations

from weekly_seo_agent.models import AnalysisResult, Finding, KeyDelta, MetricRow, MetricSummary, VisibilityPoint


def _empty_row(key: str) -> MetricRow:
    return MetricRow(key=key, clicks=0.0, impressions=0.0, ctr=0.0, position=0.0)


def _safe_pct(current: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if current > 0 else 0.0
    return (current - baseline) / baseline


def _build_delta_rows(
    current_rows: list[MetricRow],
    previous_rows: list[MetricRow],
    yoy_rows: list[MetricRow],
) -> list[KeyDelta]:
    current_map = {row.key: row for row in current_rows}
    previous_map = {row.key: row for row in previous_rows}
    yoy_map = {row.key: row for row in yoy_rows}

    keys = set(current_map) | set(previous_map) | set(yoy_map)
    delta_rows: list[KeyDelta] = []
    for key in keys:
        current = current_map.get(key) or _empty_row(key)
        previous = previous_map.get(key) or _empty_row(key)
        yoy = yoy_map.get(key) or _empty_row(key)

        delta_rows.append(
            KeyDelta(
                key=key,
                current_clicks=current.clicks,
                previous_clicks=previous.clicks,
                yoy_clicks=yoy.clicks,
                click_delta_vs_previous=current.clicks - previous.clicks,
                click_delta_pct_vs_previous=_safe_pct(current.clicks, previous.clicks),
                click_delta_vs_yoy=current.clicks - yoy.clicks,
                click_delta_pct_vs_yoy=_safe_pct(current.clicks, yoy.clicks),
                current_ctr=current.ctr,
                previous_ctr=previous.ctr,
                yoy_ctr=yoy.ctr,
                current_position=current.position,
                previous_position=previous.position,
                yoy_position=yoy.position,
            )
        )

    return delta_rows


def analyze_rows(
    current_rows: list[MetricRow],
    previous_rows: list[MetricRow],
    yoy_rows: list[MetricRow],
    top_n: int,
    min_click_loss_absolute: int,
    min_click_loss_pct: float,
) -> AnalysisResult:
    summary_current = MetricSummary.from_rows(current_rows)
    summary_previous = MetricSummary.from_rows(previous_rows)
    summary_yoy = MetricSummary.from_rows(yoy_rows)

    delta_rows = _build_delta_rows(current_rows, previous_rows, yoy_rows)
    losers = sorted(delta_rows, key=lambda row: row.click_delta_vs_previous)[:top_n]
    winners = sorted(delta_rows, key=lambda row: row.click_delta_vs_previous, reverse=True)[:top_n]

    findings: list[Finding] = []

    previous_click_delta = summary_current.clicks - summary_previous.clicks
    previous_click_delta_pct = _safe_pct(summary_current.clicks, summary_previous.clicks)
    yoy_click_delta = summary_current.clicks - summary_yoy.clicks
    yoy_click_delta_pct = _safe_pct(summary_current.clicks, summary_yoy.clicks)

    if (
        previous_click_delta <= -float(min_click_loss_absolute)
        and previous_click_delta_pct <= -min_click_loss_pct
    ):
        findings.append(
            Finding(
                severity="high",
                title="Significant click drop vs previous 28 days",
                details=(
                    f"Clicks changed by {previous_click_delta:.0f} ({previous_click_delta_pct * 100:.1f}%)."
                ),
                recommendation="Audit top losing queries/pages and check matching external signals.",
            )
        )

    if (
        yoy_click_delta <= -float(min_click_loss_absolute)
        and yoy_click_delta_pct <= -min_click_loss_pct
    ):
        findings.append(
            Finding(
                severity="high",
                title="Significant click drop YoY (52-week aligned)",
                details=f"Clicks changed by {yoy_click_delta:.0f} ({yoy_click_delta_pct * 100:.1f}%).",
                recommendation="Validate seasonality assumptions and compare against visibility trends.",
            )
        )

    ctr_delta_prev = summary_current.ctr - summary_previous.ctr
    if ctr_delta_prev <= -0.01:
        findings.append(
            Finding(
                severity="medium",
                title="CTR down vs previous 28 days",
                details=f"CTR changed by {ctr_delta_prev * 100:.2f} percentage points.",
                recommendation="Review snippets for affected pages and queries.",
            )
        )

    position_delta_prev = summary_current.position - summary_previous.position
    if position_delta_prev >= 0.5:
        findings.append(
            Finding(
                severity="medium",
                title="Average position worsened",
                details=f"Average position changed by +{position_delta_prev:.2f}.",
                recommendation="Focus optimization on the most impacted clusters.",
            )
        )

    return AnalysisResult(
        summary_current=summary_current,
        summary_previous=summary_previous,
        summary_yoy=summary_yoy,
        top_losers=losers,
        top_winners=winners,
        findings=findings,
    )


def summarize_visibility(points: list[VisibilityPoint]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    avg = sum(point.visibility for point in points) / len(points)
    latest = points[-1].visibility
    return avg, latest
