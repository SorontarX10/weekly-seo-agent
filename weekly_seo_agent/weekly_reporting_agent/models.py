from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class DateWindow:
    name: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass
class MetricRow:
    key: str
    clicks: float
    impressions: float
    ctr: float
    position: float


@dataclass
class MetricSummary:
    clicks: float = 0.0
    impressions: float = 0.0
    ctr: float = 0.0
    position: float = 0.0

    @classmethod
    def from_rows(cls, rows: list[MetricRow]) -> "MetricSummary":
        if not rows:
            return cls()

        total_clicks = sum(row.clicks for row in rows)
        total_impressions = sum(row.impressions for row in rows)
        weighted_position = (
            sum(row.position * row.impressions for row in rows) / total_impressions
            if total_impressions
            else 0.0
        )
        return cls(
            clicks=total_clicks,
            impressions=total_impressions,
            ctr=(total_clicks / total_impressions if total_impressions else 0.0),
            position=weighted_position,
        )


@dataclass
class VisibilityPoint:
    day: date
    visibility: float


@dataclass
class ExternalSignal:
    source: str
    day: date
    title: str
    details: str
    severity: str = "info"
    url: str | None = None


@dataclass
class Finding:
    severity: str
    title: str
    details: str
    recommendation: str


@dataclass
class KeyDelta:
    key: str
    current_clicks: float
    previous_clicks: float
    yoy_clicks: float
    click_delta_vs_previous: float
    click_delta_pct_vs_previous: float
    click_delta_vs_yoy: float
    click_delta_pct_vs_yoy: float
    current_ctr: float
    previous_ctr: float
    yoy_ctr: float
    current_position: float
    previous_position: float
    yoy_position: float


@dataclass
class AnalysisResult:
    summary_current: MetricSummary
    summary_previous: MetricSummary
    summary_yoy: MetricSummary
    top_losers: list[KeyDelta] = field(default_factory=list)
    top_winners: list[KeyDelta] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
