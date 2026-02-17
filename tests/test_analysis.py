from weekly_seo_agent.analysis import analyze_rows
from weekly_seo_agent.models import MetricRow


def test_analyze_rows_detects_significant_drop() -> None:
    current_rows = [
        MetricRow(key="k1", clicks=100, impressions=1000, ctr=0.10, position=4.2),
        MetricRow(key="k2", clicks=80, impressions=900, ctr=0.088, position=5.0),
    ]
    previous_rows = [
        MetricRow(key="k1", clicks=260, impressions=1100, ctr=0.236, position=3.4),
        MetricRow(key="k2", clicks=150, impressions=950, ctr=0.157, position=4.0),
    ]
    yoy_rows = [
        MetricRow(key="k1", clicks=210, impressions=1200, ctr=0.175, position=3.7),
        MetricRow(key="k2", clicks=170, impressions=1000, ctr=0.170, position=4.3),
    ]

    result = analyze_rows(
        current_rows=current_rows,
        previous_rows=previous_rows,
        yoy_rows=yoy_rows,
        top_n=10,
        min_click_loss_absolute=100,
        min_click_loss_pct=0.15,
    )

    assert result.top_losers[0].key in {"k1", "k2"}
    assert any("Significant click drop" in finding.title for finding in result.findings)
