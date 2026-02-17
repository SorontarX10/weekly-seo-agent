from weekly_seo_agent.models import MetricRow
from weekly_seo_agent.query_filter import filter_irrelevant_query_rows


def test_filter_irrelevant_query_rows_drops_matching_patterns() -> None:
    rows = [
        MetricRow(key="allegro ma non troppo", clicks=10, impressions=100, ctr=0.1, position=7.0),
        MetricRow(key="iphone 16 pro", clicks=55, impressions=400, ctr=0.1375, position=4.2),
        MetricRow(key="hotel allegro warszawa", clicks=12, impressions=90, ctr=0.133, position=8.1),
    ]

    filtered, dropped = filter_irrelevant_query_rows(
        rows,
        ("allegro ma non troppo", "hotel allegro"),
    )

    assert dropped == 2
    assert len(filtered) == 1
    assert filtered[0].key == "iphone 16 pro"


def test_filter_works_for_multi_dimension_row_key() -> None:
    rows = [
        MetricRow(
            key="allegro con brio | PL",
            clicks=5,
            impressions=50,
            ctr=0.1,
            position=9.0,
        ),
        MetricRow(
            key="telewizor samsung 55 | PL",
            clicks=40,
            impressions=500,
            ctr=0.08,
            position=5.5,
        ),
    ]

    filtered, dropped = filter_irrelevant_query_rows(rows, ("allegro con brio",))

    assert dropped == 1
    assert len(filtered) == 1
    assert filtered[0].key.startswith("telewizor")
