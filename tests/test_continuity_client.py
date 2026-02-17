from __future__ import annotations

from datetime import date

from weekly_seo_agent.clients.continuity_client import ContinuityClient


def _client() -> ContinuityClient:
    return ContinuityClient(
        client_secret_path="secret.json",
        token_path=".google_drive_token.json",
        reports_folder_name="SEO Weekly Reports",
    )


def test_extract_drive_id_from_urls() -> None:
    folder_url = "https://drive.google.com/drive/folders/1XdOhuo4k4pWNRB8cRn0rLcp05fyLL7Do"
    doc_url = "https://docs.google.com/document/d/1u32yuA4DsmDJ4OlQOhO_vi2LjvpN5g4J-HBcXoMmsI4/edit"
    assert (
        ContinuityClient._extract_drive_id(folder_url)
        == "1XdOhuo4k4pWNRB8cRn0rLcp05fyLL7Do"
    )
    assert (
        ContinuityClient._extract_drive_id(doc_url)
        == "1u32yuA4DsmDJ4OlQOhO_vi2LjvpN5g4J-HBcXoMmsI4"
    )


def test_parse_report_date_supports_multiple_formats() -> None:
    assert ContinuityClient._parse_report_date("2026_02_10_seo_weekly_report") == date(2026, 2, 10)
    assert ContinuityClient._parse_report_date("seo_weekly_10-02-2026") == date(2026, 2, 10)
    assert ContinuityClient._parse_report_date("no-date-here") is None


def test_select_recent_and_yoy() -> None:
    client = _client()
    run_date = date(2026, 2, 10)

    rows = [
        {"id": "r1", "name": "2026_02_03", "report_date": date(2026, 2, 3), "modifiedTime": "", "webViewLink": ""},
        {"id": "r2", "name": "2026_01_27", "report_date": date(2026, 1, 27), "modifiedTime": "", "webViewLink": ""},
        {"id": "r3", "name": "2026_01_20", "report_date": date(2026, 1, 20), "modifiedTime": "", "webViewLink": ""},
        {"id": "y1", "name": "2025_02_11", "report_date": date(2025, 2, 11), "modifiedTime": "", "webViewLink": ""},
    ]

    recent, yoy = client._select_recent_and_yoy(rows=rows, run_date=run_date)

    assert [row["id"] for row in recent] == ["r1", "r2", "r3"]
    assert yoy is not None
    assert yoy["id"] == "y1"


def test_parse_flexible_date_examples() -> None:
    assert ContinuityClient._parse_flexible_date("2026-02-10") == date(2026, 2, 10)
    assert ContinuityClient._parse_flexible_date("10.02.2026") == date(2026, 2, 10)
    assert ContinuityClient._parse_flexible_date(46000) == date(2025, 12, 9)


def test_pick_date_column_prefers_status_date() -> None:
    headers = ["Topic", "Status date", "Owner"]
    assert ContinuityClient._pick_date_column(headers) == 1


def test_parse_numeric_cell_handles_k_and_decimal_formats() -> None:
    assert ContinuityClient._parse_numeric_cell("1,5k") == 1500.0
    assert ContinuityClient._parse_numeric_cell("2.200") == 2200.0
    assert ContinuityClient._parse_numeric_cell("2 200") == 2200.0
    assert ContinuityClient._parse_numeric_cell("n/a") is None


def test_extract_non_brand_yoy_rows_filters_brand_terms() -> None:
    client = _client()
    tables = [
        {
            "sheet": "Current",
            "headers": ["keyword", "page", "seo_score", "timestamp"],
            "rows": [
                ["sanki dziecięce", "https://allegro.pl/listing?string=sanki", "1000", "2026-02-10"],
                ["prezent walentynki", "https://allegro.pl/listing?string=walentynki", "800", "2026-02-10"],
                ["allegro smart", "https://allegro.pl/smart", "2500", "2026-02-10"],
                ["sanki dziecięce", "https://allegro.pl/listing?string=sanki", "500", "2025-02-11"],
                ["prezent walentynki", "https://allegro.pl/listing?string=walentynki", "300", "2025-02-11"],
                ["topinambur", "https://allegro.cz/listing?string=topinambur", "900", "2026-02-10"],
            ],
        }
    ]
    rows = client._extract_non_brand_yoy_rows(
        tables,
        run_date=date(2026, 2, 10),
        top_rows=5,
        target_domain="allegro.pl",
    )
    trends = [str(row.get("trend", "")) for row in rows]
    assert "allegro smart" not in trends
    assert "topinambur" not in trends
    assert "sanki dziecięce" in trends
    assert "prezent walentynki" in trends


def test_extract_non_brand_upcoming_rows_uses_horizon() -> None:
    client = _client()
    tables = [
        {
            "sheet": "Upcoming",
            "headers": ["Date", "Trend", "Page", "Score"],
            "rows": [
                ["2026-02-20", "prezent walentynki", "https://allegro.pl/listing?string=walentynki", "1200"],
                ["2026-03-25", "wielkanoc dekoracje", "https://allegro.pl/listing?string=wielkanoc", "1400"],
                ["2026-02-18", "allegro smart", "https://allegro.pl/smart", "2000"],
                ["2026-02-19", "buty trekkingowe", "https://allegro.cz/listing?string=buty", "1800"],
            ],
        }
    ]
    rows = client._extract_non_brand_upcoming_rows(
        tables=tables,
        run_date=date(2026, 2, 10),
        horizon_days=31,
        top_rows=10,
        target_domain="allegro.pl",
    )
    assert len(rows) == 1
    assert rows[0]["trend"] == "prezent walentynki"


def test_extract_non_brand_upcoming_rows_without_date_column_uses_sheet_scope() -> None:
    client = _client()
    tables = [
        {
            "sheet": "next31D",
            "headers": ["keyword", "page", "seo_score"],
            "rows": [
                ["glebogryzarka spalinowa", "https://allegro.pl/listing?string=glebogryzarka", "182.7"],
                ["allegro days", "https://allegro.pl/listing?string=allegro%20days", "111.0"],
                ["wertykulator", "https://allegro.pl/listing?string=wertykulator", "128.7"],
            ],
        }
    ]
    rows = client._extract_non_brand_upcoming_rows(
        tables=tables,
        run_date=date(2026, 2, 10),
        horizon_days=31,
        top_rows=10,
        target_domain="allegro.pl",
    )
    trends = [str(row.get("trend", "")) for row in rows]
    assert "glebogryzarka spalinowa" in trends
    assert "wertykulator" in trends
    assert "allegro days" not in trends
