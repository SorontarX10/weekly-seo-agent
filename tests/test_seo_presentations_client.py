from __future__ import annotations

from datetime import date

from weekly_seo_agent.clients.seo_presentations_client import SEOPresentationsClient


def test_extract_folder_id_from_url() -> None:
    url = "https://drive.google.com/drive/folders/1XdOhuo4k4pWNRB8cRn0rLcp05fyLL7Do?fbclid=abc"
    assert (
        SEOPresentationsClient._extract_folder_id(url)
        == "1XdOhuo4k4pWNRB8cRn0rLcp05fyLL7Do"
    )


def test_extract_folder_id_from_raw_id() -> None:
    folder_id = "1XdOhuo4k4pWNRB8cRn0rLcp05fyLL7Do"
    assert SEOPresentationsClient._extract_folder_id(folder_id) == folder_id


def test_parse_filename_date() -> None:
    assert SEOPresentationsClient._parse_filename_date("2026-02-10 SEO weekly.pptx") == date(2026, 2, 10)
    assert SEOPresentationsClient._parse_filename_date("SEO_10.02.2026_summary") == date(2026, 2, 10)
    assert SEOPresentationsClient._parse_filename_date("SEO weekly summary") is None


def test_generic_highlight_filtering() -> None:
    assert SEOPresentationsClient._is_generic_highlight(
        "SEO Team DEMO, 2026.01.15",
        file_name="2026-01-15 Demo SEO Monthly",
    )
    assert not SEOPresentationsClient._is_generic_highlight(
        "Non-brand traffic +12.4% YoY after category page title rollout.",
        file_name="2026-01-15 Demo SEO Monthly",
    )
    assert SEOPresentationsClient._is_generic_highlight(
        "from: Hantle bitumiczne regulowane Trex Sport 2x 20 kg",
        file_name="2025-10-23 Demo SEO Monthly",
    )


def test_dedupe_presentation_rows_prefers_non_copy_google_slides() -> None:
    client = SEOPresentationsClient(
        client_secret_path="secret.json",
        token_path=".google_drive_token.json",
        folder_reference="1XdOhuo4k4pWNRB8cRn0rLcp05fyLL7Do",
    )
    rows = [
        {
            "id": "1",
            "name": "Copy of 2026-01-15 Demo SEO Monthly",
            "mimeType": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "modifiedTime": "2026-01-16T10:00:00Z",
            "file_date": "2026-01-15",
            "sort_day": date(2026, 1, 15),
        },
        {
            "id": "2",
            "name": "2026-01-15 Demo SEO Monthly",
            "mimeType": "application/vnd.google-apps.presentation",
            "modifiedTime": "2026-01-15T10:00:00Z",
            "file_date": "2026-01-15",
            "sort_day": date(2026, 1, 15),
        },
    ]

    deduped = client._dedupe_presentation_rows(rows)
    assert len(deduped) == 1
    assert deduped[0]["id"] == "2"


def test_detect_url_slug_noise() -> None:
    assert SEOPresentationsClient._looks_like_url_or_slug(
        "https://allegro.pl/oferta/hantle-bitumiczne-regulowane-trex-sport-2x-20-kg-3c201971-058b-4b81-829d-bfa2c6b8a344"
    )
    assert SEOPresentationsClient._looks_like_url_or_slug(
        "…hantle-bitumiczne-regulowane-trex-sport-2x-20-kg-3c201971-058b-4b81-829d-bfa2c6b8a344"
    )
    assert not SEOPresentationsClient._looks_like_url_or_slug(
        "GMV: 103 M (+24.31% vs Target)"
    )
