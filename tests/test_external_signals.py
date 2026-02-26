from __future__ import annotations

from datetime import date

from weekly_seo_agent.clients import external_signals
from weekly_seo_agent.clients.external_signals import ExternalSignalsClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def _mock_get(url: str, params=None, timeout: int = 30):
    del timeout
    if url.endswith("/Subdivisions"):
        return _FakeResponse(
            [
                {
                    "code": "PL-MZ",
                    "name": [{"language": "PL", "text": "Mazowieckie"}],
                }
            ]
        )
    if url.endswith("/PublicHolidays"):
        return _FakeResponse(
            [
                {
                    "startDate": "2026-01-01",
                    "name": [{"language": "PL", "text": "Nowy Rok"}],
                    "regionalScope": "National",
                    "nationwide": True,
                },
                {
                    "startDate": "2026-01-06",
                    "name": [{"language": "PL", "text": "Swieto regionalne"}],
                    "regionalScope": "Regional",
                    "nationwide": False,
                    "subdivisions": [{"code": "PL-MZ"}],
                },
            ]
        )
    raise AssertionError(f"Unexpected URL: {url} params={params}")


def test_holiday_signals_are_parsed_from_openholidays(monkeypatch) -> None:
    monkeypatch.setattr(external_signals.requests, "get", _mock_get)

    client = ExternalSignalsClient(
        latitude=52.2297,
        longitude=21.0122,
        weather_label="PL-central",
        weather_context_enabled=True,
        market_country_code="PL",
        status_endpoint="https://status.search.google.com/incidents.json",
        blog_rss_url="https://feeds.feedburner.com/blogspot/amDG",
        holidays_country_code="PL",
        holidays_api_base_url="https://openholidaysapi.org",
        holidays_language_code="PL",
        news_scraping_enabled=False,
        news_rss_urls_pl=(),
        news_rss_urls_global=(),
        news_html_urls_pl=(),
        news_html_urls_global=(),
        news_keywords=(),
        news_max_signals=20,
    )

    rows = client._holiday_signals(start=date(2026, 1, 1), end=date(2026, 1, 10))
    assert len(rows) == 2
    assert rows[0].source == "Public Holidays"
    assert rows[0].title.startswith("Holiday:")
    assert any("Regional holiday" in row.details for row in rows)
