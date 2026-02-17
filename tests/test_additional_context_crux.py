from datetime import date

from weekly_seo_agent.additional_context import (
    _fetch_campaign_tracker_signals,
    _fetch_market_event_calendar,
    _fetch_imgw_warnings,
    _origin_from_url,
    _overall_cwv_category,
    _parse_crux_p75,
)


def test_origin_from_url_normalizes_path() -> None:
    assert _origin_from_url("https://allegro.pl/") == "https://allegro.pl"
    assert _origin_from_url("https://allegro.pl/some/path") == "https://allegro.pl"


def test_parse_crux_p75_handles_cls_scale() -> None:
    metrics = {
        "cumulative_layout_shift": {
            "percentiles": {"p75": 12},
        }
    }
    assert _parse_crux_p75(metrics, ("cumulative_layout_shift",)) == 0.12


def test_overall_cwv_category_uses_worst_metric() -> None:
    assert _overall_cwv_category(lcp_ms=1800, inp_ms=180, cls=0.08) == "good"
    assert _overall_cwv_category(lcp_ms=4500, inp_ms=180, cls=0.08) == "poor"
    assert _overall_cwv_category(lcp_ms=1800, inp_ms=350, cls=0.08) == "needs_improvement"


class _FakeResponse:
    def __init__(self, text: str = "", payload=None):
        self.text = text
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_fetch_campaign_tracker_signals_from_rss(monkeypatch) -> None:
    rss = """<?xml version="1.0"?>
<rss><channel>
<item>
<title>Allegro Days campaign starts this week</title>
<link>https://example.com/allegro-days</link>
<description>Allegro promocja for Allegro Days.</description>
<pubDate>Tue, 10 Feb 2026 10:00:00 GMT</pubDate>
</item>
</channel></rss>"""

    def _mock_get(url: str, timeout: int = 30):
        del url, timeout
        return _FakeResponse(text=rss)

    monkeypatch.setattr("weekly_seo_agent.additional_context.requests.get", _mock_get)
    signals = _fetch_campaign_tracker_signals(since=date(2026, 1, 1), run_date=date(2026, 2, 10))
    assert signals
    assert any("Allegro Days" in row.title for row in signals)


def test_fetch_imgw_warnings_parses_severity(monkeypatch) -> None:
    payload = [
        {
            "nazwa_zdarzenia": "Intense snowfall",
            "stopien": "2 stopnia",
            "obowiazuje_od": "2026-02-09T00:00:00+01:00",
            "obowiazuje_do": "2026-02-11T00:00:00+01:00",
            "wojewodztwo": "mazowieckie",
        }
    ]

    def _mock_get(url: str, timeout: int = 30):
        del url, timeout
        return _FakeResponse(payload=payload)

    monkeypatch.setattr("weekly_seo_agent.additional_context.requests.get", _mock_get)
    rows = _fetch_imgw_warnings(
        run_date=date(2026, 2, 10),
        imgw_warnings_url="https://example.com/imgw",
    )
    assert rows
    assert rows[0]["severity"] == "2"


def test_fetch_market_event_calendar_parses_gdelt_rows(monkeypatch) -> None:
    payload = {
        "articles": [
            {
                "title": "Amazon launches Black Friday campaign in Czechia",
                "url": "https://example.com/amazon-black-friday",
                "domain": "example.com",
                "seendate": "20260210T080000Z",
            },
            {
                "title": "Logistics strike may disrupt deliveries in Prague",
                "url": "https://example.com/logistics-strike",
                "domain": "example.com",
                "seendate": "20260209T080000Z",
            },
        ]
    }

    def _mock_get(url: str, params=None, timeout: int = 35):
        del url, params, timeout
        return _FakeResponse(payload=payload)

    monkeypatch.setattr("weekly_seo_agent.additional_context.requests.get", _mock_get)
    rows = _fetch_market_event_calendar(
        country_code="CZ",
        since=date(2026, 2, 1),
        until=date(2026, 2, 15),
        top_rows=5,
    )
    assert rows
    assert any(row["event_type"] == "Campaign/Promotions" for row in rows)
    assert any(row["event_type"] == "Logistics/Delivery" for row in rows)
