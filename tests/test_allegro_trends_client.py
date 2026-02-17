from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from weekly_seo_agent.clients.allegro_trends_client import AllegroTrendsClient


def _response(status_code: int, payload: object) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    response.url = "https://search-trends-service-prod-passive.allegrogroup.com/searches/time-series"
    return response


def _build_client() -> AllegroTrendsClient:
    return AllegroTrendsClient(
        basic_auth_login="search-trends-ui",
        basic_auth_password="secret_basic",
        technical_account_login="seo_allegro",
        technical_account_password="secret_tech",
    )


def test_fetch_query_summary_aggregates_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()

    def fake_post(url: str, **_: object) -> requests.Response:
        assert "oauth" in url
        return _response(200, {"access_token": "tok_123"})

    def fake_get(url: str, **_: object) -> requests.Response:
        assert "searches/time-series" in url
        return _response(
            200,
            [
                {"visit": 10, "pv": 20, "offers": 2, "deals": 1, "gmv": 100.5},
                {"visit": 4, "pv": 6, "offers": None, "deals": 0, "gmv": 20},
            ],
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    summary = client.fetch_query_summary(
        query="pellet",
        from_day=date(2026, 2, 3),
        till_day=date(2026, 2, 9),
    )

    assert summary["query"] == "pellet"
    assert summary["visit"] == 14.0
    assert summary["pv"] == 26.0
    assert summary["offers"] == 2.0
    assert summary["deals"] == 1.0
    assert summary["gmv"] == 120.5
    assert summary["points"] == 2
    assert summary["http_code"] == 200


def test_fetch_query_summary_refreshes_token_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()
    post_calls = {"count": 0}
    get_calls = {"count": 0}

    def fake_post(_: str, **__: object) -> requests.Response:
        post_calls["count"] += 1
        token = f"tok_{post_calls['count']}"
        return _response(200, {"access_token": token})

    def fake_get(_: str, **__: object) -> requests.Response:
        get_calls["count"] += 1
        if get_calls["count"] == 1:
            return _response(401, {"message": "expired token"})
        return _response(200, [{"visit": 1, "pv": 2, "offers": 3, "deals": 4, "gmv": 5}])

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    summary = client.fetch_query_summary(
        query="sanki",
        from_day=date(2026, 2, 3),
        till_day=date(2026, 2, 9),
    )

    assert summary["gmv"] == 5.0
    assert summary["visit"] == 1.0
    assert summary["http_code"] == 200
    assert post_calls["count"] == 2
    assert get_calls["count"] == 2
