from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from weekly_seo_agent.clients.senuto_client import SenutoClient


def _response(status_code: int, payload: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    response.url = "https://api.senuto.com/api/users/token"
    return response


def _build_client(token: str = "", token_endpoint: str = "/api/users/token") -> SenutoClient:
    return SenutoClient(
        token=token,
        email="user@example.com",
        password="secret",
        token_endpoint=token_endpoint,
        base_url="https://api.senuto.com",
        domain="allegro.pl",
        visibility_endpoint=(
            "/api/visibility_analysis/reports/domain_positions/"
            "getPositionsSumsDistributionChartData"
        ),
    )


def test_bearer_token_is_normalized() -> None:
    client = _build_client(token="Bearer tok_123")
    assert client._authenticate(force_refresh=False) == "tok_123"


def test_authentication_tries_fallback_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    called_urls: list[str] = []

    def fake_post(url: str, **_: object) -> requests.Response:
        called_urls.append(url)
        if url.endswith("/api/users/token"):
            return _response(200, {"token": "tok_abc"})
        return _response(404, {"message": "not found"})

    monkeypatch.setattr(requests, "post", fake_post)

    client = _build_client(token="", token_endpoint="/users/token")
    token = client._authenticate(force_refresh=True)

    assert token == "tok_abc"
    assert any(url.endswith("/users/token") for url in called_urls)
    assert any(url.endswith("/api/users/token") for url in called_urls)


def test_authentication_error_includes_api_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **_: object) -> requests.Response:
        return _response(
            418,
            {
                "success": False,
                "data": {
                    "error": {
                        "type": "unauthorized",
                        "message": "Invalid username or password",
                    }
                },
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = _build_client(token="")
    with pytest.raises(RuntimeError) as error:
        client._authenticate(force_refresh=True)

    assert "Invalid username or password" in str(error.value)


def test_get_request_retries_after_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(token="tok_old")

    def fake_auth(force_refresh: bool = False) -> str:
        return "tok_new" if force_refresh else "tok_old"

    monkeypatch.setattr(client, "_authenticate", fake_auth)

    called_auth_headers: list[str] = []

    def fake_request(
        method: str,
        url: str,
        headers: dict[str, str],
        **_: object,
    ) -> requests.Response:
        assert method.upper() == "GET"
        called_auth_headers.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer tok_old":
            return _response(401, {"message": "expired token"})
        return _response(200, {"data": {}})

    monkeypatch.setattr(requests, "request", fake_request)

    response = client._get_with_auth_retry(url="https://api.senuto.com/api/test", params={})

    assert response.status_code == 200
    assert called_auth_headers == ["Bearer tok_old", "Bearer tok_new"]


def test_resolve_country_id_prefers_pl_new(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(token="tok")

    monkeypatch.setattr(
        client,
        "fetch_countries_list",
        lambda: [
            {"id": 1, "code": "pl"},
            {"id": 200, "code": "pl_new"},
            {"id": 50, "code": "cz"},
        ],
    )

    assert client.resolve_country_id("PL", fallback_country_id=1) == 200
    assert client.resolve_country_id("CZ", fallback_country_id=1) == 50
    assert client.resolve_country_id("HU", fallback_country_id=1) == 1


def test_fetch_competitor_keyword_gap_filters_brand_and_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(token="tok")
    client.country_id = 1

    payload = {
        "success": True,
        "data": [
            {
                "keyword": "allegro smart",
                "searches": 9000,
                "difficulty": 20,
                "allegro.pl_pos": 1,
                "temu.com_pos": 2,
            },
            {
                "keyword": "grill gazowy 4 palniki",
                "searches": 5400,
                "difficulty": 40,
                "allegro.pl_pos": 0,
                "temu.com_pos": 4,
                "amazon.pl_pos": 5,
            },
            {
                "keyword": "robot sprzatajacy",
                "searches": 2400,
                "difficulty": 30,
                "allegro.pl_pos": 28,
                "temu.com_pos": 6,
                "amazon.pl_pos": 8,
            },
        ],
    }

    monkeypatch.setattr(
        client,
        "_post_json_with_auth_retry",
        lambda url, payload: _response(200, payload=payload_data),  # type: ignore[misc]
    )

    # Python lambda above cannot capture `payload` argument and keep fixture payload cleanly
    # for mypy, so we inject through outer variable.
    payload_data = payload

    rows = client.fetch_competitor_keyword_gap(
        competitors_domains=["temu.com", "amazon.pl"],
        top_n=10,
    )

    assert len(rows) == 2
    assert rows[0]["keyword"] in {"grill gazowy 4 palniki", "robot sprzatajacy"}
    assert all("allegro" not in str(row["keyword"]).lower() for row in rows)


def test_fetch_competitor_keyword_gap_returns_empty_when_allegro_already_leads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(token="tok")
    client.country_id = 1

    payload_data = {
        "success": True,
        "data": [
            {
                "keyword": "lampka do czytania",
                "searches": 9900,
                "difficulty": 52,
                "allegro.pl_pos": 1,
                "temu.com_pos": 4,
            },
            {
                "keyword": "torba na silownie",
                "searches": 4400,
                "difficulty": 41,
                "allegro.pl_pos": 2,
                "temu.com_pos": 4,
            },
        ],
    }

    monkeypatch.setattr(
        client,
        "_post_json_with_auth_retry",
        lambda url, payload: _response(200, payload=payload_data),  # type: ignore[misc]
    )

    rows = client.fetch_competitor_keyword_gap(
        competitors_domains=["temu.com"],
        top_n=10,
    )

    assert rows == []


def test_fetch_serp_volatility_supports_dict_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(token="tok")

    payload_data = {
        "success": True,
        "data": [
            {
                "domain": "allegro.pl",
                "positions": {"2026-02-04": 1, "2026-02-08": 3},
                "monthly_positions": {"2026-02-04": 1, "2026-02-08": 3},
            }
        ],
    }

    monkeypatch.setattr(
        client,
        "_post_form_with_auth_retry",
        lambda url, payload: _response(200, payload=payload_data),  # type: ignore[misc]
    )

    rows = client.fetch_serp_volatility(
        keyword="allegro",
        start_date=date(2026, 2, 3),
        end_date=date(2026, 2, 9),
        top_n=10,
    )

    assert len(rows) == 1
    assert rows[0]["positions_points"] == 2
    assert rows[0]["monthly_points"] == 2
