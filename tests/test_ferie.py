from __future__ import annotations

from datetime import date

from weekly_seo_agent import ferie
from weekly_seo_agent.ferie import build_ferie_context, build_upcoming_ferie_trends
from weekly_seo_agent.models import DateWindow


class _FakeResponse:
    def __init__(self, payload: list[dict]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict]:
        return self._payload


def _windows() -> dict[str, DateWindow]:
    return {
        "current_28d": DateWindow("Current 28 days", date(2026, 1, 13), date(2026, 2, 9)),
        "previous_28d": DateWindow("Previous 28 days", date(2025, 12, 16), date(2026, 1, 12)),
        "yoy_52w": DateWindow("YoY aligned (52 weeks ago)", date(2025, 1, 14), date(2025, 2, 10)),
    }


def _mock_get(url: str, params: dict[str, str] | None = None, timeout: int = 30):
    del timeout
    if url.endswith("/Subdivisions"):
        return _FakeResponse(
            [
                {
                    "code": "PL-MZ",
                    "name": [{"language": "PL", "text": "Mazowieckie"}],
                },
                {
                    "code": "PL-DS",
                    "name": [{"language": "PL", "text": "Dolnoslaskie"}],
                },
                {
                    "code": "PL-LU",
                    "name": [{"language": "PL", "text": "Lubelskie"}],
                },
            ]
        )

    if url.endswith("/SchoolHolidays"):
        _ = params or {}
        return _FakeResponse(
            [
                {
                    "name": [{"language": "PL", "text": "Ferie zimowe"}],
                    "startDate": "2026-01-19",
                    "endDate": "2026-02-01",
                    "subdivisions": [{"code": "PL-MZ"}, {"code": "PL-LU"}],
                },
                {
                    "name": [{"language": "PL", "text": "Ferie zimowe"}],
                    "startDate": "2026-02-02",
                    "endDate": "2026-02-15",
                    "subdivisions": [{"code": "PL-DS"}],
                },
                {
                    "name": [{"language": "PL", "text": "Ferie zimowe"}],
                    "startDate": "2025-01-20",
                    "endDate": "2025-02-02",
                    "subdivisions": [{"code": "PL-DS"}, {"code": "PL-MZ"}],
                },
                {
                    "name": [{"language": "PL", "text": "Przerwa swiateczna"}],
                    "startDate": "2026-12-23",
                    "endDate": "2026-12-31",
                    "subdivisions": [{"code": "PL-MZ"}],
                },
            ]
        )

    raise AssertionError(f"Unexpected URL: {url}")


def test_build_ferie_context_contains_window_and_profile_stats(monkeypatch) -> None:
    monkeypatch.setattr(ferie.requests, "get", _mock_get)
    context = build_ferie_context(_windows())

    profiles = context["profiles_ranked"]
    assert isinstance(profiles, list)
    assert profiles
    gmv_total = sum(float(profile["gmv_share"]) for profile in profiles)
    assert abs(gmv_total - 1.0) < 1e-9

    window_stats = context["window_stats"]
    assert isinstance(window_stats, dict)
    assert int(window_stats["current_28d"]["days_with_ferie"]) > 0
    assert float(window_stats["current_28d"]["avg_daily_gmv_share"]) > 0.0

    yoy = context["yoy_comparison"]
    assert "avg_daily_delta_pp" in yoy
    assert isinstance(yoy["rows"], list)


def test_upcoming_ferie_trends_in_horizon(monkeypatch) -> None:
    monkeypatch.setattr(ferie.requests, "get", _mock_get)
    rows = build_upcoming_ferie_trends(run_date=date(2026, 2, 10), horizon_days=60)
    assert rows
    assert any("Winter break" in row[2] for row in rows)
    assert all(row[0] > date(2026, 2, 10) for row in rows)
