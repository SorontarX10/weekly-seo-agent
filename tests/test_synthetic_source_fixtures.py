from __future__ import annotations

import json
from pathlib import Path

from weekly_seo_agent.models import DateWindow
from weekly_seo_agent.workflow import _build_metric_window_availability


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_source_coverage_cases.json"


def _window(name: str, start: str, end: str) -> DateWindow:
    from datetime import date

    return DateWindow(name=name, start=date.fromisoformat(start), end=date.fromisoformat(end))


def test_synthetic_source_coverage_fixtures() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    current = _window("current", "2026-02-16", "2026-02-22")
    previous = _window("previous", "2026-02-09", "2026-02-15")
    yoy = _window("yoy", "2025-02-17", "2025-02-23")

    for case in fixtures:
        payload = _build_metric_window_availability(
            current_window=current,
            previous_window=previous,
            yoy_window=yoy,
            gsc_data_coverage=case["gsc_data_coverage"],
            gsc_daily_rows=case["gsc_daily_rows"],
            gsc_feature_split=case["gsc_feature_split"],
            daily_serp_feature_shifts=case["daily_serp_feature_shifts"],
            daily_gsc_anomalies=case["daily_gsc_anomalies"],
            weather_summary=case["weather_summary"],
            additional_context=case["additional_context"],
        )
        by_source = {
            str(row.get("source", "")): row
            for row in payload.get("rows", [])
            if isinstance(row, dict)
        }
        for source_name, expected in case.get("expect", {}).items():
            row = by_source[source_name]
            assert bool(row.get("current_available", False)) is bool(
                expected.get("current_available", False)
            ), f"{case['name']}::{source_name} current_available mismatch"
            assert bool(row.get("yoy_available", False)) is bool(
                expected.get("yoy_available", False)
            ), f"{case['name']}::{source_name} yoy_available mismatch"
