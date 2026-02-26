from __future__ import annotations

from types import SimpleNamespace

from weekly_seo_agent.weekly_reporting_agent.main import (
    _apply_quality_gate,
    _should_publish_to_drive,
)


def test_apply_quality_gate_filters_failed_countries() -> None:
    run_results = [
        {
            "country_code": "PL",
            "quality_score": "91",
            "quality_passed": "true",
            "quality_issues": "[]",
        },
        {
            "country_code": "CZ",
            "quality_score": "84",
            "quality_passed": "true",
            "quality_issues": "[\"YoY coverage too limited\"]",
        },
        {
            "country_code": "SK",
            "quality_score": "93",
            "quality_passed": "false",
            "quality_issues": "[\"Missing required section\"]",
        },
    ]

    passed, failed = _apply_quality_gate(
        run_results,
        gate_enabled=True,
        min_score=88,
    )
    assert [row["country_code"] for row in passed] == ["PL"]
    assert [code for code, _ in failed] == ["CZ", "SK"]


def test_should_publish_to_drive_blocks_when_any_gate_failure_exists() -> None:
    cfg = SimpleNamespace(
        eval_gate_enabled=True,
        eval_gate_block_drive_upload=True,
    )
    successful_runs = [{"country_code": "PL", "quality_score": "90", "quality_passed": "true"}]
    gate_failures = [("CZ", "Evaluation gate failed")]
    assert _should_publish_to_drive(cfg, successful_runs, gate_failures) is False


def test_should_publish_to_drive_allows_when_gate_disabled() -> None:
    cfg = SimpleNamespace(
        eval_gate_enabled=False,
        eval_gate_block_drive_upload=True,
    )
    successful_runs = [{"country_code": "PL", "quality_score": "80", "quality_passed": "false"}]
    assert _should_publish_to_drive(cfg, successful_runs, gate_failures=[]) is True
