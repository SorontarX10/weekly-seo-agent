from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from weekly_seo_agent.weekly_reporting_agent.config import AgentConfig
from weekly_seo_agent.weekly_reporting_agent.main import (
    _assert_llm_runtime_ready,
    _build_run_manifest,
    _enforce_llm_only_mode,
    _post_run_notification,
    _write_run_manifest,
)


def test_enforce_llm_only_mode_forces_flags() -> None:
    base = AgentConfig.from_env()
    cfg = replace(base, use_llm_analysis=False, use_llm_validator=False)
    enforced = _enforce_llm_only_mode(cfg)
    assert enforced.use_llm_analysis is True
    assert enforced.use_llm_validator is True


def test_assert_llm_runtime_ready_raises_when_gaia_not_ready() -> None:
    cfg = SimpleNamespace(
        use_llm_analysis=True,
        use_llm_validator=True,
        gaia_llm_enabled=False,
    )
    with pytest.raises(SystemExit):
        _assert_llm_runtime_ready(cfg)  # type: ignore[arg-type]


def test_post_run_notification_aggregates_counts() -> None:
    country_status = {
        "PL": {
            "country_code": "PL",
            "final_status": "success",
            "drive_status": "uploaded",
            "quality_gate_status": "passed",
            "quality_score": 91,
            "generation_status": "success",
        },
        "CZ": {
            "country_code": "CZ",
            "final_status": "failed",
            "drive_status": "failed",
            "quality_gate_status": "failed",
            "quality_score": 77,
            "generation_status": "success",
        },
    }
    summary = _post_run_notification(country_status)
    assert summary["total_countries"] == 2
    assert summary["success_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["uploaded_count"] == 1
    assert summary["gate_failed_count"] == 1


def test_build_and_write_run_manifest(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        use_llm_analysis=True,
        use_llm_validator=True,
        gaia_model="gpt-5.2",
        strict_llm_profile_enabled=True,
    )
    country_status = {
        "PL": {
            "country_code": "PL",
            "generation_status": "success",
            "quality_gate_status": "passed",
            "drive_status": "uploaded",
            "final_status": "success",
            "quality_score": 90,
            "quality_passed": True,
            "runtime_sec": 42.0,
            "error": "",
            "warnings": ["source stale"],
            "source_freshness": [{"source": "Weather", "status": "stale"}],
            "date_windows": {
                "current": {"start": "2026-02-16", "end": "2026-02-22"},
            },
        }
    }
    manifest = _build_run_manifest(
        run_date=date(2026, 2, 26),
        run_started_at=1000.0,
        run_finished_at=1100.0,
        config=cfg,  # type: ignore[arg-type]
        country_codes=["PL"],
        country_status=country_status,
        gate_failures=[],
    )
    assert manifest["schema_version"] == "weekly_run_manifest_v1"
    assert manifest["llm_mode"]["enforced"] is True
    assert manifest["countries_requested"] == ["PL"]
    assert manifest["summary"]["success_count"] == 1

    manifest_path = _write_run_manifest(tmp_path, date(2026, 2, 26), manifest)
    assert manifest_path.exists()
    payload = manifest_path.read_text(encoding="utf-8")
    assert "weekly_run_manifest_v1" in payload
