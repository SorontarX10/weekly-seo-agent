from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import shutil
from datetime import date, timedelta
from pathlib import Path
import time
from typing import Any

from dotenv import find_dotenv, load_dotenv

from weekly_seo_agent.weekly_reporting_agent.clients.continuity_client import ContinuityClient
from weekly_seo_agent.weekly_reporting_agent.clients.google_drive_client import GoogleDriveClient
from weekly_seo_agent.weekly_reporting_agent.clients.gsc_client import GSCClient
from weekly_seo_agent.weekly_reporting_agent.clients.merchant_center_client import (
    MerchantCenterClient,
)
from weekly_seo_agent.weekly_reporting_agent.config import AgentConfig
from weekly_seo_agent.weekly_reporting_agent.evaluation import evaluate_report_text
from weekly_seo_agent.weekly_reporting_agent.models import DateWindow
from weekly_seo_agent.weekly_reporting_agent.reporting import (
    enforce_manager_quality_guardrail,
    write_docx,
)
from weekly_seo_agent.weekly_reporting_agent.workflow import run_weekly_workflow


QUALITY_MAX_GAIA_MODEL = "gpt-5.2"
QUALITY_MAX_MIN_EVAL_SCORE = 88
RUNTIME_SOURCE_TOGGLES: dict[str, str] = {
    "news": "news_scraping_enabled",
    "weather": "weather_context_enabled",
    "trade-plan": "trade_plan_enabled",
    "events": "market_events_enabled",
    "platform-pulse": "platform_pulse_enabled",
    "free-public-sources": "free_public_sources_enabled",
    "status-log": "status_log_enabled",
    "product-trends": "product_trends_enabled",
    "weekly-news": "weekly_news_summary_enabled",
}


def _apply_weekly_quality_max_profile(config: AgentConfig) -> AgentConfig:
    """Force high-quality LLM settings for weekly reporter only.

    This profile is applied in weekly reporter runtime code and intentionally
    does not require any .env changes.
    """

    return replace(
        config,
        eval_gate_enabled=True,
        eval_gate_min_score=max(QUALITY_MAX_MIN_EVAL_SCORE, int(config.eval_gate_min_score)),
        eval_gate_block_drive_upload=True,
        use_llm_analysis=True,
        use_llm_validator=True,
        gaia_model=QUALITY_MAX_GAIA_MODEL,
        gaia_temperature=0.0,
        gaia_timeout_sec=max(180, int(config.gaia_timeout_sec)),
        gaia_max_retries=max(2, int(config.gaia_max_retries)),
        gaia_max_output_tokens=max(2200, int(config.gaia_max_output_tokens)),
        llm_map_max_tokens=max(900, int(config.llm_map_max_tokens)),
        llm_reduce_max_tokens=max(2200, int(config.llm_reduce_max_tokens)),
        llm_validator_max_tokens=max(1200, int(config.llm_validator_max_tokens)),
        llm_packet_max_chars=max(5200, int(config.llm_packet_max_chars)),
        llm_appendix_max_chars=max(2600, int(config.llm_appendix_max_chars)),
        llm_map_max_packets=max(6, int(config.llm_map_max_packets)),
        llm_validation_max_rounds=max(3, int(config.llm_validation_max_rounds)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly SEO Intelligence Agent")
    parser.add_argument(
        "--run-date",
        dest="run_date",
        help="Execution date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run startup preflight checks only and exit.",
    )
    merchant_group = parser.add_mutually_exclusive_group()
    merchant_group.add_argument(
        "--merchant-center",
        dest="merchant_center_enabled",
        action="store_true",
        help="Enable Merchant Center enrichment/preflight for this run.",
    )
    merchant_group.add_argument(
        "--no-merchant-center",
        dest="merchant_center_enabled",
        action="store_false",
        help="Disable Merchant Center enrichment/preflight for this run.",
    )
    parser.set_defaults(merchant_center_enabled=None)

    strict_profile_group = parser.add_mutually_exclusive_group()
    strict_profile_group.add_argument(
        "--strict-llm-profile",
        dest="strict_llm_profile_enabled",
        action="store_true",
        help="Force strict quality-max LLM profile for weekly reporter run.",
    )
    strict_profile_group.add_argument(
        "--no-strict-llm-profile",
        dest="strict_llm_profile_enabled",
        action="store_false",
        help="Use base LLM settings from environment for this run.",
    )
    parser.set_defaults(strict_llm_profile_enabled=None)
    parser.add_argument(
        "--enable-source",
        action="append",
        choices=sorted(RUNTIME_SOURCE_TOGGLES.keys()),
        default=[],
        help="Enable specific data source for this run (can be repeated).",
    )
    parser.add_argument(
        "--disable-source",
        action="append",
        choices=sorted(RUNTIME_SOURCE_TOGGLES.keys()),
        default=[],
        help="Disable specific data source for this run (can be repeated).",
    )
    return parser.parse_args()


def _parse_run_date(raw: str | None) -> date:
    if not raw:
        return date.today()
    return date.fromisoformat(raw)


def _apply_runtime_toggles(
    config: AgentConfig,
    merchant_center_enabled: bool | None,
    strict_llm_profile_enabled: bool | None,
    enable_sources: list[str],
    disable_sources: list[str],
) -> AgentConfig:
    updated = config
    if merchant_center_enabled is not None:
        updated = replace(
            updated,
            merchant_center_enabled=bool(merchant_center_enabled),
        )
    if strict_llm_profile_enabled is not None:
        updated = replace(
            updated,
            strict_llm_profile_enabled=bool(strict_llm_profile_enabled),
        )
    field_overrides: dict[str, bool] = {}
    for source_name in enable_sources:
        field_name = RUNTIME_SOURCE_TOGGLES.get(source_name)
        if field_name:
            field_overrides[field_name] = True
    for source_name in disable_sources:
        field_name = RUNTIME_SOURCE_TOGGLES.get(source_name)
        if field_name:
            field_overrides[field_name] = False
    if field_overrides:
        updated = replace(updated, **field_overrides)
    if updated.strict_llm_profile_enabled:
        return _apply_weekly_quality_max_profile(updated)
    return updated


def _run_merchant_center_health_check(
    config: AgentConfig, country_codes: list[str]
) -> None:
    if not config.merchant_center_enabled:
        print("Merchant Center preflight: disabled.")
        return

    if not config.merchant_center_api_enabled:
        raise SystemExit(
            "Merchant Center preflight failed: MERCHANT_CENTER_ENABLED=true but "
            "MERCHANT_CENTER_CREDENTIALS_PATH is missing."
        )

    client = MerchantCenterClient(
        credentials_path=config.merchant_center_credentials_path,
        mca_id=config.merchant_center_mca_id,
        account_id_map=config.merchant_center_account_id_map,
    )
    try:
        validation = client.validate_country_access(country_codes)
    except Exception as exc:
        raise SystemExit(
            "Merchant Center preflight failed: "
            f"{exc}"
        ) from exc
    missing = validation.get("missing_country_mappings") or []
    inaccessible = validation.get("inaccessible_country_accounts") or {}
    if missing or inaccessible:
        missing_text = ", ".join(str(item) for item in missing) or "none"
        inaccessible_text = (
            ", ".join(
                f"{country}:{account_id}"
                for country, account_id in sorted(
                    (inaccessible or {}).items(), key=lambda item: item[0]
                )
            )
            or "none"
        )
        raise SystemExit(
            "Merchant Center preflight failed: "
            f"missing country->account mapping for [{missing_text}]; "
            f"inaccessible mapped account IDs [{inaccessible_text}]. "
            "Grant access for the configured service account and verify "
            "MERCHANT_CENTER_ACCOUNT_ID_MAP / MERCHANT_CENTER_MCA_ID."
        )

    validated = validation.get("validated_country_accounts") or {}
    validated_text = ", ".join(
        f"{country}:{account_id}"
        for country, account_id in sorted(validated.items(), key=lambda item: item[0])
    )
    account_email = str(validation.get("service_account_email", "")).strip() or "unknown"
    print(
        "Merchant Center preflight: OK | "
        f"service_account={account_email} | "
        f"country_accounts={validated_text}"
    )


def _preflight_severity(config: AgentConfig, source_group: str) -> str:
    blockers = {token.strip().lower() for token in config.startup_preflight_blocking_sources}
    return "blocker" if source_group.strip().lower() in blockers else "warning"


def _append_preflight_row(
    rows: list[dict[str, str]],
    *,
    source_group: str,
    source_label: str,
    status: str,
    details: str,
    config: AgentConfig,
) -> None:
    rows.append(
        {
            "source_group": source_group,
            "source": source_label,
            "status": status,
            "severity": _preflight_severity(config, source_group),
            "details": details,
        }
    )


def _is_gsc_mapping_explicit(config: AgentConfig, country_codes: list[str]) -> bool:
    if len(country_codes) <= 1:
        return True
    for code in country_codes:
        if code not in config.gsc_site_url_map:
            return False
    return True


def _check_gsc_country_access(
    config: AgentConfig,
    run_date: date,
    country_code: str,
) -> tuple[bool, str]:
    country_config, _, _ = _build_country_config(config, country_code)
    probe_day = run_date - timedelta(days=3)
    window = DateWindow(name="preflight_probe", start=probe_day, end=probe_day)
    client = GSCClient(
        site_url=country_config.gsc_site_url,
        credentials_path=country_config.gsc_credentials_path,
        oauth_client_secret_path=country_config.gsc_oauth_client_secret_path,
        oauth_refresh_token=country_config.gsc_oauth_refresh_token,
        oauth_token_uri=country_config.gsc_oauth_token_uri,
        country_filter=country_config.gsc_country_filter,
        row_limit=max(100, int(country_config.gsc_row_limit)),
    )
    summary = client.fetch_totals(window)
    return (
        True,
        f"site={country_config.gsc_site_url} filter={country_config.gsc_country_filter or 'none'} "
        f"probe_day={probe_day.isoformat()} clicks={summary.clicks:.0f}",
    )


def _build_continuity_client(config: AgentConfig) -> ContinuityClient:
    return ContinuityClient(
        client_secret_path=config.google_drive_client_secret_path,
        token_path=config.google_drive_token_path,
        reports_folder_name=config.google_drive_folder_name,
        reports_folder_id=config.google_drive_folder_id,
        status_file_reference=config.status_file_reference,
        max_recent_reports=max(1, int(config.historical_reports_count)),
        yoy_tolerance_days=max(7, int(config.historical_reports_yoy_tolerance_days)),
        max_status_rows=max(1, int(config.status_max_rows)),
    )


def _check_sheet_reference(
    continuity: ContinuityClient, reference: str
) -> tuple[bool, str]:
    ref = reference.strip()
    if not ref:
        return False, "reference is empty"
    meta = continuity._resolve_sheet_file(ref)
    if not isinstance(meta, dict):
        return False, "sheet not found or not accessible"
    sheet_id = str(meta.get("id", "")).strip()
    sheet_name = str(meta.get("name", "")).strip()
    if not sheet_id:
        return False, "sheet id missing"
    return True, f"id={sheet_id} name={sheet_name}"


def _check_sheet_tab_access(
    continuity: ContinuityClient,
    spreadsheet_id: str,
    tab_name: str,
) -> tuple[bool, str]:
    sheet_id = spreadsheet_id.strip()
    tab = tab_name.strip()
    if not sheet_id:
        return False, "spreadsheet id missing"
    if not tab:
        return False, "tab is empty"
    safe_tab = tab.replace("'", "''")
    try:
        continuity._sheets_service().spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"'{safe_tab}'!A1:A1",
            majorDimension="ROWS",
        ).execute()
        return True, f"tab={tab}"
    except Exception as exc:
        return False, f"tab={tab} inaccessible: {exc}"


def _run_startup_preflight(
    config: AgentConfig,
    run_date: date,
    country_codes: list[str],
) -> None:
    if not config.startup_preflight_enabled:
        print("Startup preflight: disabled.")
        return

    rows: list[dict[str, str]] = []

    # GSC mapping and access checks.
    explicit_mapping = _is_gsc_mapping_explicit(config, country_codes)
    if explicit_mapping:
        _append_preflight_row(
            rows,
            source_group="gsc",
            source_label="GSC property mapping",
            status="ready",
            details="All selected countries have explicit entries in GSC_SITE_URL_MAP.",
            config=config,
        )
    else:
        _append_preflight_row(
            rows,
            source_group="gsc",
            source_label="GSC property mapping",
            status="warn",
            details="Some countries rely on fallback GSC_SITE_URL; explicit per-country mapping is recommended.",
            config=config,
        )
    for country_code in country_codes:
        try:
            ok, details = _check_gsc_country_access(config, run_date, country_code)
            _append_preflight_row(
                rows,
                source_group="gsc",
                source_label=f"GSC access {country_code}",
                status="ready" if ok else "error",
                details=details,
                config=config,
            )
        except Exception as exc:
            _append_preflight_row(
                rows,
                source_group="gsc",
                source_label=f"GSC access {country_code}",
                status="error",
                details=str(exc),
                config=config,
            )

    continuity: ContinuityClient | None = None
    continuity_error = ""
    if not config.google_drive_client_secret_path.strip():
        continuity_error = "GOOGLE_DRIVE_CLIENT_SECRET_PATH is missing."
    else:
        try:
            continuity = _build_continuity_client(config)
        except Exception as exc:
            continuity_error = str(exc)

    if continuity is None:
        _append_preflight_row(
            rows,
            source_group="sheets",
            source_label="Google Sheets credentials",
            status="error",
            details=continuity_error or "Unable to initialize continuity client.",
            config=config,
        )
    else:
        _append_preflight_row(
            rows,
            source_group="sheets",
            source_label="Google Sheets credentials",
            status="ready",
            details="Continuity client initialized.",
            config=config,
        )

    # Sheets source availability checks.
    if continuity is not None:
        if config.status_log_enabled:
            ok, details = _check_sheet_reference(continuity, config.status_file_reference)
            _append_preflight_row(
                rows,
                source_group="sheets",
                source_label="Status log sheet",
                status="ready" if ok else "warn",
                details=details,
                config=config,
            )
        else:
            _append_preflight_row(
                rows,
                source_group="sheets",
                source_label="Status log sheet",
                status="skipped",
                details="Disabled by status_log_enabled=false.",
                config=config,
            )

        if config.product_trends_enabled:
            for label, reference in (
                ("Product trends comparison sheet", config.product_trends_comparison_sheet_reference),
                ("Product trends upcoming sheet", config.product_trends_upcoming_sheet_reference),
                ("Product trends current sheet", config.product_trends_current_sheet_reference),
            ):
                ok, details = _check_sheet_reference(continuity, reference)
                _append_preflight_row(
                    rows,
                    source_group="sheets",
                    source_label=label,
                    status="ready" if ok else "warn",
                    details=details,
                    config=config,
                )
        else:
            _append_preflight_row(
                rows,
                source_group="sheets",
                source_label="Product trend sheets",
                status="skipped",
                details="Disabled by product_trends_enabled=false.",
                config=config,
            )

        if config.trade_plan_enabled:
            trade_ref_ok, trade_ref_details = _check_sheet_reference(
                continuity, config.trade_plan_sheet_reference
            )
            _append_preflight_row(
                rows,
                source_group="sheets",
                source_label="Trade plan sheet",
                status="ready" if trade_ref_ok else "warn",
                details=trade_ref_details,
                config=config,
            )
            if trade_ref_ok:
                trade_meta = continuity._resolve_sheet_file(config.trade_plan_sheet_reference)
                trade_sheet_id = str((trade_meta or {}).get("id", "")).strip() if isinstance(trade_meta, dict) else ""
                for country_code in country_codes:
                    tab_name = config.trade_plan_tab_map.get(country_code, "").strip()
                    ok, details = _check_sheet_tab_access(continuity, trade_sheet_id, tab_name)
                    _append_preflight_row(
                        rows,
                        source_group="sheets",
                        source_label=f"Trade plan tab {country_code}",
                        status="ready" if ok else "warn",
                        details=details,
                        config=config,
                    )
        else:
            _append_preflight_row(
                rows,
                source_group="sheets",
                source_label="Trade plan sheet",
                status="skipped",
                details="Disabled by trade_plan_enabled=false.",
                config=config,
            )

    # Drive publish readiness check.
    if config.google_drive_upload_enabled:
        try:
            drive_client = GoogleDriveClient(
                client_secret_path=config.google_drive_client_secret_path,
                token_path=config.google_drive_token_path,
                folder_name=config.google_drive_folder_name,
                folder_id=config.google_drive_folder_id,
            )
            folder_id = drive_client._find_or_create_folder()
            _append_preflight_row(
                rows,
                source_group="drive",
                source_label="Google Drive publish access",
                status="ready",
                details=f"target_folder_id={folder_id}",
                config=config,
            )
        except Exception as exc:
            _append_preflight_row(
                rows,
                source_group="drive",
                source_label="Google Drive publish access",
                status="error",
                details=str(exc),
                config=config,
            )
    else:
        _append_preflight_row(
            rows,
            source_group="drive",
            source_label="Google Drive publish access",
            status="skipped",
            details="Disabled by GOOGLE_DRIVE_ENABLED=false or missing Drive config.",
            config=config,
        )

    print("Startup preflight matrix:")
    for row in rows:
        print(
            f"- [{row['status'].upper():7}] [{row['severity'].upper():8}] "
            f"{row['source']}: {row['details']}"
        )

    blocking_errors = [
        row
        for row in rows
        if row.get("status") == "error" and row.get("severity") == "blocker"
    ]
    if blocking_errors:
        summary = "; ".join(
            f"{row.get('source')}: {row.get('details')}" for row in blocking_errors
        )
        raise SystemExit(
            "Startup preflight failed on blocker sources. "
            f"Details: {summary}"
        )


def _clear_same_day_outputs(
    output_dir: Path,
    run_date: date,
    country_codes: list[str],
) -> None:
    if not output_dir.exists():
        return
    prefix = f"{run_date.strftime('%Y_%m_%d')}_"
    allowed_prefixes = {
        f"{prefix}{code.strip().lower()}_seo_weekly_report"
        for code in country_codes
        if code.strip()
    }
    # Backward compatibility for legacy single-country naming.
    allowed_prefixes.add(f"{prefix}seo_weekly_report")
    for entry in output_dir.iterdir():
        if not entry.name.startswith(prefix):
            continue
        if not any(entry.name.startswith(chunk) for chunk in allowed_prefixes):
            continue
        if entry.is_file() or entry.is_symlink():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)


def _gsc_country_filter_for_country(country_code: str) -> str:
    code = country_code.strip().upper()
    mapping = {
        "PL": "pol",
        "CZ": "cze",
        "SK": "svk",
        "HU": "hun",
    }
    return mapping.get(code, code.lower())


def _build_country_config(config: AgentConfig, country_code: str) -> tuple[AgentConfig, int, str]:
    gsc_country_filter = _gsc_country_filter_for_country(country_code)
    gsc_site_url = config.gsc_site_url_map.get(
        country_code,
        config.gsc_site_url,
    )
    senuto_country_id = config.senuto_country_id_map.get(
        country_code, config.senuto_country_id
    )
    weather_latitude = config.weather_latitude_map.get(
        country_code,
        config.weather_latitude,
    )
    weather_longitude = config.weather_longitude_map.get(
        country_code,
        config.weather_longitude,
    )
    weather_label = config.weather_label_map.get(
        country_code,
        config.weather_label,
    )
    holidays_country_code = config.holidays_country_code_map.get(
        country_code,
        country_code,
    )
    holidays_language_code = config.holidays_language_code_map.get(
        country_code,
        config.holidays_language_code,
    )
    google_trends_rss_url = config.google_trends_rss_url_map.get(
        country_code,
        config.google_trends_rss_url,
    )
    nbp_api_base_url = config.nbp_api_base_url if country_code == "PL" else ""
    imgw_warnings_url = config.imgw_warnings_url if country_code == "PL" else ""
    news_rss_urls_pl = config.news_rss_urls_pl if country_code == "PL" else ()
    news_html_urls_pl = config.news_html_urls_pl if country_code == "PL" else ()
    country_config = replace(
        config,
        report_country_code=country_code,
        gsc_site_url=gsc_site_url,
        gsc_country_filter=gsc_country_filter,
        senuto_country_id=senuto_country_id,
        weather_latitude=weather_latitude,
        weather_longitude=weather_longitude,
        weather_label=weather_label,
        holidays_country_code=holidays_country_code,
        nager_holidays_country_code=holidays_country_code,
        holidays_language_code=holidays_language_code,
        google_trends_rss_url=google_trends_rss_url,
        nbp_api_base_url=nbp_api_base_url,
        imgw_warnings_url=imgw_warnings_url,
        news_rss_urls_pl=news_rss_urls_pl,
        news_html_urls_pl=news_html_urls_pl,
    )
    return country_config, senuto_country_id, gsc_country_filter


def _run_country_report(
    run_date: date,
    config: AgentConfig,
    country_code: str,
    output_dir_str: str,
) -> dict[str, str]:
    started = time.time()
    country_config, senuto_country_id, gsc_country_filter = _build_country_config(
        config, country_code
    )
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = run_weekly_workflow(run_date, country_config)
    final_report = state.get("final_report") or state["markdown_report"]
    final_report = enforce_manager_quality_guardrail(final_report, max_words=1260)
    quality = evaluate_report_text(final_report)
    report_stem = f"{run_date.strftime('%Y_%m_%d')}_{country_code.lower()}_seo_weekly_report"
    docx_path = output_dir / f"{report_stem}.docx"
    write_docx(
        docx_path,
        title=f"Weekly SEO Report {run_date.isoformat()} ({country_code})",
        content=final_report,
    )
    return {
        "country_code": country_code,
        "docx_path": str(docx_path),
        "gsc_country_filter": gsc_country_filter,
        "senuto_country_id": str(senuto_country_id),
        "quality_score": str(int(quality.get("score", 0) or 0)),
        "quality_passed": "true" if bool(quality.get("passed", False)) else "false",
        "quality_issues": json.dumps(quality.get("issues", []), ensure_ascii=False),
        "quality_metrics": json.dumps(quality.get("metrics", {}), ensure_ascii=False),
        "runtime_sec": f"{(time.time() - started):.2f}",
    }


def _apply_quality_gate(
    run_results: list[dict[str, str]],
    *,
    gate_enabled: bool,
    min_score: int,
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    if not gate_enabled:
        return run_results, []

    gated_runs: list[dict[str, str]] = []
    gate_failures: list[tuple[str, str]] = []
    for result in run_results:
        score = int(float(result.get("quality_score", "0") or 0.0))
        passed = str(result.get("quality_passed", "")).lower() == "true"
        if score < int(min_score) or not passed:
            gate_failures.append(
                (
                    str(result.get("country_code", "unknown")),
                    "Evaluation gate failed: "
                    f"score={score}/100 (min {int(min_score)}), "
                    f"passed={passed}, issues={result.get('quality_issues', '[]')}",
                )
            )
            continue
        gated_runs.append(result)
    return gated_runs, gate_failures


def _should_publish_to_drive(
    config: AgentConfig,
    successful_runs: list[dict[str, str]],
    gate_failures: list[tuple[str, str]],
) -> bool:
    if not successful_runs:
        return False
    if not config.eval_gate_enabled:
        return True
    if not config.eval_gate_block_drive_upload:
        return True
    # Strict acceptance policy: publish only when every generated report
    # passes the quality gate for the current batch.
    return not gate_failures


def main() -> None:
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    args = _parse_args()
    run_date = _parse_run_date(args.run_date)
    base_config = AgentConfig.from_env()
    config = _apply_runtime_toggles(
        base_config,
        merchant_center_enabled=args.merchant_center_enabled,
        strict_llm_profile_enabled=args.strict_llm_profile_enabled,
        enable_sources=list(args.enable_source or []),
        disable_sources=list(args.disable_source or []),
    )

    if not config.gsc_enabled:
        raise SystemExit(
            "GSC is not configured. Provide either "
            "GSC_CREDENTIALS_PATH or GSC_OAUTH_CLIENT_SECRET_PATH + GSC_OAUTH_REFRESH_TOKEN."
        )

    country_codes = [code.strip().upper() for code in config.report_countries if code.strip()]
    if not country_codes:
        country_codes = [config.report_country_code.strip().upper() or "PL"]

    _run_startup_preflight(config, run_date, country_codes)

    if args.preflight_only:
        print("Preflight-only mode: checks completed, workflow run skipped.")
        return

    _run_merchant_center_health_check(config, country_codes)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_same_day_outputs(output_dir, run_date, country_codes)

    drive_client: GoogleDriveClient | None = None
    if config.google_drive_upload_enabled:
        drive_client = GoogleDriveClient(
            client_secret_path=config.google_drive_client_secret_path,
            token_path=config.google_drive_token_path,
            folder_name=config.google_drive_folder_name,
            folder_id=config.google_drive_folder_id,
        )

    failed_countries: list[tuple[str, str]] = []
    successful_runs: list[dict[str, str]] = []
    telemetry_rows: list[dict[str, str]] = []
    max_workers = max(1, min(len(country_codes), 4))
    force_serial = len(country_codes) == 1
    mode_label = "serial" if force_serial else "parallel"
    print(
        f"Starting {mode_label} batch: countries={','.join(country_codes)} | workers={1 if force_serial else max_workers}"
    )

    if force_serial:
        country_code = country_codes[0]
        try:
            result = _run_country_report(
                run_date=run_date,
                config=config,
                country_code=country_code,
                output_dir_str=str(output_dir),
            )
            successful_runs.append(result)
            telemetry_rows.append(result)
            print(
                "Report generated: "
                f"{result['docx_path']} | country={result['country_code']} | "
                f"GSC={result['gsc_country_filter']} | Senuto country_id={result['senuto_country_id']} | "
                f"quality={result.get('quality_score', '0')}/100"
            )
        except Exception as exc:
            failed_countries.append((country_code, str(exc)))
            print(f"Country run failed: {country_code} | {exc}")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _run_country_report,
                    run_date,
                    config,
                    country_code,
                    str(output_dir),
                ): country_code
                for country_code in country_codes
            }
            for future in as_completed(future_map):
                country_code = future_map[future]
                try:
                    result = future.result()
                    successful_runs.append(result)
                    telemetry_rows.append(result)
                    print(
                        "Report generated: "
                        f"{result['docx_path']} | country={result['country_code']} | "
                        f"GSC={result['gsc_country_filter']} | Senuto country_id={result['senuto_country_id']} | "
                        f"quality={result.get('quality_score', '0')}/100"
                    )
                except Exception as exc:
                    failed_countries.append((country_code, str(exc)))
                    print(f"Country run failed: {country_code} | {exc}")

    successful_runs, gate_failures = _apply_quality_gate(
        successful_runs,
        gate_enabled=bool(config.eval_gate_enabled),
        min_score=int(config.eval_gate_min_score),
    )
    if gate_failures:
        failed_countries.extend(gate_failures)

    if drive_client is not None and _should_publish_to_drive(
        config, successful_runs, gate_failures
    ):
        drive_upload_errors: list[str] = []
        for result in successful_runs:
            docx_path = Path(result["docx_path"])
            try:
                uploaded = drive_client.upload_docx_as_google_doc(docx_path)
                print(
                    "Google Doc created: "
                    f"{uploaded.get('name')} | {uploaded.get('webViewLink', 'no-link')}"
                )
            except Exception as exc:
                message = (
                    f"{result.get('country_code', 'unknown')}: {exc}"
                )
                drive_upload_errors.append(message)
                print(f"Google Drive upload failed: {message}")
        if drive_upload_errors:
            print("Google Drive upload completed with errors (local DOCX reports are still generated):")
            for err in drive_upload_errors:
                print(f"- {err}")
    elif drive_client is not None and successful_runs and gate_failures:
        print(
            "Google Drive upload skipped: at least one country failed the quality gate "
            "and eval_gate_block_drive_upload=true."
        )

    if config.telemetry_enabled and telemetry_rows:
        telemetry_dir = output_dir / "_telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = telemetry_dir / f"{run_date.strftime('%Y_%m_%d')}_weekly_reporting_observability.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as handle:
            for row in telemetry_rows:
                handle.write(
                    json.dumps(
                        {
                            "country_code": row.get("country_code", ""),
                            "docx_path": row.get("docx_path", ""),
                            "quality_score": int(float(row.get("quality_score", "0") or 0.0)),
                            "quality_passed": str(row.get("quality_passed", "")).lower() == "true",
                            "runtime_sec": float(row.get("runtime_sec", "0.0") or 0.0),
                            "quality_issues": json.loads(str(row.get("quality_issues", "[]") or "[]")),
                            "quality_metrics": json.loads(str(row.get("quality_metrics", "{}") or "{}")),
                            "timestamp": time.time(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Observability log written: {telemetry_path}")

    if not successful_runs:
        raise SystemExit(
            "Run failed: no country report was generated. "
            "Check credentials/secrets and country-level errors above."
        )

    if failed_countries:
        print("Run finished with country-level failures:")
        for code, message in failed_countries:
            print(f"- {code}: {message}")
        if config.eval_gate_enabled:
            raise SystemExit("Run failed quality/evaluation gate for at least one country.")


if __name__ == "__main__":
    main()
