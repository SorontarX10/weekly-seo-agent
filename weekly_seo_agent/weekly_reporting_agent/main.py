from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import shutil
from datetime import date
from pathlib import Path
import time

from dotenv import find_dotenv, load_dotenv

from weekly_seo_agent.weekly_reporting_agent.clients.google_drive_client import GoogleDriveClient
from weekly_seo_agent.weekly_reporting_agent.config import AgentConfig
from weekly_seo_agent.weekly_reporting_agent.evaluation import evaluate_report_text
from weekly_seo_agent.weekly_reporting_agent.reporting import (
    enforce_manager_quality_guardrail,
    write_docx,
)
from weekly_seo_agent.weekly_reporting_agent.workflow import run_weekly_workflow


QUALITY_MAX_GAIA_MODEL = "gpt-5.2"
QUALITY_MAX_MIN_EVAL_SCORE = 88


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
    return parser.parse_args()


def _parse_run_date(raw: str | None) -> date:
    if not raw:
        return date.today()
    return date.fromisoformat(raw)


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


def main() -> None:
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    args = _parse_args()
    run_date = _parse_run_date(args.run_date)
    config = _apply_weekly_quality_max_profile(AgentConfig.from_env())

    if not config.gsc_enabled:
        raise SystemExit(
            "GSC is not configured. Provide either "
            "GSC_CREDENTIALS_PATH or GSC_OAUTH_CLIENT_SECRET_PATH + GSC_OAUTH_REFRESH_TOKEN."
        )

    country_codes = [code.strip().upper() for code in config.report_countries if code.strip()]
    if not country_codes:
        country_codes = [config.report_country_code.strip().upper() or "PL"]

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

    gated_runs: list[dict[str, str]] = []
    for result in successful_runs:
        score = int(float(result.get("quality_score", "0") or 0.0))
        passed = str(result.get("quality_passed", "")).lower() == "true"
        if config.eval_gate_enabled and (score < int(config.eval_gate_min_score) or not passed):
            failed_countries.append(
                (
                    str(result.get("country_code", "unknown")),
                    "Evaluation gate failed: "
                    f"score={score}/100 (min {int(config.eval_gate_min_score)}), "
                    f"passed={passed}, issues={result.get('quality_issues', '[]')}",
                )
            )
            continue
        gated_runs.append(result)
    successful_runs = gated_runs

    if drive_client is not None and (not config.eval_gate_enabled or not config.eval_gate_block_drive_upload or successful_runs):
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
