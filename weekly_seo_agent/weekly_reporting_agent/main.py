from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import shutil
from datetime import date
from pathlib import Path
import sys

from dotenv import find_dotenv, load_dotenv

from weekly_seo_agent.weekly_reporting_agent.clients.google_drive_client import GoogleDriveClient
from weekly_seo_agent.weekly_reporting_agent.config import AgentConfig
from weekly_seo_agent.weekly_reporting_agent.reporting import write_docx
from weekly_seo_agent.weekly_reporting_agent.time_windows import compute_monthly_windows
from weekly_seo_agent.weekly_reporting_agent.workflow import run_reporting_workflow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly SEO Intelligence Agent")
    parser.add_argument(
        "--run-date",
        dest="run_date",
        help="Execution date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--mode",
        choices=("weekly", "monthly"),
        default="weekly",
        help="Reporting mode: weekly (default) or monthly",
    )
    parser.add_argument(
        "--month",
        dest="target_month",
        help="Target month for monthly mode in YYYY-MM format. "
        "If omitted, monthly mode uses last fully completed month.",
    )
    parser.add_argument(
        "--trends-from-date",
        dest="trends_from_date",
        help="Start date (YYYY-MM-DD) for upcoming trends summary window.",
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
    report_mode: str,
    target_month: str,
    trends_from_date: str,
) -> dict[str, str]:
    country_config, senuto_country_id, gsc_country_filter = _build_country_config(
        config, country_code
    )
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = run_reporting_workflow(
        run_date,
        country_config,
        report_mode=report_mode,
        target_month=target_month,
        trends_from_date=trends_from_date,
    )
    final_report = state.get("final_report") or state["markdown_report"]
    if report_mode == "monthly":
        windows = compute_monthly_windows(run_date, target_month=target_month or None)
        current_window = windows["current_28d"]
        report_stem = (
            f"{current_window.start.strftime('%Y_%m')}_{country_code.lower()}_seo_report_Monthly"
        )
        title = f"Monthly SEO Report {current_window.start.strftime('%Y-%m')} ({country_code})"
    else:
        report_stem = f"{run_date.strftime('%Y_%m_%d')}_{country_code.lower()}_seo_weekly_report"
        title = f"Weekly SEO Report {run_date.isoformat()} ({country_code})"
    docx_path = output_dir / f"{report_stem}.docx"
    write_docx(
        docx_path,
        title=title,
        content=final_report,
    )
    return {
        "country_code": country_code,
        "docx_path": str(docx_path),
        "gsc_country_filter": gsc_country_filter,
        "senuto_country_id": str(senuto_country_id),
    }


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    total_safe = max(1, int(total))
    done_clamped = max(0, min(int(done), total_safe))
    filled = int(width * done_clamped / total_safe)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done_clamped}/{total_safe}"


def _print_progress(stage: str, done: int, total: int, *, finalize: bool = False) -> None:
    line = f"{stage}: {_progress_bar(done, total)}"
    if sys.stdout.isatty():
        end = "\n" if finalize else "\r"
        print(line, end=end, flush=True)
    else:
        # CI logs: keep each update as a separate line.
        print(line, flush=True)


def main() -> None:
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    args = _parse_args()
    run_date = _parse_run_date(args.run_date)
    report_mode = str(args.mode).strip().lower() or "weekly"
    target_month = str(args.target_month or "").strip()
    trends_from_date = str(args.trends_from_date or "").strip()
    config = AgentConfig.from_env()

    if report_mode == "monthly" and not trends_from_date:
        raise SystemExit(
            "Monthly mode requires --trends-from-date YYYY-MM-DD. "
            "Provide the date you want for upcoming trends analysis."
        )

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
        drive_folder_name = config.google_drive_folder_name
        if report_mode == "monthly":
            drive_folder_name = f"{drive_folder_name.rstrip('/')}/Monthly"
        drive_client = GoogleDriveClient(
            client_secret_path=config.google_drive_client_secret_path,
            token_path=config.google_drive_token_path,
            folder_name=drive_folder_name,
            folder_id=config.google_drive_folder_id,
        )

    failed_countries: list[tuple[str, str]] = []
    successful_runs: list[dict[str, str]] = []
    max_workers = max(1, min(len(country_codes), 4))
    print(f"Starting parallel batch: countries={','.join(country_codes)} | workers={max_workers}")
    _print_progress("Generation progress", 0, len(country_codes))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _run_country_report,
                run_date,
                config,
                country_code,
                str(output_dir),
                report_mode,
                target_month,
                trends_from_date,
            ): country_code
            for country_code in country_codes
        }
        completed_generation = 0
        for future in as_completed(future_map):
            country_code = future_map[future]
            try:
                result = future.result()
                successful_runs.append(result)
                print(
                    "Report generated: "
                    f"{result['docx_path']} | country={result['country_code']} | "
                    f"GSC={result['gsc_country_filter']} | Senuto country_id={result['senuto_country_id']}"
                )
            except Exception as exc:
                failed_countries.append((country_code, str(exc)))
                print(f"Country run failed: {country_code} | {exc}")
            finally:
                completed_generation += 1
                _print_progress(
                    "Generation progress",
                    completed_generation,
                    len(country_codes),
                    finalize=completed_generation >= len(country_codes),
                )

    # Retry failed countries once, sequentially (stabilizes transient API/process errors).
    if failed_countries:
        print("Retrying failed countries sequentially (single retry each)...")
        remaining_failures: list[tuple[str, str]] = []
        for country_code, first_error in failed_countries:
            try:
                result = _run_country_report(
                    run_date,
                    config,
                    country_code,
                    str(output_dir),
                    report_mode,
                    target_month,
                    trends_from_date,
                )
                successful_runs.append(result)
                print(
                    "Retry success: "
                    f"{result['docx_path']} | country={result['country_code']} | "
                    f"GSC={result['gsc_country_filter']} | Senuto country_id={result['senuto_country_id']}"
                )
            except Exception as retry_exc:
                message = f"initial={first_error} | retry={retry_exc}"
                remaining_failures.append((country_code, message))
                print(f"Retry failed: {country_code} | {message}")
        failed_countries = remaining_failures

    if drive_client is not None:
        drive_upload_errors: list[str] = []
        uploaded_countries: set[str] = set()
        total_uploads = len(successful_runs)
        completed_uploads = 0
        if total_uploads > 0:
            _print_progress("Upload progress", 0, total_uploads)
        for result in successful_runs:
            docx_path = Path(result["docx_path"])
            try:
                uploaded = drive_client.upload_docx_as_google_doc(docx_path)
                uploaded_countries.add(str(result.get("country_code", "")).upper())
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
            finally:
                if total_uploads > 0:
                    completed_uploads += 1
                    _print_progress(
                        "Upload progress",
                        completed_uploads,
                        total_uploads,
                        finalize=completed_uploads >= total_uploads,
                    )

        # Retry missing uploads once (per country) to reduce partial-export cases.
        expected_countries = {
            str(result.get("country_code", "")).upper()
            for result in successful_runs
            if str(result.get("country_code", "")).strip()
        }
        missing_uploads = sorted(expected_countries - uploaded_countries)
        if missing_uploads:
            print(
                "Retrying missing Google Drive uploads once for countries: "
                + ",".join(missing_uploads)
            )
            for missing_country in missing_uploads:
                candidate = next(
                    (
                        row
                        for row in successful_runs
                        if str(row.get("country_code", "")).upper() == missing_country
                    ),
                    None,
                )
                if not candidate:
                    continue
                docx_path = Path(str(candidate.get("docx_path", "")))
                try:
                    uploaded = drive_client.upload_docx_as_google_doc(docx_path)
                    uploaded_countries.add(missing_country)
                    print(
                        "Google Doc retry success: "
                        f"{uploaded.get('name')} | {uploaded.get('webViewLink', 'no-link')}"
                    )
                except Exception as retry_exc:
                    message = f"{missing_country}: retry failed: {retry_exc}"
                    drive_upload_errors.append(message)
                    print(f"Google Drive upload retry failed: {message}")

        if drive_upload_errors:
            print("Google Drive upload completed with errors (local DOCX reports are still generated):")
            for err in drive_upload_errors:
                print(f"- {err}")

    if not successful_runs:
        raise SystemExit(
            "Run failed: no country report was generated. "
            "Check credentials/secrets and country-level errors above."
        )

    if failed_countries:
        print("Run finished with country-level failures:")
        for code, message in failed_countries:
            print(f"- {code}: {message}")


if __name__ == "__main__":
    main()
