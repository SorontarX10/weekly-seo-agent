from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from typing import Any

import requests

from weekly_seo_agent.weekly_reporting_agent.models import DateWindow


DEFAULT_OPEN_HOLIDAYS_API_BASE_URL = "https://openholidaysapi.org"
DEFAULT_OPEN_HOLIDAYS_LANGUAGE_CODE = "PL"


# Proxy ekonomiczne wykorzystywane do przyblizenia "sily GMV" regionu.
ECONOMIC_PROFILES_BY_VOIVODESHIP: dict[str, dict[str, float]] = {
    "dolnoslaskie": {"population_m": 2.88, "avg_salary_pln": 8620.0},
    "kujawsko-pomorskie": {"population_m": 2.00, "avg_salary_pln": 7410.0},
    "lubelskie": {"population_m": 2.08, "avg_salary_pln": 7240.0},
    "lubuskie": {"population_m": 0.99, "avg_salary_pln": 7510.0},
    "lodzkie": {"population_m": 2.39, "avg_salary_pln": 7730.0},
    "malopolskie": {"population_m": 3.45, "avg_salary_pln": 8060.0},
    "mazowieckie": {"population_m": 5.52, "avg_salary_pln": 9860.0},
    "opolskie": {"population_m": 0.93, "avg_salary_pln": 7480.0},
    "podkarpackie": {"population_m": 2.05, "avg_salary_pln": 7060.0},
    "podlaskie": {"population_m": 1.16, "avg_salary_pln": 7350.0},
    "pomorskie": {"population_m": 2.37, "avg_salary_pln": 8590.0},
    "slaskie": {"population_m": 4.32, "avg_salary_pln": 8740.0},
    "swietokrzyskie": {"population_m": 1.15, "avg_salary_pln": 7060.0},
    "warminsko-mazurskie": {"population_m": 1.36, "avg_salary_pln": 7010.0},
    "wielkopolskie": {"population_m": 3.51, "avg_salary_pln": 8190.0},
    "zachodniopomorskie": {"population_m": 1.67, "avg_salary_pln": 7850.0},
}


def _normalize_text(value: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"\s+", " ", ascii_text).strip()


def _pick_localized_name(values: object, language_code: str) -> str:
    if isinstance(values, list):
        normalized_lang = language_code.strip().lower()
        for row in values:
            if not isinstance(row, dict):
                continue
            row_lang = str(row.get("language", "")).strip().lower()
            if row_lang == normalized_lang:
                text = str(row.get("text", "")).strip()
                if text:
                    return text
        for row in values:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if text:
                return text
    return ""


def _iter_window_days(window: DateWindow):
    for offset in range(window.days):
        yield window.start + timedelta(days=offset)


def _request_json_list(url: str, params: dict[str, str]) -> list[dict[str, Any]]:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _load_subdivisions(
    country_code: str,
    language_code: str,
    api_base_url: str,
) -> dict[str, str]:
    url = f"{api_base_url.rstrip('/')}/Subdivisions"
    rows = _request_json_list(
        url,
        params={"countryIsoCode": country_code},
    )

    result: dict[str, str] = {}
    for row in rows:
        code = str(row.get("code", "")).strip()
        if not code:
            continue
        name = _pick_localized_name(row.get("name"), language_code) or code
        result[code] = name
    return result


def _load_school_ferie_periods(
    valid_from: date,
    valid_to: date,
    country_code: str,
    language_code: str,
    api_base_url: str,
) -> list[dict[str, object]]:
    url = f"{api_base_url.rstrip('/')}/SchoolHolidays"
    rows = _request_json_list(
        url,
        params={
            "countryIsoCode": country_code,
            "languageIsoCode": language_code,
            "validFrom": valid_from.isoformat(),
            "validTo": valid_to.isoformat(),
        },
    )

    periods: list[dict[str, object]] = []
    for row in rows:
        name = _pick_localized_name(row.get("name"), language_code)
        normalized_name = _normalize_text(name)
        if "ferie" not in normalized_name:
            continue

        try:
            start_day = date.fromisoformat(str(row.get("startDate", "")).strip())
            end_day = date.fromisoformat(str(row.get("endDate", "")).strip())
        except ValueError:
            continue
        if end_day < start_day:
            continue

        subdivisions = row.get("subdivisions")
        subdivision_codes: tuple[str, ...]
        if isinstance(subdivisions, list):
            subdivision_codes = tuple(
                str(item.get("code", "")).strip()
                for item in subdivisions
                if isinstance(item, dict) and str(item.get("code", "")).strip()
            )
        else:
            subdivision_codes = ()

        periods.append(
            {
                "name": name or "Ferie zimowe",
                "start": start_day,
                "end": end_day,
                "subdivision_codes": subdivision_codes,
            }
        )

    return periods


def _build_gmv_weights(subdivision_names: dict[str, str]) -> tuple[dict[str, float], list[dict[str, object]]]:
    raw: dict[str, float] = {}
    ranked: list[dict[str, object]] = []
    for code, name in subdivision_names.items():
        normalized_name = _normalize_text(name)
        profile = ECONOMIC_PROFILES_BY_VOIVODESHIP.get(normalized_name)
        if not profile:
            continue
        value = profile["population_m"] * profile["avg_salary_pln"]
        raw[code] = value
        ranked.append(
            {
                "code": code,
                "name": name,
                "population_m": profile["population_m"],
                "avg_salary_pln": profile["avg_salary_pln"],
                "gmv_share": 0.0,
            }
        )

    total = sum(raw.values())
    if total <= 0:
        return {}, ranked

    weights = {code: value / total for code, value in raw.items()}
    for row in ranked:
        row["gmv_share"] = weights.get(str(row["code"]), 0.0)
    ranked.sort(key=lambda item: float(item.get("gmv_share", 0.0)), reverse=True)
    return weights, ranked


def _build_day_index(
    periods: list[dict[str, object]],
    valid_from: date,
    valid_to: date,
) -> dict[date, set[str]]:
    day_index: dict[date, set[str]] = {}

    for period in periods:
        start = period.get("start")
        end = period.get("end")
        codes = period.get("subdivision_codes")
        if not isinstance(start, date) or not isinstance(end, date):
            continue
        if not isinstance(codes, tuple):
            continue

        cursor = max(start, valid_from)
        period_end = min(end, valid_to)
        while cursor <= period_end:
            day_index.setdefault(cursor, set()).update(codes)
            cursor += timedelta(days=1)

    return day_index


def _window_ferie_stats(
    window: DateWindow,
    day_index: dict[date, set[str]],
    weights: dict[str, float],
) -> dict[str, object]:
    voiv_days: dict[str, int] = {code: 0 for code in weights}
    days_with_ferie = 0
    cumulative_daily_share = 0.0
    peak_daily_share = 0.0

    for day in _iter_window_days(window):
        active_codes = day_index.get(day, set())
        if active_codes:
            days_with_ferie += 1

        day_share = 0.0
        for code in active_codes:
            if code in voiv_days:
                voiv_days[code] += 1
            day_share += weights.get(code, 0.0)

        cumulative_daily_share += day_share
        if day_share > peak_daily_share:
            peak_daily_share = day_share

    window_days = max(window.days, 1)
    avg_daily_share = cumulative_daily_share / window_days

    top_voivodeships = sorted(
        (
            {
                "code": code,
                "days": voiv_days.get(code, 0),
                "gmv_share": weights.get(code, 0.0),
                "weighted_days": voiv_days.get(code, 0) * weights.get(code, 0.0),
            }
            for code in weights
            if voiv_days.get(code, 0) > 0
        ),
        key=lambda row: row["weighted_days"],
        reverse=True,
    )

    return {
        "days_with_ferie": days_with_ferie,
        "avg_daily_gmv_share": avg_daily_share,
        "peak_daily_gmv_share": peak_daily_share,
        "voivodeship_days": voiv_days,
        "top_voivodeships": top_voivodeships[:8],
    }


def build_ferie_context(
    windows: dict[str, DateWindow],
    country_code: str = "PL",
    language_code: str = DEFAULT_OPEN_HOLIDAYS_LANGUAGE_CODE,
    api_base_url: str = DEFAULT_OPEN_HOLIDAYS_API_BASE_URL,
) -> dict[str, object]:
    if not windows:
        return {
            "source": "OpenHolidays API",
            "source_url": api_base_url,
            "profiles_ranked": [],
            "window_stats": {},
            "missing_years": [],
            "yoy_comparison": {"avg_daily_delta_pp": 0.0, "rows": []},
            "errors": ["No windows provided."],
        }

    valid_from = min(window.start for window in windows.values())
    valid_to = max(window.end for window in windows.values())

    errors: list[str] = []
    subdivisions: dict[str, str] = {}
    periods: list[dict[str, object]] = []

    try:
        subdivisions = _load_subdivisions(
            country_code=country_code,
            language_code=language_code,
            api_base_url=api_base_url,
        )
    except Exception as exc:
        errors.append(f"Subdivisions fetch failed: {exc}")

    try:
        periods = _load_school_ferie_periods(
            valid_from=valid_from,
            valid_to=valid_to,
            country_code=country_code,
            language_code=language_code,
            api_base_url=api_base_url,
        )
    except Exception as exc:
        errors.append(f"School holidays fetch failed: {exc}")

    weights, profiles_ranked = _build_gmv_weights(subdivisions)
    day_index = _build_day_index(periods, valid_from=valid_from, valid_to=valid_to)

    window_stats: dict[str, dict[str, object]] = {}
    for window_key, window in windows.items():
        window_stats[window_key] = _window_ferie_stats(
            window=window,
            day_index=day_index,
            weights=weights,
        )

    yoy_rows: list[dict[str, object]] = []
    yoy_avg_delta_pp = 0.0
    if "current_28d" in windows and "yoy_52w" in windows:
        current_days = max(windows["current_28d"].days, 1)
        current_stats = window_stats.get("current_28d", {})
        yoy_stats = window_stats.get("yoy_52w", {})
        current_voiv_days = current_stats.get("voivodeship_days", {})
        yoy_voiv_days = yoy_stats.get("voivodeship_days", {})

        if isinstance(current_voiv_days, dict) and isinstance(yoy_voiv_days, dict):
            yoy_avg_delta_pp = (
                float(current_stats.get("avg_daily_gmv_share", 0.0))
                - float(yoy_stats.get("avg_daily_gmv_share", 0.0))
            ) * 100.0

            for code, weight in weights.items():
                current_voiv = int(current_voiv_days.get(code, 0))
                yoy_voiv = int(yoy_voiv_days.get(code, 0))
                delta_days = current_voiv - yoy_voiv
                if delta_days == 0:
                    continue

                contribution_pp = (delta_days / current_days) * weight * 100.0
                yoy_rows.append(
                    {
                        "code": code,
                        "name": subdivisions.get(code, code),
                        "current_days": current_voiv,
                        "yoy_days": yoy_voiv,
                        "delta_days": delta_days,
                        "gmv_share": weight,
                        "contribution_pp": contribution_pp,
                    }
                )

            yoy_rows.sort(key=lambda row: abs(float(row["contribution_pp"])), reverse=True)

    return {
        "source": "OpenHolidays API (SchoolHolidays + Subdivisions)",
        "source_url": api_base_url,
        "valid_from": valid_from.isoformat(),
        "valid_to": valid_to.isoformat(),
        "profiles_ranked": profiles_ranked,
        "window_stats": window_stats,
        "missing_years": [],
        "yoy_comparison": {
            "avg_daily_delta_pp": yoy_avg_delta_pp,
            "rows": yoy_rows[:10],
        },
        "errors": errors,
    }


def build_upcoming_ferie_trends(
    run_date: date,
    country_code: str = "PL",
    language_code: str = DEFAULT_OPEN_HOLIDAYS_LANGUAGE_CODE,
    api_base_url: str = DEFAULT_OPEN_HOLIDAYS_API_BASE_URL,
    horizon_days: int = 60,
) -> list[tuple[date, str, str, str, str]]:
    valid_from = run_date + timedelta(days=1)
    valid_to = run_date + timedelta(days=horizon_days)

    try:
        subdivisions = _load_subdivisions(
            country_code=country_code,
            language_code=language_code,
            api_base_url=api_base_url,
        )
        periods = _load_school_ferie_periods(
            valid_from=valid_from,
            valid_to=valid_to,
            country_code=country_code,
            language_code=language_code,
            api_base_url=api_base_url,
        )
    except Exception:
        return []

    weights, _ = _build_gmv_weights(subdivisions)
    rows: list[tuple[date, str, str, str, str]] = []

    for period in periods:
        start = period.get("start")
        end = period.get("end")
        codes = period.get("subdivision_codes")
        if not isinstance(start, date) or not isinstance(end, date) or not isinstance(codes, tuple):
            continue

        visible_start = max(start, valid_from)
        visible_end = min(end, valid_to)
        if visible_end < visible_start:
            continue

        voivodeship_names = ", ".join(subdivisions.get(code, code) for code in codes) or "regiony PL"
        period_share = sum(weights.get(code, 0.0) for code in codes)

        title = f"Winter break: {voivodeship_names}"
        impact = f"Approx. {period_share * 100:.1f}% of GMV proxy is in regions currently on winter break."
        action = (
            "Account for regional seasonality: monitor winter-intent queries and promote travel/family categories."
        )

        rows.append(
            (
                visible_start,
                f"{visible_start.isoformat()} - {visible_end.isoformat()}",
                title,
                impact,
                action,
            )
        )

    rows.sort(key=lambda item: item[0])
    deduped: list[tuple[date, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row[1], row[2])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:6]
