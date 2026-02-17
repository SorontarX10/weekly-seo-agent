from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

from weekly_seo_agent.models import MetricRow


BASE_BRAND_REGEX = (
    r"алегро|allego|allegro|alegro|allegro\.pl|allegro pl|allegor|alergo|allegr|alllegro|allwgro|аллегро|"
    r"alkegro|allegeo|allgro|allegto|allero|allehro|allgero|alleg|allefro|sllegro|llegro|allrgro|allergo|"
    r"aleegro|^all$|ałlegro|all3gro|фддупкщ|фдупкщ|алегра|алешро|олегро|алєгро|a;;egro|akegro|akkegro|"
    r"aklegro|al;legro|aledrogo|alefro|ałegro|alehro|alelgro|alergro|algero|alklegro|alkwgro|allaegro|allagro|"
    r"alldgro|allgr|allgreo|allkegro|alloegro|allregro|allsgro|allwegro|allwgeo|allwhro|alrgro|alwgro|aplegro|"
    r"aregro|alegr|alle|aleg|allgeo|allle|алерго|allegrp|alegrro|alegeo|aлего|alrgo|alergp|alehgro|allgegro|"
    r"aqlegro|aoegro|aslegro|aůůegro|algegro|alegfro|alekgro|alllgro|allrego|llwgro|alerggo|alergeo|aiiegro|"
    r"akwgro|aleegr|alefgro|alegei|aleggo|alegrao|allagero|alllego|alregro|alégro|slegto|akergo|alelgo|algergo|"
    r"alsgro|a;egro|aiegro|ajjegro|akegr|aldgro|aleegeo|alegllo|alegti|alerfo|alggro|alkegeo|alkego|alkegto|"
    r"alkergo|all gro|allaegor|allfgro|allgeor|allgwro|alllegor|alllero|alllgero|allogro|allwgo|allwgri|allwgrp|"
    r"alnegro|alwgeo|aolegro|slgro|sllegto|álegro|алекро|alegr9|aletgo|elrgo|алегпо|arelgo|alwrgo|alerrgo|"
    r"aloegro|aelgro|alegrl|alwegro|alegero|алнгро|alegoro|alekro|alegreo|alegfo|aleegor|laegro|alerho|alewgro"
)

# Country-specific brand typo simulation for SK/CZ/HU (and PL consistency).
# This expands common typo cores into:
# - domain-like variant:   typo.tld
# - spaced variant:        typo tld
# - concatenated variant:  typotld
COUNTRY_SUFFIXES = ("pl", "cz", "sk", "hu")
BRAND_TYPO_CORES = (
    "allegro",
    "alegro",
    "allego",
    "allegor",
    "alllegro",
    "allegr",
    "allgro",
    "allegto",
    "allero",
    "allergo",
    "aleegro",
    "alergro",
    "algero",
)


def _country_brand_regex_map() -> dict[str, re.Pattern[str]]:
    out: dict[str, re.Pattern[str]] = {}
    for code in COUNTRY_SUFFIXES:
        suffix_parts: list[str] = []
        for core in BRAND_TYPO_CORES:
            suffix_parts.append(fr"{core}\.{code}")
            suffix_parts.append(fr"{core}\s+{code}")
            suffix_parts.append(fr"{core}{code}")
        # Keep Cyrillic brand variants with optional country suffix.
        suffix_parts.extend(
            (
                fr"аллегро\.{code}",
                fr"алегро\.{code}",
                fr"аллегро\s+{code}",
                fr"алегро\s+{code}",
            )
        )
        pattern = BASE_BRAND_REGEX + "|" + "|".join(suffix_parts)
        out[code.upper()] = re.compile(pattern, re.IGNORECASE)
    # Fallback for unsupported markets.
    out["DEFAULT"] = re.compile(BASE_BRAND_REGEX, re.IGNORECASE)
    return out


BRAND_REGEX_BY_COUNTRY = _country_brand_regex_map()


def _normalize_text(value: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"\s+", " ", ascii_text).strip()


def _safe_pct(current: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if current > 0 else 0.0
    return (current - baseline) / baseline


def _classify_brand(query: str, country_code: str = "PL") -> str:
    query_raw = (query or "").lower().strip()
    normalized = _normalize_text(query)
    regex = BRAND_REGEX_BY_COUNTRY.get(country_code.upper(), BRAND_REGEX_BY_COUNTRY["DEFAULT"])
    if regex.search(query_raw) or regex.search(normalized):
        return "brand"
    return "non_brand"


def _classify_page_template(page_url: str) -> str:
    parsed = urlparse(page_url)
    path = (parsed.path or "/").lower()
    if path in {"", "/"}:
        return "home"
    if path.startswith("/listing"):
        return "listing"
    if path.startswith("/kategoria/"):
        return "category"
    if path.startswith("/oferta/"):
        return "offer"
    if path.startswith("/pomoc"):
        return "help"
    if path.startswith("/logowanie"):
        return "login"
    if path.startswith("/dzial/"):
        return "section"
    if path.startswith("/uzytkownik/"):
        return "profile"
    return "other"


def _aggregate_rows(rows: list[MetricRow], classifier) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        segment = classifier(row.key)
        target = grouped.setdefault(
            segment,
            {"clicks": 0.0, "impressions": 0.0, "weighted_position": 0.0},
        )
        target["clicks"] += row.clicks
        target["impressions"] += row.impressions
        target["weighted_position"] += row.position * row.impressions

    for segment, row in grouped.items():
        impressions = row["impressions"]
        row["ctr"] = (row["clicks"] / impressions) if impressions else 0.0
        row["position"] = (row["weighted_position"] / impressions) if impressions else 0.0
        del row["weighted_position"]
    return grouped


def _segment_rows(
    current: dict[str, dict[str, float]],
    previous: dict[str, dict[str, float]],
    yoy: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    segments = set(current) | set(previous) | set(yoy)
    for segment in sorted(segments):
        cur = current.get(segment, {"clicks": 0.0, "impressions": 0.0, "ctr": 0.0, "position": 0.0})
        prev = previous.get(segment, {"clicks": 0.0, "impressions": 0.0, "ctr": 0.0, "position": 0.0})
        yoy_row = yoy.get(segment, {"clicks": 0.0, "impressions": 0.0, "ctr": 0.0, "position": 0.0})
        out.append(
            {
                "segment": segment,
                "current_clicks": float(cur["clicks"]),
                "previous_clicks": float(prev["clicks"]),
                "yoy_clicks": float(yoy_row["clicks"]),
                "delta_vs_previous": float(cur["clicks"] - prev["clicks"]),
                "delta_pct_vs_previous": _safe_pct(float(cur["clicks"]), float(prev["clicks"])),
                "delta_vs_yoy": float(cur["clicks"] - yoy_row["clicks"]),
                "delta_pct_vs_yoy": _safe_pct(float(cur["clicks"]), float(yoy_row["clicks"])),
                "current_impressions": float(cur["impressions"]),
                "previous_impressions": float(prev["impressions"]),
                "yoy_impressions": float(yoy_row["impressions"]),
                "impressions_delta_vs_previous": float(cur["impressions"] - prev["impressions"]),
                "impressions_delta_pct_vs_previous": _safe_pct(
                    float(cur["impressions"]), float(prev["impressions"])
                ),
                "impressions_delta_vs_yoy": float(cur["impressions"] - yoy_row["impressions"]),
                "impressions_delta_pct_vs_yoy": _safe_pct(
                    float(cur["impressions"]), float(yoy_row["impressions"])
                ),
                "ctr_current": float(cur["ctr"]),
                "position_current": float(cur["position"]),
            }
        )
    out.sort(key=lambda row: float(row.get("current_clicks", 0.0)), reverse=True)
    return out


def build_segment_diagnostics(
    query_current: list[MetricRow],
    query_previous: list[MetricRow],
    query_yoy: list[MetricRow],
    page_current: list[MetricRow],
    page_previous: list[MetricRow],
    page_yoy: list[MetricRow],
    device_current: list[MetricRow],
    device_previous: list[MetricRow],
    device_yoy: list[MetricRow],
    country_code: str = "PL",
) -> dict[str, list[dict[str, float | str]]]:
    brand_current = _aggregate_rows(
        query_current, lambda key: _classify_brand(key, country_code=country_code)
    )
    brand_previous = _aggregate_rows(
        query_previous, lambda key: _classify_brand(key, country_code=country_code)
    )
    brand_yoy = _aggregate_rows(
        query_yoy, lambda key: _classify_brand(key, country_code=country_code)
    )

    template_current = _aggregate_rows(page_current, _classify_page_template)
    template_previous = _aggregate_rows(page_previous, _classify_page_template)
    template_yoy = _aggregate_rows(page_yoy, _classify_page_template)

    device_current_grouped = _aggregate_rows(device_current, lambda key: key.lower())
    device_previous_grouped = _aggregate_rows(device_previous, lambda key: key.lower())
    device_yoy_grouped = _aggregate_rows(device_yoy, lambda key: key.lower())

    return {
        "brand_non_brand": _segment_rows(brand_current, brand_previous, brand_yoy),
        "page_template": _segment_rows(template_current, template_previous, template_yoy),
        "device": _segment_rows(
            device_current_grouped,
            device_previous_grouped,
            device_yoy_grouped,
        ),
    }
