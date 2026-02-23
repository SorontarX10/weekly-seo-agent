from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

from docx import Document
from docx.shared import Pt, RGBColor

from weekly_seo_agent.weekly_reporting_agent.models import (
    AnalysisResult,
    DateWindow,
    ExternalSignal,
    Finding,
    KeyDelta,
    MetricSummary,
)


BOLD_MARKDOWN_RE = re.compile(r"\*\*(.+?)\*\*")
SIGNED_VALUE_RE = re.compile(
    r"(?<!\w)([+-](?:\d[\d\s.,]*)(?:k|K|m|M|b|B)?%?(?:\s*pp)?)"
)

DARK_GREEN = RGBColor(0x1B, 0x5E, 0x20)
DARK_RED = RGBColor(0x8B, 0x00, 0x00)

CAMPAIGN_EVENT_TOKENS = (
    "black week",
    "black friday",
    "cyber monday",
    "smart week",
    "allegro days",
    "megaraty",
    "prime day",
    "sale",
    "promocj",
    "wyprzedaz",
    "deal",
    "rabat",
    "kupon",
)

CAMPAIGN_QUERY_TOKENS = (
    "black week",
    "black friday",
    "cyber monday",
    "smart week",
    "allegro days",
    "megaraty",
    "allegro smart",
    "promocj",
    "wyprzedaz",
    "rabat",
    "kupon",
    "prime day",
    "sale",
)

COMPETITOR_NAME_TOKENS = (
    ("temu", "Temu"),
    ("amazon", "Amazon"),
    ("aliexpress", "AliExpress"),
    ("shein", "SHEIN"),
    ("ebay", "eBay"),
    ("ceneo", "Ceneo"),
    ("olx", "OLX"),
    ("empik", "Empik"),
    ("media expert", "Media Expert"),
    ("rtv euro agd", "RTV Euro AGD"),
    ("x-kom", "x-kom"),
    ("morele", "Morele"),
)

BRAND_QUERY_TOKENS = (
    "allegro",
    "аллегро",
    "алегро",
)

NOISY_QUERY_EXACT_TOKENS = {
    "all",
    "alegro",
    "allegor",
    "allegroo",
    "allegto",
    "algro",
    "algero",
    "ałegro",
}

QUERY_CLUSTER_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Brand & navigation",
        (
            "allegro",
            "allegro.pl",
            "allegro pl",
            "allegro moje",
            "logowanie",
            "strona glowna",
            "kontakt",
            "sledzenie paczki",
        ),
    ),
    (
        "WOŚP & charity events",
        ("wosp", "wieniawa", "nocowanka", "licytac", "musial"),
    ),
    (
        "Winter & weather demand",
        (
            "sanki",
            "fajerwerk",
            "petard",
            "odsniez",
            "snieg",
            "raczki",
            "sol drog",
            "swider do lod",
            "lopata do sniegu",
            "pellet",
        ),
    ),
    (
        "Valentine's demand",
        ("walentyn", "prezent na walentynki", "dla niej", "dla niego"),
    ),
    (
        "Automotive intent",
        ("vin", "motoryz", "samochod", "golf", "opon", "akumulator"),
    ),
    (
        "Home & garden spring",
        ("glebogryz", "wertykulator", "traw", "nasion", "sekator", "kosiark", "ogrod"),
    ),
)

EXECUTIVE_THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Brand & navigation",
        (
            "allegro",
            "allegro.pl",
            "allegro pl",
            "allegro moje",
            "logowanie",
            "strona glowna",
            "kontakt",
            "sledzenie paczki",
        ),
    ),
    (
        "Heating & fireplace demand",
        ("pellet", "komink", "kominek", "drewno opal", "brykiet", "rozpal"),
    ),
    (
        "Winter sports & anti-slip",
        ("sanki", "raczki", "lyzwy", "narty", "snowboard", "lod", "snieg"),
    ),
    (
        "Gardening & spring prep",
        ("glebogryz", "wertykulator", "sekator", "nasion", "traw", "ogrod", "kosiark"),
    ),
    (
        "Spring fashion",
        ("wiosenn", "kurtka", "sukien", "bluz", "trencz", "mokasyn"),
    ),
    (
        "Summer tires & automotive season",
        ("opony letnie", "opona letnia", "opony", "felgi", "letnie"),
    ),
    (
        "WOŚP & charity events",
        ("wosp", "wieniawa", "nocowanka", "licytac", "musial"),
    ),
    (
        "Valentine's gifts",
        ("walentyn", "prezent dla niej", "prezent dla niego"),
    ),
)

TREND_EVENT_DRIVER_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Valentine's Day", ("walentyn", "valentin", "prezent dla niej", "prezent dla niego")),
    ("Easter", ("wielkanoc", "easter", "pisank", "zajac", "swieconk")),
    ("Spring gardening", ("glebogryz", "wertykulator", "traw", "nasion", "sekator", "kosiark", "lampy solar")),
    ("Winter seasonality", ("sanki", "fajerwerk", "petard", "odsniez", "snieg", "lod", "raczk", "sol drog", "lopata", "pellet")),
    ("Auto demand", ("vin", "opon", "golf", "samochod", "motoryz")),
    ("WOŚP / charity events", ("wosp", "licytac", "wieniawa", "nocowanka")),
    ("Campaign periods", CAMPAIGN_EVENT_TOKENS),
)

EXTERNAL_SIGNAL_SOURCE_QUALITY: tuple[tuple[str, str], ...] = (
    ("Google Search Status", "high"),
    ("Google Search Central Blog", "high"),
    ("Public Holidays", "high"),
    ("NBP FX", "high"),
    ("IMGW warnings", "high"),
    ("SEO Update Analysis", "medium"),
    ("Campaign tracker", "medium"),
    ("News RSS", "medium"),
    ("News HTML", "low"),
    ("SEO Team Presentations", "medium"),
    ("Historical SEO Reports", "medium"),
    ("SEO Status Log", "medium"),
    ("Product Trends", "high"),
)

NEWS_GMV_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Home & heating", ("pellet", "kominek", "piec", "ogrzew", "opa", "brykiet")),
    ("Winter & weather demand", ("sanki", "śnieg", "snieg", "fajerwerk", "petard", "odśnie", "odsnie", "lód", "lod", "raczk")),
    ("Automotive", ("opon", "vin", "samochod", "motoryz", "części", "czesci")),
    ("Electronics", ("iphone", "samsung", "realme", "oppo", "playstation", "ps5")),
    ("Campaigns & promotions", ("allegro days", "smart week", "black friday", "cyber monday", "sale", "promocj", "rabat", "kupon")),
    ("Marketplace competition", ("temu", "amazon", "shein", "ceneo", "olx", "aliexpress")),
    ("Regulatory & tax", ("vat", "ksef", "e-faktur", "podatek", "regulation", "compliance")),
    ("Logistics & payments", ("kurier", "dostaw", "warehouse", "logistics", "inpost", "dpd", "dhl", "payu", "przelewy24", "blik", "visa", "mastercard", "awaria")),
)

# GA4 is intentionally excluded from decision logic until parity checks with GSC are stable.
USE_GA4_IN_REPORT = False
# Keep report short for decision-makers: no appendix in final output.
INCLUDE_APPENDIX_IN_REPORT = False

CONTINUITY_THEME_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Seasonality", ("season", "winter", "ferie", "weather", "temperature")),
    ("Events/Campaigns", ("campaign", "event", "wosp", "walent", "allegro days", "smart week")),
    ("Algorithm", ("algorithm", "update", "discover", "google")),
    ("Technical/UX", ("cwv", "pagespeed", "ux", "device", "template", "page name")),
    ("Non-brand trends", ("non-brand", "trend", "product", "query cluster")),
)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _signed_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _fmt_int(value: float | int) -> str:
    try:
        rounded = int(round(float(value)))
    except (TypeError, ValueError):
        return "0"
    return f"{rounded:,}".replace(",", " ")


def _fmt_signed_int(value: float | int) -> str:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    sign = "+" if raw >= 0 else "-"
    return f"{sign}{_fmt_int(abs(raw))}"


def _fmt_compact(value: float | int) -> str:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    abs_raw = abs(raw)
    suffix = ""
    scale = 1.0
    if abs_raw >= 1_000_000_000:
        suffix = "B"
        scale = 1_000_000_000.0
    elif abs_raw >= 1_000_000:
        suffix = "M"
        scale = 1_000_000.0
    elif abs_raw >= 1_000:
        suffix = "k"
        scale = 1_000.0
    if suffix:
        value_scaled = raw / scale
        text = f"{value_scaled:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}{suffix}"
    return _fmt_int(raw)


def _fmt_signed_compact(value: float | int) -> str:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    sign = "+" if raw >= 0 else "-"
    return f"{sign}{_fmt_compact(abs(raw))}"


def _ga4_metric_available(ga4_context: dict[str, object], metric: str) -> bool:
    availability = ga4_context.get("metric_availability", {})
    if not isinstance(availability, dict):
        return True
    return bool(availability.get(metric, True))


def _ga4_num(summary: dict[str, object], metric: str) -> float | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_delta(current: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if current > 0 else 0.0
    return (current - baseline) / baseline


def _flatten_findings(scope_results: list[tuple[str, AnalysisResult]]) -> list[tuple[str, Finding]]:
    result: list[tuple[str, Finding]] = []
    for scope_name, analysis in scope_results:
        for finding in analysis.findings:
            result.append((scope_name, finding))
    return result


def _normalize_text(value: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"\s+", " ", ascii_text)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def _is_noise_or_brand_query(query: str) -> bool:
    normalized = _normalize_text(query).strip()
    if not normalized:
        return True
    if any(token in normalized for token in BRAND_QUERY_TOKENS):
        return True
    if normalized in NOISY_QUERY_EXACT_TOKENS:
        return True
    compact = normalized.replace(" ", "")
    if compact in NOISY_QUERY_EXACT_TOKENS:
        return True
    if re.fullmatch(r"[a-z]{1,3}", normalized):
        return True
    return False


def _find_scope(scope_results: list[tuple[str, AnalysisResult]], scope_name: str) -> AnalysisResult | None:
    for name, analysis in scope_results:
        if name == scope_name:
            return analysis
    return None


def _campaign_query_rows(query_scope: AnalysisResult | None) -> list[KeyDelta]:
    if query_scope is None:
        return []
    rows = query_scope.top_winners + query_scope.top_losers
    out: list[KeyDelta] = []
    for row in rows:
        text = _normalize_text(row.key)
        if any(token in text for token in CAMPAIGN_QUERY_TOKENS):
            out.append(row)
            continue
        if any(token in text for token, _ in COMPETITOR_NAME_TOKENS) and any(
            token in text for token in ("promocj", "sale", "black", "cyber", "week")
        ):
            out.append(row)
    out.sort(key=lambda item: abs(item.click_delta_vs_previous), reverse=True)
    return out[:8]


def _campaign_event_context(
    external_signals: list[ExternalSignal],
    query_scope: AnalysisResult | None,
) -> dict[str, object]:
    allegro_events: list[ExternalSignal] = []
    competitor_events: list[tuple[ExternalSignal, str]] = []
    seen_allegro: set[str] = set()
    seen_comp: set[str] = set()

    for signal in external_signals:
        text = _normalize_text(f"{signal.title} {signal.details} {signal.source}")
        source_text = _normalize_text(signal.source)
        is_campaign_tracker_source = "campaign tracker" in source_text
        has_campaign_token = any(token in text for token in CAMPAIGN_EVENT_TOKENS)
        has_competitor_token = any(token in text for token, _ in COMPETITOR_NAME_TOKENS)
        has_promo_context = any(
            token in text
            for token in ("promocj", "promo", "sale", "deal", "discount", "black", "cyber", "week")
        )
        if not (has_campaign_token or is_campaign_tracker_source or (has_competitor_token and has_promo_context)):
            continue

        if "allegro" in text or "smart week" in text or "allegro days" in text or "megaraty" in text:
            key = re.sub(
                r"\s*[-|:]\s*[^-|:]{1,80}$",
                "",
                _normalize_text(signal.title),
            ).strip() or _normalize_text(signal.title)
            if key not in seen_allegro:
                seen_allegro.add(key)
                allegro_events.append(signal)

        for token, label in COMPETITOR_NAME_TOKENS:
            if token in text:
                key = (
                    re.sub(
                        r"\s*[-|:]\s*[^-|:]{1,80}$",
                        "",
                        _normalize_text(signal.title),
                    ).strip()
                    + "::"
                    + label.lower()
                )
                if key not in seen_comp:
                    seen_comp.add(key)
                    competitor_events.append((signal, label))
                break

    query_events = _campaign_query_rows(query_scope)
    return {
        "allegro_events": sorted(allegro_events, key=lambda item: item.day, reverse=True)[:10],
        "competitor_events": sorted(
            competitor_events, key=lambda item: item[0].day, reverse=True
        )[:10],
        "query_events": query_events,
    }


def _cluster_label_for_query(query: str) -> str:
    normalized = _normalize_text(query)
    for label, tokens in QUERY_CLUSTER_RULES:
        if any(token in normalized for token in tokens):
            return label
    return "Other non-brand demand"


def _executive_theme_for_query(query: str) -> str:
    normalized = _normalize_text(query)
    for label, tokens in EXECUTIVE_THEME_RULES:
        if any(token in normalized for token in tokens):
            return label
    return "Other demand themes"


def _executive_yoy_theme_summary(
    query_scope: AnalysisResult | None,
    limit: int = 3,
) -> tuple[str, str]:
    if query_scope is None:
        return "", ""

    by_query: dict[str, KeyDelta] = {}
    for row in query_scope.top_winners + query_scope.top_losers:
        key = _normalize_text(row.key)
        if not key:
            continue
        if _is_noise_or_brand_query(row.key):
            continue
        current = by_query.get(key)
        if current is None or abs(row.click_delta_vs_yoy) > abs(current.click_delta_vs_yoy):
            by_query[key] = row

    buckets: dict[str, dict[str, object]] = {}
    for row in by_query.values():
        normalized_key = _normalize_text(row.key)
        if any(token in normalized_key for token in BRAND_QUERY_TOKENS):
            continue
        if abs(row.click_delta_vs_yoy) < 100:
            continue
        label = _executive_theme_for_query(row.key)
        payload = buckets.setdefault(
            label,
            {"delta": 0.0, "examples": []},
        )
        payload["delta"] = float(payload.get("delta", 0.0)) + float(row.click_delta_vs_yoy)
        examples = payload.get("examples")
        if isinstance(examples, list) and len(examples) < 3:
            examples.append(str(row.key).strip())

    if not buckets:
        return "", ""

    growth = sorted(
        [
            (label, float(payload.get("delta", 0.0)), payload)
            for label, payload in buckets.items()
            if float(payload.get("delta", 0.0)) > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    declines = sorted(
        [
            (label, float(payload.get("delta", 0.0)), payload)
            for label, payload in buckets.items()
            if float(payload.get("delta", 0.0)) < 0
        ],
        key=lambda item: item[1],
    )[:limit]

    if len(growth) > 1:
        growth = [row for row in growth if row[0] != "Other demand themes"][:limit]
    if len(declines) > 1:
        declines = [row for row in declines if row[0] != "Other demand themes"][:limit]

    def _render(label: str, delta: float, payload: dict[str, object]) -> str:
        rendered_label = label
        if label == "Other demand themes":
            examples = payload.get("examples")
            if isinstance(examples, list) and examples:
                rendered_examples = ", ".join(
                    f"`{str(item).strip()}`"
                    for item in examples[:2]
                    if str(item).strip()
                )
                if rendered_examples:
                    rendered_label = f"Other demand themes (e.g. {rendered_examples})"
        return f"{rendered_label} ({_fmt_signed_int(delta)} YoY)"

    growth_line = ""
    if growth:
        growth_line = "; ".join(
            _render(label, delta, payload)
            for label, delta, payload in growth
        )
    decline_line = ""
    if declines:
        decline_line = "; ".join(
            _render(label, delta, payload)
            for label, delta, payload in declines
        )
    return growth_line, decline_line


def _query_cluster_rows(query_scope: AnalysisResult | None) -> list[dict[str, object]]:
    if query_scope is None:
        return []

    by_query: dict[str, KeyDelta] = {}
    for row in query_scope.top_winners + query_scope.top_losers:
        key = _normalize_text(row.key)
        if not key:
            continue
        if _is_noise_or_brand_query(row.key):
            continue
        previous = by_query.get(key)
        if previous is None or row.current_clicks > previous.current_clicks:
            by_query[key] = row

    buckets: dict[str, dict[str, object]] = {}
    for row in by_query.values():
        label = _cluster_label_for_query(row.key)
        payload = buckets.setdefault(
            label,
            {
                "cluster": label,
                "rows": 0,
                "current_clicks": 0.0,
                "delta_vs_previous": 0.0,
                "delta_vs_yoy": 0.0,
                "queries": [],
            },
        )
        payload["rows"] = int(payload.get("rows", 0)) + 1
        payload["current_clicks"] = float(payload.get("current_clicks", 0.0)) + float(
            row.current_clicks
        )
        payload["delta_vs_previous"] = float(payload.get("delta_vs_previous", 0.0)) + float(
            row.click_delta_vs_previous
        )
        payload["delta_vs_yoy"] = float(payload.get("delta_vs_yoy", 0.0)) + float(
            row.click_delta_vs_yoy
        )
        payload_queries = payload.get("queries")
        if isinstance(payload_queries, list):
            payload_queries.append((float(row.current_clicks), row.key))

    out = list(buckets.values())
    for row in out:
        samples = row.get("queries")
        if isinstance(samples, list):
            ordered = sorted(
                [item for item in samples if isinstance(item, tuple) and len(item) == 2],
                key=lambda item: float(item[0]),
                reverse=True,
            )
            row["samples"] = [str(item[1]) for item in ordered[:3]]
        else:
            row["samples"] = []
        row.pop("queries", None)

    out.sort(
        key=lambda row: (
            float(row.get("current_clicks", 0.0)),
            abs(float(row.get("delta_vs_previous", 0.0))),
        ),
        reverse=True,
    )
    return out[:8]


def _query_cluster_summary_line(query_scope: AnalysisResult | None) -> str:
    rows = _query_cluster_rows(query_scope)
    if not rows:
        return ""
    gains = sorted(
        [row for row in rows if float(row.get("delta_vs_previous", 0.0)) > 0],
        key=lambda row: float(row.get("delta_vs_previous", 0.0)),
        reverse=True,
    )[:3]
    losses = sorted(
        [row for row in rows if float(row.get("delta_vs_previous", 0.0)) < 0],
        key=lambda row: float(row.get("delta_vs_previous", 0.0)),
    )[:3]
    largest = sorted(
        rows,
        key=lambda row: float(row.get("current_clicks", 0.0)),
        reverse=True,
    )[:3]

    parts: list[str] = []
    if gains:
        gains_text = "; ".join(
            f"{row.get('cluster', '')} ({_fmt_signed_int(row.get('delta_vs_previous', 0.0))} vs prev)"
            for row in gains
            if str(row.get("cluster", "")).strip()
        )
        if gains_text:
            parts.append(f"gains: {gains_text}")
    if losses:
        losses_text = "; ".join(
            f"{row.get('cluster', '')} ({_fmt_signed_int(row.get('delta_vs_previous', 0.0))} vs prev)"
            for row in losses
            if str(row.get("cluster", "")).strip()
        )
        if losses_text:
            parts.append(f"losses: {losses_text}")
    if largest:
        largest_text = "; ".join(
            f"{row.get('cluster', '')} ({_fmt_int(row.get('current_clicks', 0.0))} clicks)"
            for row in largest
            if str(row.get("cluster", "")).strip()
        )
        if largest_text:
            parts.append(f"largest demand clusters: {largest_text}")
    if not parts:
        return ""
    return "Query-cluster split by direction: " + " | ".join(parts) + "."


def _query_cluster_contribution_rows(
    query_scope: AnalysisResult | None,
    total_click_delta_yoy: float,
    limit: int = 8,
) -> list[dict[str, object]]:
    rows = _query_cluster_rows(query_scope)
    if not rows:
        return []

    tracked_total = sum(float(row.get("delta_vs_yoy", 0.0)) for row in rows)
    if abs(total_click_delta_yoy) >= 1.0:
        denominator = abs(total_click_delta_yoy)
        basis = "total_yoy"
    elif abs(tracked_total) >= 1.0:
        denominator = abs(tracked_total)
        basis = "tracked_yoy"
    else:
        denominator = 1.0
        basis = "tracked_yoy"

    out: list[dict[str, object]] = []
    for row in rows:
        delta_yoy = float(row.get("delta_vs_yoy", 0.0))
        out.append(
            {
                "cluster": str(row.get("cluster", "")).strip(),
                "current_clicks": float(row.get("current_clicks", 0.0)),
                "delta_vs_yoy": delta_yoy,
                "contribution_pct": (delta_yoy / denominator) * 100.0,
                "basis": basis,
                "rows": int(row.get("rows", 0)),
                "samples": row.get("samples", []),
            }
        )
    out.sort(key=lambda item: abs(float(item.get("delta_vs_yoy", 0.0))), reverse=True)
    return out[: max(1, limit)]


def _normalized_page_key(raw_url: str) -> str:
    raw = str(raw_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = (parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "/").strip()
    if not path.startswith("/"):
        path = "/" + path
    if path != "/":
        path = path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"


def _pick_source_quality(source: str) -> str:
    lowered = _normalize_text(source)
    for token, label in EXTERNAL_SIGNAL_SOURCE_QUALITY:
        if _normalize_text(token) in lowered:
            return label
    return "low"


def _signal_quality_rows(
    external_signals: list[ExternalSignal],
    limit: int = 12,
) -> list[dict[str, object]]:
    deduped: dict[str, dict[str, object]] = {}
    severity_rank = {"info": 1, "medium": 2, "high": 3}
    quality_rank = {"low": 1, "medium": 2, "high": 3}

    for signal in sorted(external_signals, key=lambda row: row.day, reverse=True):
        title_key = _normalize_text(re.sub(r"\s*-\s*[^-]{1,50}$", "", signal.title)).strip()
        if not title_key:
            title_key = _normalize_text(signal.title)
        if not title_key:
            continue

        entry = deduped.get(title_key)
        source_quality = _pick_source_quality(signal.source)
        if entry is None:
            deduped[title_key] = {
                "title": signal.title,
                "latest_day": signal.day,
                "severity": signal.severity.lower(),
                "sources": {signal.source},
                "mentions": 1,
                "quality": source_quality,
            }
            continue

        entry["mentions"] = int(entry.get("mentions", 0)) + 1
        sources = entry.get("sources")
        if isinstance(sources, set):
            sources.add(signal.source)
        if signal.day > entry.get("latest_day"):  # type: ignore[arg-type]
            entry["latest_day"] = signal.day
            entry["title"] = signal.title
        if severity_rank.get(signal.severity.lower(), 1) > severity_rank.get(
            str(entry.get("severity", "info")).lower(), 1
        ):
            entry["severity"] = signal.severity.lower()
        if quality_rank.get(source_quality, 1) > quality_rank.get(
            str(entry.get("quality", "low")).lower(), 1
        ):
            entry["quality"] = source_quality

    rows: list[dict[str, object]] = []
    for payload in deduped.values():
        sources = payload.get("sources")
        source_count = len(sources) if isinstance(sources, set) else 0
        quality = str(payload.get("quality", "low")).lower()
        if source_count >= 3 and quality == "medium":
            quality = "high"
        rows.append(
            {
                "date": payload.get("latest_day"),
                "title": str(payload.get("title", "")).strip(),
                "severity": str(payload.get("severity", "info")).lower(),
                "quality": quality,
                "source_count": source_count,
                "mentions": int(payload.get("mentions", 0)),
            }
        )
    rows.sort(
        key=lambda item: (
            quality_rank.get(str(item.get("quality", "low")), 1),
            severity_rank.get(str(item.get("severity", "info")), 1),
            int(item.get("mentions", 0)),
        ),
        reverse=True,
    )
    return rows[: max(1, limit)]


def _news_category_impact_rows(
    external_signals: list[ExternalSignal],
    limit: int = 8,
) -> list[dict[str, object]]:
    category_map: dict[str, dict[str, object]] = {}
    severity_weight = {"high": 3.0, "medium": 2.0, "info": 1.0}
    for signal in external_signals:
        source_norm = _normalize_text(signal.source)
        if "news" not in source_norm and "campaign tracker" not in source_norm:
            continue
        blob = _normalize_text(f"{signal.title} {signal.details} {signal.source}")
        category = "Other market events"
        for label, tokens in NEWS_GMV_CATEGORY_RULES:
            if any(_normalize_text(token) in blob for token in tokens):
                category = label
                break
        row = category_map.setdefault(
            category,
            {
                "category": category,
                "signals": 0,
                "weighted_impact": 0.0,
                "latest_day": signal.day,
                "examples": [],
            },
        )
        row["signals"] = int(row.get("signals", 0)) + 1
        row["weighted_impact"] = float(row.get("weighted_impact", 0.0)) + severity_weight.get(
            signal.severity.lower(), 1.0
        )
        if signal.day > row.get("latest_day"):  # type: ignore[arg-type]
            row["latest_day"] = signal.day
        examples = row.get("examples", [])
        if isinstance(examples, list) and signal.title and signal.title not in examples:
            examples.append(signal.title)
            row["examples"] = examples[:3]

    rows = list(category_map.values())
    rows.sort(
        key=lambda item: (
            float(item.get("weighted_impact", 0.0)),
            int(item.get("signals", 0)),
        ),
        reverse=True,
    )
    return rows[: max(1, limit)]


def _trend_page_coverage_rows(
    scope_results: list[tuple[str, AnalysisResult]],
    additional_context: dict[str, object] | None,
    limit: int = 8,
) -> list[dict[str, object]]:
    product_trends = (additional_context or {}).get("product_trends", {})
    if not isinstance(product_trends, dict):
        return []

    current_rows = product_trends.get("current_non_brand", [])
    upcoming_rows = product_trends.get("upcoming_31d", [])
    if not isinstance(current_rows, list):
        current_rows = []
    if not isinstance(upcoming_rows, list):
        upcoming_rows = []

    candidates: list[dict[str, object]] = []
    seen_trends: set[str] = set()
    for row in current_rows + upcoming_rows:
        if not isinstance(row, dict):
            continue
        trend = str(row.get("trend", "")).strip()
        if not trend:
            continue
        trend_key = _normalize_text(trend)
        if trend_key in seen_trends:
            continue
        seen_trends.add(trend_key)
        candidates.append(row)
        if len(candidates) >= max(5, limit):
            break

    page_scope = _find_scope(scope_results, "page")
    page_index: dict[str, KeyDelta] = {}
    if page_scope is not None:
        for row in page_scope.top_winners + page_scope.top_losers:
            key = _normalized_page_key(row.key)
            if not key:
                continue
            existing = page_index.get(key)
            if existing is None or row.current_clicks > existing.current_clicks:
                page_index[key] = row

    out: list[dict[str, object]] = []
    for row in candidates:
        trend = str(row.get("trend", "")).strip()
        page = str(row.get("page", "")).strip()
        page_key = _normalized_page_key(page)
        page_perf = page_index.get(page_key) if page_key else None
        median_pos = _safe_float(row, "median_position")

        status = "Coverage present"
        gap_reason = "Landing page exists and has measurable demand."
        if not page:
            status = "Coverage gap"
            gap_reason = "Trend row has no mapped landing page URL."
        elif page_perf is None:
            status = "Coverage gap"
            gap_reason = "Landing page not visible among tracked top page movers."
        else:
            if page_perf.click_delta_vs_previous < -1000:
                status = "At-risk coverage"
                gap_reason = "Mapped landing page is losing clicks WoW."
            elif page_perf.current_position > 6.0:
                status = "Optimization gap"
                gap_reason = "Mapped landing page has ranking headroom (position > 6)."
            elif page_perf.current_ctr < 0.02 and page_perf.current_clicks > 500:
                status = "Optimization gap"
                gap_reason = "Mapped landing page has low CTR for current demand."

        out.append(
            {
                "trend": trend,
                "page": page or "-",
                "sheet_value": _safe_float(row, "value"),
                "sheet_date": str(row.get("date", "")).strip(),
                "median_position": median_pos,
                "status": status,
                "gap_reason": gap_reason,
                "page_delta_vs_prev": float(page_perf.click_delta_vs_previous) if page_perf else 0.0,
                "page_delta_vs_yoy": float(page_perf.click_delta_vs_yoy) if page_perf else 0.0,
                "page_ctr": float(page_perf.current_ctr) if page_perf else 0.0,
                "page_position": float(page_perf.current_position) if page_perf else 0.0,
            }
        )
    return out[: max(1, limit)]


def _hypothesis_continuity_rows(
    hypotheses: list[dict[str, object]],
    additional_context: dict[str, object] | None,
    limit: int = 6,
) -> list[dict[str, str]]:
    historical = (additional_context or {}).get("historical_reports", {})
    if not isinstance(historical, dict) or not historical.get("enabled"):
        return []

    notes: list[str] = []
    recent_reports = historical.get("recent_reports", [])
    if isinstance(recent_reports, list):
        notes.extend(_collect_context_notes(recent_reports))
    yoy_report = historical.get("yoy_report", {})
    if isinstance(yoy_report, dict):
        notes.extend(_collect_context_notes([yoy_report], fallback_key="excerpt"))

    notes_blob = _normalize_text(" ".join(notes))
    out: list[dict[str, str]] = []
    for row in hypotheses[: max(1, limit)]:
        category = str(row.get("category", "")).strip() or "Unclassified"
        thesis = str(row.get("thesis", "")).strip()
        category_key = _normalize_text(category)
        matched_tokens: tuple[str, ...] = ()
        for label, tokens in CONTINUITY_THEME_TOKENS:
            if _normalize_text(label) in category_key:
                matched_tokens = tokens
                break
        if not matched_tokens:
            matched_tokens = tuple(
                token for token in re.findall(r"[a-z]{4,}", _normalize_text(thesis))[:4]
            )

        if not notes:
            status = "No prior-report context"
            evidence = "No earlier weekly reports with extractable highlights."
        elif matched_tokens and any(token in notes_blob for token in matched_tokens):
            status = "Recurring vs prior report"
            hit = next((token for token in matched_tokens if token in notes_blob), "")
            evidence = f"Theme overlap found in prior reports (token: `{hit}`)."
        else:
            status = "New this week"
            evidence = "No direct overlap with extracted hypotheses/themes from prior reports."
        out.append(
            {
                "category": category,
                "status": status,
                "evidence": evidence,
            }
        )
    return out


def _follow_up_flags(
    totals: dict[str, MetricSummary],
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
    external_signals: list[ExternalSignal],
    weather_summary: dict[str, float],
) -> list[str]:
    flags: list[str] = []
    current = totals["current_28d"]
    yoy = totals["yoy_52w"]

    brand_rows = (segment_diagnostics or {}).get("brand_non_brand") or []
    non_brand_row = next(
        (row for row in brand_rows if str(row.get("segment", "")).strip() == "non_brand"),
        None,
    )
    if isinstance(non_brand_row, dict):
        yoy_drop = float(non_brand_row.get("delta_pct_vs_yoy", 0.0))
        ctr_drop_pp = (current.ctr - yoy.ctr) * 100.0
        if yoy_drop <= -0.15 and ctr_drop_pp <= -0.30:
            flags.append(
                "Non-brand pressure flag: non-brand clicks are down "
                f"{yoy_drop * 100:.2f}% YoY and site CTR is down {ctr_drop_pp:.2f} pp YoY. Further analysis recommended."
            )

    page_rows = (segment_diagnostics or {}).get("page_template") or []
    page_loser = _top_negative_segment(page_rows, min_clicks=3000.0)
    if isinstance(page_loser, dict):
        pct_drop = float(page_loser.get("delta_pct_vs_previous", 0.0)) * 100.0
        if pct_drop <= -20.0:
            flags.append(
                "Page Name concentration flag: "
                f"`{page_loser.get('segment', '')}` is down {pct_drop:.2f}% WoW. Run section-level diagnosis."
            )

    seo_update_rows = [
        row
        for row in external_signals
        if "seo update analysis" in _normalize_text(row.source)
        and "risk" in _normalize_text(row.details)
    ]
    if seo_update_rows:
        latest = max(seo_update_rows, key=lambda row: row.day)
        flags.append(
            "Algorithm-volatility flag: external SEO media indicate elevated risk "
            f"({latest.day.isoformat()}: `{latest.title}`). Track affected query/page clusters."
        )

    forecast_avg = float(weather_summary.get("forecast_avg_temp_c", 0.0))
    current_avg = float(weather_summary.get("avg_temp_current_c", 0.0))
    if abs(forecast_avg - current_avg) >= 3.0:
        flags.append(
            "Weather-shift flag: next-week average temperature differs by "
            f"{forecast_avg - current_avg:+.1f}C vs current week. Monitor season-sensitive demand timing."
        )
    return flags[:5]


def _fallback_non_brand_yoy_from_gsc(
    query_scope: AnalysisResult | None,
    top_rows: int,
) -> list[dict[str, object]]:
    if query_scope is None:
        return []
    rows = query_scope.top_winners + query_scope.top_losers
    deduped: dict[str, dict[str, object]] = {}
    for row in rows:
        normalized_key = _normalize_text(row.key)
        if not normalized_key:
            continue
        if _is_noise_or_brand_query(row.key):
            continue
        if row.current_clicks <= 0 and row.yoy_clicks <= 0:
            continue
        delta = row.current_clicks - row.yoy_clicks
        if abs(delta) < 100:
            continue
        key = normalized_key
        payload = {
            "trend": row.key,
            "current_value": float(row.current_clicks),
            "previous_value": float(row.yoy_clicks),
            "delta_value": float(delta),
            "delta_pct": float(row.click_delta_pct_vs_yoy * 100.0),
            "sheet": "GSC query fallback",
        }
        current = deduped.get(key)
        if current is None or abs(float(payload["delta_value"])) > abs(float(current["delta_value"])):
            deduped[key] = payload

    ranked = sorted(
        deduped.values(),
        key=lambda item: abs(float(item.get("delta_value", 0.0))),
        reverse=True,
    )
    return ranked[: max(1, top_rows)]


def _sum_losses(rows: list[KeyDelta], keywords: tuple[str, ...]) -> float:
    total = 0.0
    for row in rows:
        if row.click_delta_vs_previous < 0 and _contains_any(row.key, keywords):
            total += -row.click_delta_vs_previous
    return total


def _sum_gains(rows: list[KeyDelta], keywords: tuple[str, ...]) -> float:
    total = 0.0
    for row in rows:
        if row.click_delta_vs_previous > 0 and _contains_any(row.key, keywords):
            total += row.click_delta_vs_previous
    return total


def _top_negative_segment(
    rows: list[dict[str, float | str]] | None,
    min_clicks: float = 1000.0,
) -> dict[str, float | str] | None:
    if not rows:
        return None
    candidates = [
        row
        for row in rows
        if float(row.get("current_clicks", 0.0)) >= min_clicks
        and float(row.get("delta_vs_previous", 0.0)) < 0
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row.get("delta_vs_previous", 0.0)))[0]


def _safe_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _source_freshness_rows(additional_context: dict[str, object] | None) -> list[dict[str, object]]:
    payload = (additional_context or {}).get("source_freshness", [])
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, object]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        if not source:
            continue
        rows.append(
            {
                "source": source,
                "status": str(row.get("status", "")).strip().lower() or "unknown",
                "last_day": str(row.get("last_day", "")).strip(),
                "ttl_hours": _safe_float(row, "ttl_hours"),
                "cache_mode": str(row.get("cache_mode", "")).strip() or "-",
                "note": str(row.get("note", "")).strip() or "-",
            }
        )
    return rows


def _impact_attribution_rows(
    *,
    totals: dict[str, MetricSummary],
    scope_results: list[tuple[str, AnalysisResult]],
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
    limit: int = 8,
) -> list[dict[str, object]]:
    total_delta = float(totals["current_28d"].clicks - totals["previous_28d"].clicks)
    denom = abs(total_delta) if abs(total_delta) > 0 else 1.0
    rows: list[dict[str, object]] = []

    query_scope = _find_scope(scope_results, "query")
    for row in _query_cluster_rows(query_scope):
        delta = float(row.get("delta_vs_previous", 0.0))
        if abs(delta) < 500:
            continue
        rows.append(
            {
                "dimension": "Query cluster",
                "name": str(row.get("cluster", "")).strip(),
                "delta": delta,
                "share_of_total": (delta / denom) * 100.0,
            }
        )

    for group_name, key in (
        ("Brand split", "brand_non_brand"),
        ("Device", "device"),
        ("Page Name", "page_template"),
    ):
        for row in (segment_diagnostics or {}).get(key, []) or []:
            if not isinstance(row, dict):
                continue
            delta = float(row.get("delta_vs_previous", 0.0))
            if abs(delta) < 500:
                continue
            rows.append(
                {
                    "dimension": group_name,
                    "name": str(row.get("segment", "")).strip() or "-",
                    "delta": delta,
                    "share_of_total": (delta / denom) * 100.0,
                }
            )

    rows.sort(key=lambda item: abs(float(item.get("delta", 0.0))), reverse=True)
    return rows[: max(1, limit)]


def _query_anomaly_rows(
    query_scope: AnalysisResult | None,
    limit: int = 6,
) -> list[dict[str, object]]:
    if query_scope is None:
        return []
    rows = query_scope.top_winners + query_scope.top_losers
    deltas = [float(row.click_delta_vs_previous) for row in rows if abs(float(row.click_delta_vs_previous)) >= 200]
    if len(deltas) < 5:
        return []
    sorted_abs = sorted(abs(value) for value in deltas)
    mid = len(sorted_abs) // 2
    median_abs = sorted_abs[mid]
    mad = sorted(abs(abs(value) - median_abs) for value in deltas)[mid]
    threshold = median_abs + 2.5 * max(1.0, mad)

    out: list[dict[str, object]] = []
    for row in rows:
        delta = float(row.click_delta_vs_previous)
        if abs(delta) < max(2000.0, threshold):
            continue
        out.append(
            {
                "query": str(row.key).strip(),
                "delta": delta,
                "delta_pct": float(row.click_delta_pct_vs_previous) * 100.0,
                "current_clicks": float(row.current_clicks),
            }
        )
    out.sort(key=lambda item: abs(float(item.get("delta", 0.0))), reverse=True)
    return out[: max(1, limit)]


def _data_quality_score(
    *,
    additional_context: dict[str, object] | None,
    external_signals: list[ExternalSignal],
    senuto_error: str | None,
) -> tuple[int, list[str]]:
    score = 100
    notes: list[str] = []
    ctx = additional_context or {}
    errors = ctx.get("errors", [])
    if isinstance(errors, list):
        penalty = min(25, 4 * len([item for item in errors if str(item).strip()]))
        score -= penalty
        if penalty:
            notes.append(f"{len(errors)} context-source errors detected.")

    freshness_rows = _source_freshness_rows(ctx)
    stale_count = sum(1 for row in freshness_rows if str(row.get("status", "")).lower() == "stale")
    degraded_count = sum(1 for row in freshness_rows if str(row.get("status", "")).lower() == "degraded")
    score -= min(20, stale_count * 5 + degraded_count * 8)
    if stale_count:
        notes.append(f"{stale_count} source(s) served from stale cache.")
    if degraded_count:
        notes.append(f"{degraded_count} source(s) degraded/no live data.")

    if senuto_error:
        score -= 10
        notes.append("Senuto unavailable in this run.")

    if not external_signals:
        score -= 8
        notes.append("No external signals available.")

    score = max(0, min(100, score))
    if not notes:
        notes.append("No material data-quality risks detected.")
    return score, notes[:4]


def _decision_one_pager_rows(hypotheses: list[dict[str, object]], limit: int = 6) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in hypotheses[: max(1, limit)]:
        thesis = str(row.get("thesis", "")).strip()
        category = str(row.get("category", "")).strip() or "Unknown"
        confidence = int(row.get("confidence", 0) or 0)
        evidence_rows = row.get("evidence", [])
        evidence = ""
        if isinstance(evidence_rows, list) and evidence_rows:
            evidence = str(evidence_rows[0]).strip()
        if not evidence:
            evidence = "Evidence available in supporting appendix."
        rows.append(
            {
                "what_changed": category,
                "why_it_matters": f"{thesis} (confidence {confidence}/100). Evidence: {evidence}",
            }
        )
    return rows


def _brand_proxy_from_gsc(
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
) -> dict[str, float] | None:
    rows = (segment_diagnostics or {}).get("brand_non_brand") or []
    if not isinstance(rows, list):
        return None
    brand_row = next(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("segment", "")).strip().lower() == "brand"
        ),
        None,
    )
    if not isinstance(brand_row, dict):
        return None
    return {
        "current_clicks": float(brand_row.get("current_clicks", 0.0)),
        "delta_vs_previous": float(brand_row.get("delta_vs_previous", 0.0)),
        "delta_pct_vs_previous": float(brand_row.get("delta_pct_vs_previous", 0.0)) * 100.0,
        "delta_vs_yoy": float(brand_row.get("delta_vs_yoy", 0.0)),
        "delta_pct_vs_yoy": float(brand_row.get("delta_pct_vs_yoy", 0.0)) * 100.0,
        "current_impressions": float(brand_row.get("current_impressions", 0.0)),
        "impressions_delta_vs_previous": float(brand_row.get("impressions_delta_vs_previous", 0.0)),
        "impressions_delta_pct_vs_previous": float(
            brand_row.get("impressions_delta_pct_vs_previous", 0.0)
        )
        * 100.0,
        "impressions_delta_vs_yoy": float(brand_row.get("impressions_delta_vs_yoy", 0.0)),
        "impressions_delta_pct_vs_yoy": float(
            brand_row.get("impressions_delta_pct_vs_yoy", 0.0)
        )
        * 100.0,
    }


def _brand_ctr_proxy_from_gsc(
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
) -> dict[str, float] | None:
    rows = (segment_diagnostics or {}).get("brand_non_brand") or []
    if not isinstance(rows, list):
        return None
    brand_row = next(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("segment", "")).strip().lower() == "brand"
        ),
        None,
    )
    if not isinstance(brand_row, dict):
        return None

    current_clicks = float(brand_row.get("current_clicks", 0.0))
    previous_clicks = float(brand_row.get("previous_clicks", 0.0))
    yoy_clicks = float(brand_row.get("yoy_clicks", 0.0))
    current_impr = float(brand_row.get("current_impressions", 0.0))
    previous_impr = float(brand_row.get("previous_impressions", 0.0))
    yoy_impr = float(brand_row.get("yoy_impressions", 0.0))

    current_ctr = (current_clicks / current_impr) if current_impr else 0.0
    previous_ctr = (previous_clicks / previous_impr) if previous_impr else 0.0
    yoy_ctr = (yoy_clicks / yoy_impr) if yoy_impr else 0.0
    return {
        "current_ctr": current_ctr,
        "previous_ctr": previous_ctr,
        "yoy_ctr": yoy_ctr,
        "delta_pp_vs_previous": (current_ctr - previous_ctr) * 100.0,
        "delta_pp_vs_yoy": (current_ctr - yoy_ctr) * 100.0,
    }


def _trade_plan_paid_pressure(
    additional_context: dict[str, object] | None,
) -> tuple[str, float]:
    trade_plan = (additional_context or {}).get("trade_plan", {})
    if not isinstance(trade_plan, dict) or not trade_plan.get("enabled"):
        return ("unknown", 0.0)
    rows = trade_plan.get("channel_split", [])
    if not isinstance(rows, list):
        return ("unknown", 0.0)
    paid_tokens = ("paid", "google ads", "ads", "cpc", "sem", "search")
    delta_sum = 0.0
    matched = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        channel = str(row.get("channel", "")).strip().lower()
        if not channel:
            continue
        if any(token in channel for token in paid_tokens):
            delta_sum += float(row.get("delta_spend", 0.0) or 0.0)
            matched += 1
    if matched == 0:
        return ("unknown", 0.0)
    if delta_sum > 0:
        return ("up", delta_sum)
    if delta_sum < 0:
        return ("down", delta_sum)
    return ("flat", 0.0)


def _trade_plan_signal(additional_context: dict[str, object] | None) -> dict[str, object] | None:
    trade_plan = (additional_context or {}).get("trade_plan", {})
    if not isinstance(trade_plan, dict) or not trade_plan.get("enabled"):
        return None
    campaign_rows = trade_plan.get("campaign_rows", [])
    channel_rows = trade_plan.get("channel_split", [])
    campaign_names: list[str] = []
    if isinstance(campaign_rows, list):
        for row in campaign_rows:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("campaign", "")).strip()
            if not raw_name:
                continue
            # Ignore placeholders/noisy labels like "1", "2", "-".
            if re.fullmatch(r"[-\d\s.]+", raw_name):
                continue
            name = raw_name[:64]
            if name.lower() not in {n.lower() for n in campaign_names}:
                campaign_names.append(name)
            if len(campaign_names) >= 2:
                break

    paid_direction = "unknown"
    if isinstance(channel_rows, list) and channel_rows:
        paid_direction, _ = _trade_plan_paid_pressure(additional_context)

    confidence = 56
    if campaign_names:
        confidence += 8
    if paid_direction in {"up", "down"}:
        confidence += 10
    confidence = max(45, min(82, confidence))

    if paid_direction == "up":
        statement = (
            "Trade plan indicates higher paid-search pressure in the active campaign window; "
            "organic demand allocation can shift from SEO to paid placements."
        )
    elif paid_direction == "down":
        statement = (
            "Trade plan indicates lower paid-search pressure in the active campaign window; "
            "organic demand capture can improve for overlapping intents."
        )
    else:
        statement = (
            "Trade plan confirms active campaign windows; campaign timing should be treated as "
            "a demand-allocation context in SEO interpretation."
        )

    summary = "Trade plan context is available for this window."
    if campaign_names:
        summary = (
            "Campaign overlap detected in analyzed week "
            "(" + ", ".join(f"`{name}`" for name in campaign_names) + ")."
        )
    return {
        "summary": summary,
        "statement": statement,
        "confidence": confidence,
    }


def _top_platform_pulse_line(additional_context: dict[str, object] | None) -> str:
    pulse = (additional_context or {}).get("platform_regulatory_pulse", {})
    if not isinstance(pulse, dict) or not pulse.get("enabled"):
        return ""
    rows = pulse.get("rows", [])
    if not isinstance(rows, list):
        return ""
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return ""
    high_count = sum(
        1 for row in valid_rows if str(row.get("severity", "")).strip().lower() == "high"
    )
    medium_count = sum(
        1 for row in valid_rows if str(row.get("severity", "")).strip().lower() == "medium"
    )
    return (
        "Platform/regulatory pulse: external platform/regulation context is active "
        f"({_fmt_int(len(valid_rows))} signals; high={_fmt_int(high_count)}, medium={_fmt_int(medium_count)}). "
        "Use as supporting context, not standalone root cause."
    )


def _top_unique_feature_movers(
    rows: list[dict[str, object]],
    *,
    limit: int = 3,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_feature: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature_name = str(row.get("feature", "")).strip()
        key = feature_name.lower()
        if not key:
            continue
        prev = by_feature.get(key)
        if prev is None:
            by_feature[key] = row
            continue
        prev_abs = abs(float(prev.get("delta_clicks_vs_previous", 0.0) or 0.0))
        curr_abs = abs(float(row.get("delta_clicks_vs_previous", 0.0) or 0.0))
        if curr_abs > prev_abs:
            by_feature[key] = row

    unique_rows = list(by_feature.values())
    gains = sorted(
        [row for row in unique_rows if float(row.get("delta_clicks_vs_previous", 0.0) or 0.0) > 0.0],
        key=lambda row: float(row.get("delta_clicks_vs_previous", 0.0) or 0.0),
        reverse=True,
    )[: max(1, int(limit))]
    losses = sorted(
        [row for row in unique_rows if float(row.get("delta_clicks_vs_previous", 0.0) or 0.0) < 0.0],
        key=lambda row: float(row.get("delta_clicks_vs_previous", 0.0) or 0.0),
    )[: max(1, int(limit))]
    return gains, losses


def _brand_ads_hypothesis(
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
    additional_context: dict[str, object] | None,
    external_signals: list[ExternalSignal],
) -> dict[str, object] | None:
    brand_proxy = _brand_proxy_from_gsc(segment_diagnostics)
    brand_ctr = _brand_ctr_proxy_from_gsc(segment_diagnostics)
    if not brand_proxy or not brand_ctr:
        return None

    clicks_wow_pct = float(brand_proxy.get("delta_pct_vs_previous", 0.0))
    ctr_wow_pp = float(brand_ctr.get("delta_pp_vs_previous", 0.0))
    huge_drop = clicks_wow_pct <= -15.0 and ctr_wow_pp <= -0.25
    huge_rise = clicks_wow_pct >= 15.0 and ctr_wow_pp >= 0.25
    if not (huge_drop or huge_rise):
        return None

    paid_direction, paid_delta = _trade_plan_paid_pressure(additional_context)
    campaign_context = _campaign_event_context(
        external_signals=external_signals,
        query_scope=None,
    )
    campaign_count = len(campaign_context.get("allegro_events", [])) if isinstance(campaign_context, dict) else 0

    confidence = 58
    evidence_bits: list[str] = [
        f"brand clicks WoW {clicks_wow_pct:+.2f}%",
        f"brand CTR WoW {ctr_wow_pp:+.2f} pp",
    ]
    if paid_direction == "up":
        confidence += 12
        evidence_bits.append(f"paid/search spend signal up ({_fmt_signed_compact(paid_delta)})")
    elif paid_direction == "down":
        confidence += 10
        evidence_bits.append(f"paid/search spend signal down ({_fmt_signed_compact(paid_delta)})")
    if campaign_count > 0:
        confidence += 6
        evidence_bits.append(f"campaign signals present ({campaign_count})")
    confidence = max(45, min(85, confidence))

    if huge_drop:
        statement = (
            "Large WoW decline in brand organic clicks/CTR is consistent with higher brand paid-ad pressure in SERP "
            "(possible organic cannibalization by Google Ads)."
        )
        if paid_direction == "up":
            statement = (
                "Large WoW decline in brand organic clicks/CTR with rising paid-search pressure indicates likely "
                "Google Ads cannibalization on brand queries."
            )
    else:
        statement = (
            "Large WoW increase in brand organic clicks/CTR is consistent with reduced brand paid-ad pressure "
            "(possible lower Google Ads overlap on brand queries)."
        )
        if paid_direction == "down":
            statement = (
                "Large WoW increase in brand organic clicks/CTR with lower paid-search pressure indicates likely "
                "Google Ads brand coverage reduction or pause."
            )

    return {
        "statement": statement,
        "confidence": confidence,
        "evidence": "; ".join(evidence_bits),
    }


def _has_allegro_days_like_signal(
    external_signals: list[ExternalSignal],
    country_code: str,
) -> bool:
    code = country_code.strip().upper()
    if code not in {"CZ", "SK"}:
        return False
    tokens = ("allegro days", "dni allegro", "smart week", "megaraty")
    for signal in external_signals:
        text = _normalize_text(f"{signal.title} {signal.details} {signal.source}")
        if not any(token in text for token in tokens):
            continue
        # Prefer country-targeted campaign sources if present.
        if f"({code.lower()})" in signal.source.lower() or "campaign tracker" in signal.source.lower():
            return True
        # Fallback: allow generic source mentions as directional evidence.
        return True
    return False


def _format_trend_points(
    rows: list[dict[str, object]],
    value_key: str,
    limit: int = 3,
    with_sign: bool = False,
    compact: bool = False,
) -> str:
    points: list[str] = []
    for row in rows[:limit]:
        trend = str(row.get("trend", "")).strip()
        if not trend:
            continue
        value = _safe_float(row, value_key)
        if with_sign:
            metric = _fmt_signed_compact(value) if compact else _fmt_signed_int(value)
        else:
            metric = _fmt_compact(value) if compact else _fmt_int(value)
        points.append(f"`{trend}` ({metric})")
    return ", ".join(points)


def _product_trend_rows(
    additional_context: dict[str, object] | None,
    scope_results: list[tuple[str, AnalysisResult]],
) -> dict[str, object]:
    product_trends = (additional_context or {}).get("product_trends", {})
    if not isinstance(product_trends, dict):
        product_trends = {}

    top_rows_raw = product_trends.get("top_rows", 12)
    try:
        top_rows = max(1, int(top_rows_raw))
    except (TypeError, ValueError):
        top_rows = 12

    query_scope = _find_scope(scope_results, "query")

    yoy_rows = product_trends.get("top_yoy_non_brand", [])
    if not isinstance(yoy_rows, list):
        yoy_rows = []
    yoy_rows = [
        row
        for row in yoy_rows
        if isinstance(row, dict) and not _is_noise_or_brand_query(str(row.get("trend", "")))
    ]

    yoy_fallback = False
    if not yoy_rows:
        yoy_rows = _fallback_non_brand_yoy_from_gsc(query_scope=query_scope, top_rows=top_rows)
        yoy_fallback = bool(yoy_rows)

    current_rows = product_trends.get("current_non_brand", [])
    if not isinstance(current_rows, list):
        current_rows = []
    current_rows = [
        row
        for row in current_rows
        if isinstance(row, dict) and not _is_noise_or_brand_query(str(row.get("trend", "")))
    ]

    upcoming_rows = product_trends.get("upcoming_31d", [])
    if not isinstance(upcoming_rows, list):
        upcoming_rows = []
    upcoming_rows = [
        row
        for row in upcoming_rows
        if isinstance(row, dict) and not _is_noise_or_brand_query(str(row.get("trend", "")))
    ]

    return {
        "enabled": bool(product_trends.get("enabled", False) or yoy_rows or current_rows or upcoming_rows),
        "top_rows": top_rows,
        "horizon_days": int(product_trends.get("horizon_days", 31) or 31),
        "yoy_rows": yoy_rows,
        "current_rows": current_rows,
        "upcoming_rows": upcoming_rows,
        "yoy_fallback": yoy_fallback,
    }


def _infer_trend_drivers(
    trend_names: list[str],
    external_signals: list[ExternalSignal],
) -> list[dict[str, object]]:
    normalized_names = [_normalize_text(name) for name in trend_names if str(name).strip()]
    signal_blob = " ".join(
        _normalize_text(f"{signal.title} {signal.details} {signal.source}") for signal in external_signals
    )
    drivers: list[dict[str, object]] = []

    for label, tokens in TREND_EVENT_DRIVER_RULES:
        matched_raw: list[str] = []
        for name in trend_names:
            normalized_name = _normalize_text(name)
            if any(token in normalized_name for token in tokens):
                matched_raw.append(name)

        if not matched_raw:
            continue

        unique_matched: list[str] = []
        seen: set[str] = set()
        for name in matched_raw:
            norm = _normalize_text(name)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            unique_matched.append(name)

        if not unique_matched:
            continue

        signal_support = any(token in signal_blob for token in tokens)
        drivers.append(
            {
                "label": label,
                "count": len(unique_matched),
                "examples": unique_matched[:2],
                "signal_support": signal_support,
            }
        )

    drivers.sort(
        key=lambda row: (
            int(row.get("count", 0)),
            1 if bool(row.get("signal_support", False)) else 0,
        ),
        reverse=True,
    )
    return drivers[:3]


def _build_product_trend_summary(
    scope_results: list[tuple[str, AnalysisResult]],
    external_signals: list[ExternalSignal],
    additional_context: dict[str, object] | None,
) -> dict[str, str]:
    trend_ctx = _product_trend_rows(
        additional_context=additional_context,
        scope_results=scope_results,
    )
    if not trend_ctx.get("enabled"):
        return {}

    yoy_rows = list(trend_ctx.get("yoy_rows", []))
    current_rows = list(trend_ctx.get("current_rows", []))
    upcoming_rows = list(trend_ctx.get("upcoming_rows", []))
    top_rows = int(trend_ctx.get("top_rows", 12))
    horizon_days = int(trend_ctx.get("horizon_days", 31))

    yoy_line = ""
    if yoy_rows:
        current_sum = sum(_safe_float(row, "current_value") for row in yoy_rows)
        previous_sum = sum(_safe_float(row, "previous_value") for row in yoy_rows)
        delta = current_sum - previous_sum
        delta_pct = (delta / previous_sum * 100.0) if previous_sum else 0.0

        gainers = sorted(
            [row for row in yoy_rows if _safe_float(row, "delta_value") > 0],
            key=lambda row: _safe_float(row, "delta_value"),
            reverse=True,
        )
        decliners = sorted(
            [row for row in yoy_rows if _safe_float(row, "delta_value") < 0],
            key=lambda row: _safe_float(row, "delta_value"),
        )

        decliners_text = _format_trend_points(decliners, "delta_value", limit=2, with_sign=True, compact=True)
        yoy_line = (
            "Non-brand product YoY (top rows): "
            f"net {_fmt_signed_compact(delta)} ({delta_pct:+.2f}%)."
        )
        gainers_text = _format_trend_points(gainers, "delta_value", limit=3, with_sign=True, compact=True)
        if gainers_text:
            yoy_line += f" Top gainers: {gainers_text}."
        if decliners_text:
            yoy_line += f" Top decliners: {decliners_text}."
        if bool(trend_ctx.get("yoy_fallback")):
            yoy_line += " Source fallback: GSC query trends."

    current_line = ""
    if current_rows:
        ranked_current = sorted(
            current_rows,
            key=lambda row: _safe_float(row, "value"),
            reverse=True,
        )
        current_line = (
            "Current non-brand trend leaders: "
            f"{_format_trend_points(ranked_current, 'value', limit=5, compact=True)}."
        )

    upcoming_line = ""
    if upcoming_rows:
        ranked_upcoming = sorted(
            upcoming_rows,
            key=lambda row: _safe_float(row, "value"),
            reverse=True,
        )
        upcoming_line = (
            f"Next {horizon_days} days watchlist: "
            f"{_format_trend_points(ranked_upcoming, 'value', limit=5, compact=True)}."
        )

    trend_names: list[str] = []
    for row in (yoy_rows[:top_rows] + current_rows[:top_rows] + upcoming_rows[:top_rows]):
        if not isinstance(row, dict):
            continue
        trend = str(row.get("trend", "")).strip()
        if trend:
            trend_names.append(trend)
    drivers = _infer_trend_drivers(trend_names=trend_names, external_signals=external_signals)

    drivers_line = ""
    if drivers:
        parts = []
        for row in drivers:
            label = str(row.get("label", "")).strip()
            count = int(row.get("count", 0))
            examples = row.get("examples", [])
            if isinstance(examples, list):
                example_text = ", ".join(f"`{str(item).strip()}`" for item in examples[:2] if str(item).strip())
            else:
                example_text = ""
            support = "supported by external signals" if bool(row.get("signal_support")) else "weak external support"
            if example_text:
                parts.append(f"{label} ({count} trends; {support}; e.g. {example_text})")
            else:
                parts.append(f"{label} ({count} trends; {support})")
        drivers_line = "Likely event/campaign drivers: " + "; ".join(parts) + "."

    executive_line = ""
    if yoy_rows or current_rows or upcoming_rows:
        snippet_parts: list[str] = []
        if yoy_rows:
            current_sum = sum(_safe_float(row, "current_value") for row in yoy_rows)
            previous_sum = sum(_safe_float(row, "previous_value") for row in yoy_rows)
            delta = current_sum - previous_sum
            snippet_parts.append(f"YoY net {_fmt_signed_int(delta)} across top non-brand rows")
        if current_rows:
            snippet_parts.append(
                f"current leaders: {_format_trend_points(sorted(current_rows, key=lambda row: _safe_float(row, 'value'), reverse=True), 'value', limit=2)}"
            )
        if drivers:
            driver_names = ", ".join(str(row.get("label", "")) for row in drivers[:2] if str(row.get("label", "")).strip())
            if driver_names:
                snippet_parts.append(f"likely drivers: {driver_names}")
        executive_line = "Product-trend summary: " + "; ".join(snippet_parts) + "."

    return {
        "executive_line": executive_line,
        "yoy_line": yoy_line,
        "current_line": current_line,
        "upcoming_line": upcoming_line,
        "drivers_line": drivers_line,
    }


def _build_appendix_highlights_lines(
    run_date: date,
    windows: dict[str, DateWindow],
    scope_results: list[tuple[str, AnalysisResult]],
    external_signals: list[ExternalSignal],
    additional_context: dict[str, object] | None,
) -> list[str]:
    lines: list[str] = []

    ga4 = (additional_context or {}).get("ga4", {})
    if USE_GA4_IN_REPORT and isinstance(ga4, dict) and ga4.get("enabled"):
        summary = ga4.get("summary", {})
        if isinstance(summary, dict):
            current = summary.get("current", {})
            previous = summary.get("previous", {})
            if isinstance(current, dict) and isinstance(previous, dict):
                sessions_current = _ga4_num(current, "sessions")
                sessions_previous = _ga4_num(previous, "sessions")
                revenue_current = _ga4_num(current, "revenue")
                revenue_previous = _ga4_num(previous, "revenue")
                ga4_parts: list[str] = []
                if sessions_current is not None and sessions_previous is not None:
                    sessions_delta = sessions_current - sessions_previous
                    sessions_delta_pct = (
                        (sessions_delta / sessions_previous * 100.0)
                        if sessions_previous
                        else 0.0
                    )
                    ga4_parts.append(
                        f"sessions {_fmt_int(sessions_current)} ({sessions_delta_pct:+.2f}% WoW)"
                    )
                if revenue_current is not None and revenue_previous is not None:
                    revenue_delta = revenue_current - revenue_previous
                    revenue_delta_pct = (
                        (revenue_delta / revenue_previous * 100.0)
                        if revenue_previous
                        else 0.0
                    )
                    ga4_parts.append(
                        f"purchase revenue {_fmt_int(revenue_current)} ({revenue_delta_pct:+.2f}% WoW)"
                    )
                if ga4_parts:
                    lines.append("- GA4 weekly check: " + ", ".join(ga4_parts) + ".")

                channels = ga4.get("channels", {})
                yoy_deltas = channels.get("yoy_deltas", []) if isinstance(channels, dict) else []
                if isinstance(yoy_deltas, list) and yoy_deltas:
                    top_riser = max(
                        (row for row in yoy_deltas if isinstance(row, dict)),
                        key=lambda row: float(row.get("delta_vs_yoy", 0.0) or 0.0),
                        default=None,
                    )
                    top_faller = min(
                        (row for row in yoy_deltas if isinstance(row, dict)),
                        key=lambda row: float(row.get("delta_vs_yoy", 0.0) or 0.0),
                        default=None,
                    )
                    if isinstance(top_riser, dict) and isinstance(top_faller, dict):
                        lines.append(
                            "- GA4 YoY channels: "
                            f"top riser `{top_riser.get('channel', '')}` ({_fmt_signed_int(top_riser.get('delta_vs_yoy', 0.0))}), "
                            f"top decline `{top_faller.get('channel', '')}` ({_fmt_signed_int(top_faller.get('delta_vs_yoy', 0.0))})."
                        )
    elif USE_GA4_IN_REPORT and isinstance(ga4, dict):
        errors = ga4.get("errors", [])
        if isinstance(errors, list) and errors:
            lines.append(f"- GA4 warning: {str(errors[0]).strip()}")

    senuto_intelligence = (additional_context or {}).get("senuto_intelligence", {})
    if isinstance(senuto_intelligence, dict) and senuto_intelligence.get("enabled"):
        competitors = senuto_intelligence.get("competitors_overview", [])
        if isinstance(competitors, list) and competitors:
            top_comp = next(
                (row for row in competitors if isinstance(row, dict)),
                None,
            )
            if isinstance(top_comp, dict):
                lines.append(
                    "- Senuto competitor radar: top overlapping competitor "
                    f"`{top_comp.get('domain', '')}` with "
                    f"{_fmt_int(top_comp.get('common_keywords', 0.0))} common keywords."
                )

    allegro_trends = (additional_context or {}).get("allegro_trends", {})
    if isinstance(allegro_trends, dict) and allegro_trends.get("enabled"):
        rows = allegro_trends.get("rows", [])
        if isinstance(rows, list) and rows:
            top_row = next((row for row in rows if isinstance(row, dict)), None)
            if isinstance(top_row, dict):
                lines.append(
                    "- Allegro Trends API: top tracked query "
                    f"`{str(top_row.get('query', '')).strip()}` "
                    f"with GMV {_fmt_int(top_row.get('gmv', 0.0))} and visits {_fmt_int(top_row.get('visit', 0.0))} "
                    f"in {str(allegro_trends.get('from', '')).strip()} to {str(allegro_trends.get('till', '')).strip()}."
                )
    elif isinstance(allegro_trends, dict):
        errors = allegro_trends.get("errors", [])
        if isinstance(errors, list) and errors:
            lines.append(f"- Allegro Trends warning: {str(errors[0]).strip()}")

    trend_summary = _build_product_trend_summary(
        scope_results=scope_results,
        external_signals=external_signals,
        additional_context=additional_context,
    )
    for key in ("yoy_line", "current_line", "upcoming_line", "drivers_line"):
        text = trend_summary.get(key, "").strip()
        if text:
            lines.append(f"- {text}")

    cluster_line = _query_cluster_summary_line(_find_scope(scope_results, "query"))
    if cluster_line:
        lines.append(f"- {cluster_line}")

    campaign_context = _campaign_event_context(
        external_signals=external_signals,
        query_scope=_find_scope(scope_results, "query"),
    )
    allegro_events = campaign_context.get("allegro_events", [])
    competitor_events = campaign_context.get("competitor_events", [])
    query_events = campaign_context.get("query_events", [])
    allegro_count = len(allegro_events) if isinstance(allegro_events, list) else 0
    competitor_count = len(competitor_events) if isinstance(competitor_events, list) else 0
    query_count = len(query_events) if isinstance(query_events, list) else 0
    if allegro_count or competitor_count or query_count:
        campaign_days: list[date] = []
        if isinstance(allegro_events, list):
            campaign_days.extend(
                row.day for row in allegro_events if isinstance(row, ExternalSignal)
            )
        if isinstance(competitor_events, list):
            campaign_days.extend(
                row[0].day
                for row in competitor_events
                if isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], ExternalSignal)
            )
        period_label = ""
        if campaign_days:
            period_label = (
                f" Window: {min(campaign_days).isoformat()} to {max(campaign_days).isoformat()}."
            )
        line = (
            "- Campaign-event context from appendix: "
            f"Allegro mentions={allegro_count}, competitor mentions={competitor_count}, "
            f"campaign query movers={query_count}.{period_label}"
        )
        if isinstance(allegro_events, list) and allegro_events:
            latest = allegro_events[0]
            line += f" Latest Allegro signal: {latest.day.isoformat()} (`{latest.title}`)."
        elif isinstance(competitor_events, list) and competitor_events:
            latest_signal, competitor = competitor_events[0]
            line += (
                f" Latest competitor signal: {latest_signal.day.isoformat()} "
                f"({competitor}: `{latest_signal.title}`)."
            )
        lines.append(line)

    market_events = (additional_context or {}).get("market_event_calendar", {})
    if isinstance(market_events, dict) and market_events.get("enabled"):
        rows = market_events.get("events", [])
        if isinstance(rows, list) and rows:
            high_count = sum(
                1
                for row in rows
                if isinstance(row, dict)
                and str(row.get("impact_level", "")).strip().upper() == "HIGH"
            )
            event_days: list[date] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                day_text = str(row.get("date", "")).strip()
                if not day_text:
                    continue
                try:
                    day = date.fromisoformat(day_text[:10])
                except ValueError:
                    continue
                event_days.append(day)
            period = ""
            if event_days:
                period = f" Window: {min(event_days).isoformat()} to {max(event_days).isoformat()}."
            lines.append(
                "- Market-event API context: "
                f"{_fmt_int(len(rows))} events for {str(market_events.get('country_code', '')).strip() or 'market'}, "
                f"high-impact candidates={_fmt_int(high_count)}.{period}"
            )

    trends = (additional_context or {}).get("google_trends", [])
    if isinstance(trends, list) and trends:
        top_topics: list[str] = []
        for row in trends[:3]:
            if not isinstance(row, dict):
                continue
            topic = str(row.get("topic", "")).strip()
            if not topic:
                continue
            traffic = int(row.get("approx_traffic", 0))
            top_topics.append(f"`{topic}` ({_fmt_int(traffic)})")
        if top_topics:
            country_label = str((additional_context or {}).get("country_code", "")).strip() or "market"
            lines.append(f"- Google Trends ({country_label}) top topics: " + ", ".join(top_topics) + ".")

    pagespeed = (additional_context or {}).get("pagespeed", {})
    if isinstance(pagespeed, dict) and pagespeed:
        mobile = pagespeed.get("mobile", {})
        if isinstance(mobile, dict):
            lcp = float(mobile.get("lcp_ms", 0.0))
            inp = float(mobile.get("inp_ms", 0.0))
            cls = float(mobile.get("cls", 0.0))
            if lcp or inp or cls:
                lines.append(
                    "- CrUX mobile snapshot: "
                    f"LCP={lcp:.0f}ms, INP={inp:.0f}ms, CLS={cls:.2f} "
                    "(use this when traffic shifts are suspiciously page/device-specific)."
                )

    macro = (additional_context or {}).get("macro", {})
    if isinstance(macro, dict):
        nbp = macro.get("nbp_fx", {})
        if isinstance(nbp, dict):
            eur = nbp.get("eur_pln", {})
            usd = nbp.get("usd_pln", {})
            if isinstance(eur, dict) and isinstance(usd, dict):
                eur_delta = float(eur.get("delta_pct_vs_previous", 0.0))
                usd_delta = float(usd.get("delta_pct_vs_previous", 0.0))
                if eur_delta or usd_delta:
                    lines.append(
                        "- FX context vs previous week: "
                        f"EUR/PLN {eur_delta:+.2f}%, USD/PLN {usd_delta:+.2f}%."
                    )

    current_window = windows.get("current_28d")
    previous_window = windows.get("previous_28d")
    if isinstance(current_window, DateWindow) and isinstance(previous_window, DateWindow):
        lines.append(
            "- Date windows used in this run: "
            f"current={current_window.start.isoformat()} to {current_window.end.isoformat()}, "
            f"previous={previous_window.start.isoformat()} to {previous_window.end.isoformat()}."
        )

    return lines[:12]


def _summarize_query_delta_rows(rows: list[KeyDelta], limit: int = 5) -> str:
    points: list[str] = []
    for row in rows[:limit]:
        key = str(row.key).strip()
        if not key:
            continue
        points.append(
            f"`{key}` ({_fmt_signed_int(row.click_delta_vs_previous)} vs prev; prev={_fmt_int(row.previous_clicks)})"
        )
    return "; ".join(points)


def _top_non_brand_trends_line(
    scope_results: list[tuple[str, AnalysisResult]],
    additional_context: dict[str, object] | None,
    limit: int = 5,
) -> str:
    trend_ctx = _product_trend_rows(
        additional_context=additional_context,
        scope_results=scope_results,
    )
    current_rows = trend_ctx.get("current_rows", [])
    if isinstance(current_rows, list) and current_rows:
        ranked = sorted(
            [row for row in current_rows if isinstance(row, dict)],
            key=lambda row: _safe_float(row, "value"),
            reverse=True,
        )
        points = _format_trend_points(ranked, "value", limit=limit)
        if points:
            return f"Top {limit} current non-brand trends: {points}."

    yoy_rows = trend_ctx.get("yoy_rows", [])
    if isinstance(yoy_rows, list) and yoy_rows:
        ranked = sorted(
            [row for row in yoy_rows if isinstance(row, dict)],
            key=lambda row: abs(_safe_float(row, "delta_value")),
            reverse=True,
        )
        points = _format_trend_points(ranked, "delta_value", limit=limit, with_sign=True)
        if points:
            return f"Top {limit} YoY non-brand trend movers: {points}."
    return ""


def _build_executive_summary_lines(
    totals: dict[str, MetricSummary],
    scope_results: list[tuple[str, AnalysisResult]],
    hypotheses: list[dict[str, object]],
    external_signals: list[ExternalSignal],
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
    additional_context: dict[str, object] | None,
    senuto_summary: dict[str, float] | None,
    senuto_error: str | None,
) -> list[str]:
    current = totals["current_28d"]
    previous = totals["previous_28d"]
    yoy = totals["yoy_52w"]

    click_wow_pct = _signed_pct(_ratio_delta(current.clicks, previous.clicks))
    click_yoy_pct = _signed_pct(_ratio_delta(current.clicks, yoy.clicks))
    impr_wow_pct = _signed_pct(_ratio_delta(current.impressions, previous.impressions))
    impr_yoy_pct = _signed_pct(_ratio_delta(current.impressions, yoy.impressions))
    ctr_wow_pp = (current.ctr - previous.ctr) * 100.0
    ctr_yoy_pp = (current.ctr - yoy.ctr) * 100.0
    pos_yoy_delta = current.position - yoy.position

    lines: list[str] = []

    query_scope = _find_scope(scope_results, "query")
    growth_themes = ""
    decline_themes = ""
    val_timing_line = ""
    if query_scope:
        growth_themes, decline_themes = _executive_yoy_theme_summary(query_scope, limit=3)
        cluster_rows = _query_cluster_rows(query_scope)
        val_row = next(
            (row for row in cluster_rows if str(row.get("cluster", "")).strip() == "Valentine's demand"),
            None,
        )
        if isinstance(val_row, dict):
            val_wow = float(val_row.get("delta_vs_previous", 0.0))
            val_yoy = float(val_row.get("delta_vs_yoy", 0.0))
            if val_wow > 0 and val_yoy < 0:
                val_timing_line = (
                    "Valentine's cluster is up WoW "
                    f"({_fmt_signed_compact(val_wow)}) but still down YoY ({_fmt_signed_compact(val_yoy)}), "
                    "which points to timing shift rather than structural annual growth."
                )

    template_rows = (segment_diagnostics or {}).get("page_template") or []
    home_wow = 0.0
    home_yoy = 0.0
    home_wow_pct = ""
    home_yoy_pct = ""
    brand_wow = 0.0
    brand_yoy = 0.0
    brand_enabled = False
    brand_note = ""
    if isinstance(template_rows, list) and template_rows:
        home_row = next(
            (
                row
                for row in template_rows
                if str(row.get("segment", "")).strip().lower() == "home"
            ),
            None,
        )
        if isinstance(home_row, dict):
            home_wow = float(home_row.get("delta_vs_previous", 0.0))
            home_yoy = float(home_row.get("delta_vs_yoy", 0.0))
            home_prev = float(home_row.get("previous_clicks", 0.0))
            home_yoy_clicks = float(home_row.get("yoy_clicks", 0.0))
            home_wow_pct = _signed_pct(_ratio_delta(home_prev + home_wow, home_prev))
            home_yoy_pct = _signed_pct(_ratio_delta(home_yoy_clicks + home_yoy, home_yoy_clicks))
            if home_wow < 0 and home_prev >= 10_000:
                brand_context = (additional_context or {}).get("google_trends_brand", {})
                brand_summary = brand_context.get("summary", {}) if isinstance(brand_context, dict) else {}
                if isinstance(brand_summary, dict):
                    brand_wow = float(brand_summary.get("delta_pct_vs_previous", 0.0))
                    brand_yoy = float(brand_summary.get("delta_pct_vs_yoy", 0.0))
                    brand_enabled = bool(brand_context.get("enabled")) if isinstance(brand_context, dict) else False
                    if brand_wow > 0:
                        brand_note = "Brand demand is up WoW in Google Trends, so routing/SERP mix is more likely than demand collapse."
                    elif brand_wow < 0:
                        brand_note = "Brand demand is also down WoW in Google Trends, so part of homepage softness is likely demand-side."

    if not brand_enabled:
        brand_context = (additional_context or {}).get("google_trends_brand", {})
        brand_summary = brand_context.get("summary", {}) if isinstance(brand_context, dict) else {}
        if isinstance(brand_context, dict) and isinstance(brand_summary, dict) and brand_context.get("enabled"):
            brand_wow = float(brand_summary.get("delta_pct_vs_previous", 0.0))
            brand_yoy = float(brand_summary.get("delta_pct_vs_yoy", 0.0))
            brand_enabled = True

    # Executive summary in decision format.
    lines.append(
        "- **What changed**: GSC organic moved to **clicks "
        f"{_fmt_compact(current.clicks)} ({click_wow_pct} WoW; {click_yoy_pct} YoY)**, "
        f"impressions {_fmt_compact(current.impressions)} ({impr_wow_pct} WoW; {impr_yoy_pct} YoY), "
        f"CTR {_pct(current.ctr)} ({ctr_wow_pp:+.2f} pp WoW; {ctr_yoy_pp:+.2f} pp YoY), "
        f"avg position {current.position:.2f} ({pos_yoy_delta:+.2f} YoY)."
    )
    why_parts: list[str] = []
    if growth_themes:
        why_parts.append(f"growth themes: {growth_themes}")
    if decline_themes:
        why_parts.append(f"decline themes: {decline_themes}")
    why_text = "; ".join(why_parts) if why_parts else "cluster-level pattern shows category-level demand rotation rather than one systemic SEO failure mode"
    lines.append(
        "- **Why**: current movement is **driven by category-level demand rotation with stable efficiency**; "
        + why_text
        + "."
        + (f" {val_timing_line}" if val_timing_line else "")
    )
    business_text = (
        "short-term risk is mainly allocation (which Page Names capture demand) rather than broad ranking/indexation degradation."
    )
    trade_plan_signal = _trade_plan_signal(additional_context)
    if home_wow < 0:
        business_text += (
            f" **home** is down **{_fmt_signed_compact(home_wow)} WoW ({home_wow_pct})** and "
            f"{_fmt_signed_compact(home_yoy)} YoY ({home_yoy_pct})."
        )
    if brand_note:
        business_text += f" {brand_note}"
    lines.append("- **Business implication**: " + business_text)
    if isinstance(trade_plan_signal, dict):
        lines.append(
            "- **Trade-plan hypothesis**: "
            + str(trade_plan_signal.get("statement", "")).strip()
            + f" (confidence {int(trade_plan_signal.get('confidence', 0) or 0)}/100)."
        )
    brand_proxy = _brand_proxy_from_gsc(segment_diagnostics)
    if brand_enabled and brand_proxy:
        lines.append(
            "- **Brand demand baseline (Google Trends + GSC brand queries)**: "
            f"Google Trends **{brand_wow:+.2f}% WoW**, **{brand_yoy:+.2f}% YoY**; "
            f"GSC brand clicks **{brand_proxy.get('delta_pct_vs_previous', 0.0):+.2f}% WoW** "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_previous', 0.0))}), "
            f"**{brand_proxy.get('delta_pct_vs_yoy', 0.0):+.2f}% YoY** "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_yoy', 0.0))}); "
            f"GSC brand impressions **{brand_proxy.get('impressions_delta_pct_vs_previous', 0.0):+.2f}% WoW** "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_previous', 0.0))}), "
            f"**{brand_proxy.get('impressions_delta_pct_vs_yoy', 0.0):+.2f}% YoY** "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_yoy', 0.0))})."
        )
    elif brand_enabled:
        lines.append(
            "- **Brand search trend (Google Trends)**: "
            f"WoW **{brand_wow:+.2f}%**, YoY **{brand_yoy:+.2f}%**. "
            "GSC brand-query proxy was unavailable in this run."
        )
    elif brand_proxy:
        lines.append(
            "- **Brand demand proxy (GSC brand queries)**: "
            f"WoW **{brand_proxy.get('delta_pct_vs_previous', 0.0):+.2f}%** "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_previous', 0.0))}), "
            f"impressions WoW **{brand_proxy.get('impressions_delta_pct_vs_previous', 0.0):+.2f}%** "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_previous', 0.0))}); "
            f"YoY **{brand_proxy.get('delta_pct_vs_yoy', 0.0):+.2f}%** "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_yoy', 0.0))}), "
            f"impressions YoY **{brand_proxy.get('impressions_delta_pct_vs_yoy', 0.0):+.2f}%** "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_yoy', 0.0))}). "
            "Google Trends was unavailable, so this proxy is used for brand/home interpretation."
        )
    else:
        lines.append(
            "- **Brand search trend (Google Trends)**: unavailable in this run (no valid brand time-series response), "
            "and GSC brand-proxy segment was not available."
        )

    country_code = str((additional_context or {}).get("country_code", "")).strip().upper()
    brand_ctr_proxy = _brand_ctr_proxy_from_gsc(segment_diagnostics)
    if (
        country_code in {"CZ", "SK"}
        and _has_allegro_days_like_signal(external_signals, country_code=country_code)
        and isinstance(brand_ctr_proxy, dict)
        and float(brand_ctr_proxy.get("delta_pp_vs_previous", 0.0)) <= -0.10
    ):
        lines.append(
            "- **Brand CTR campaign hypothesis (CZ/SK)**: detected `Allegro Days`-type campaign context with "
            f"brand CTR down **{float(brand_ctr_proxy.get('delta_pp_vs_previous', 0.0)):+.2f} pp WoW**. "
            "A likely driver is stronger brand paid-search presence (Google Ads) capturing part of clicks from organic brand results."
        )
    brand_ads_hyp = _brand_ads_hypothesis(
        segment_diagnostics=segment_diagnostics,
        additional_context=additional_context,
        external_signals=external_signals,
    )
    if isinstance(brand_ads_hyp, dict):
        lines.append(
            "- **Brand Ads overlap hypothesis (WoW)**: "
            f"{str(brand_ads_hyp.get('statement', '')).strip()} "
            f"(confidence {int(brand_ads_hyp.get('confidence', 0) or 0)}/100; "
            f"evidence: {str(brand_ads_hyp.get('evidence', '')).strip()})."
        )
    lines.append(
        "- **Decision this week**: current data indicates an exposure/demand effect first "
        "(brand demand + impressions/CTR), while campaign timing and paid activity are likely incremental factors; "
        "treat technical SEO escalation as second step unless efficiency signals deteriorate."
    )

    market_events = (additional_context or {}).get("market_event_calendar", {})
    if isinstance(market_events, dict) and market_events.get("enabled"):
        rows = market_events.get("events", [])
        if isinstance(rows, list) and rows:
            high_count = sum(
                1
                for row in rows
                if isinstance(row, dict)
                and str(row.get("impact_level", "")).strip().upper() == "HIGH"
            )
            if high_count > 0:
                lines.append(
                    "- `Context`: market-event calendar "
                    f"({str(market_events.get('country_code', '')).strip() or 'market'}) captured "
                    f"{_fmt_int(len(rows))} items, including {_fmt_int(high_count)} high-impact GMV candidates."
                )

    ga4 = (additional_context or {}).get("ga4", {})
    if USE_GA4_IN_REPORT and isinstance(ga4, dict) and ga4.get("enabled"):
        summary = ga4.get("summary", {})
        if isinstance(summary, dict):
            current_ga4 = summary.get("current", {})
            yoy_ga4 = summary.get("yoy", {})
            if isinstance(current_ga4, dict) and isinstance(yoy_ga4, dict):
                ga4_parts: list[str] = []
                sessions_current = _ga4_num(current_ga4, "sessions")
                sessions_yoy = _ga4_num(yoy_ga4, "sessions")
                if sessions_current is not None and sessions_yoy is not None:
                    ga4_parts.append(
                        f"sessions {_fmt_int(sessions_current)} ({_signed_pct(_ratio_delta(sessions_current, sessions_yoy))})"
                    )
                transactions_current = _ga4_num(current_ga4, "transactions")
                transactions_yoy = _ga4_num(yoy_ga4, "transactions")
                if transactions_current is not None and transactions_yoy is not None:
                    ga4_parts.append(
                        f"transactions {_fmt_int(transactions_current)} ({_signed_pct(_ratio_delta(transactions_current, transactions_yoy))})"
                    )
                revenue_current = _ga4_num(current_ga4, "revenue")
                revenue_yoy = _ga4_num(yoy_ga4, "revenue")
                if revenue_current is not None and revenue_yoy is not None:
                    ga4_parts.append(
                        f"purchase revenue {_fmt_int(revenue_current)} ({_signed_pct(_ratio_delta(revenue_current, revenue_yoy))})"
                    )
                if ga4_parts:
                    lines.append("- `YoY` (GA4): " + ", ".join(ga4_parts) + " vs aligned 52W window.")
    # When GA4 is excluded by policy, keep Executive Summary concise and avoid policy-only filler lines.

    if hypotheses:
        top_hypothesis = hypotheses[0] if hypotheses else {}
        thesis = str(top_hypothesis.get("thesis", "")).strip()
        confidence = int(top_hypothesis.get("confidence", 0))
        if thesis:
            thesis_short = thesis.split(".")[0].strip()
            lines.append(
                f"- Primary interpretation [{confidence}/100]: {thesis_short}."
            )

    if senuto_summary:
        if "avg_delta_vs_yoy_pct" in senuto_summary:
            delta = float(senuto_summary.get("avg_delta_vs_yoy_pct", 0.0))
            if abs(delta) >= 0.01:
                lines.append(
                    "- Senuto visibility check (YoY): "
                    f"avg {delta:+.2f}% vs YoY."
                )
        else:
            delta = float(senuto_summary.get("avg_delta_pct", 0.0))
            if abs(delta) >= 0.01:
                lines.append(
                    "- Senuto visibility check: "
                    f"avg {delta:+.2f}% vs previous week."
                )
    elif senuto_error:
        lines.append("- Senuto visibility missing or inconsistent in this run; treat Senuto-derived opportunities as directional only.")

    return lines[:6]


def _build_leadership_snapshot_lines(
    totals: dict[str, MetricSummary],
    scope_results: list[tuple[str, AnalysisResult]],
) -> list[str]:
    current = totals["current_28d"]
    previous = totals["previous_28d"]
    yoy = totals["yoy_52w"]
    wow_pct = _signed_pct(_ratio_delta(current.clicks, previous.clicks))
    yoy_pct = _signed_pct(_ratio_delta(current.clicks, yoy.clicks))

    query_scope = _find_scope(scope_results, "query")
    cluster_rows = _query_cluster_rows(query_scope)
    gains = sorted(
        [row for row in cluster_rows if float(row.get("delta_vs_previous", 0.0)) > 0],
        key=lambda row: float(row.get("delta_vs_previous", 0.0)),
        reverse=True,
    )[:2]
    losses = sorted(
        [row for row in cluster_rows if float(row.get("delta_vs_previous", 0.0)) < 0],
        key=lambda row: float(row.get("delta_vs_previous", 0.0)),
    )[:2]

    gains_text = "; ".join(
        f"{row.get('cluster', '')} ({_fmt_signed_compact(row.get('delta_vs_previous', 0.0))})"
        for row in gains
        if str(row.get("cluster", "")).strip()
    )
    losses_text = "; ".join(
        f"{row.get('cluster', '')} ({_fmt_signed_compact(row.get('delta_vs_previous', 0.0))})"
        for row in losses
        if str(row.get("cluster", "")).strip()
    )

    return [
        (
            "- **What changed**: **GSC organic clicks** moved "
            f"**{_fmt_signed_compact(current.clicks - previous.clicks)} WoW ({wow_pct})** and "
            f"**{_fmt_signed_compact(current.clicks - yoy.clicks)} YoY ({yoy_pct})**. "
            + (f"Main gains: {gains_text}. " if gains_text else "")
            + (f"Main declines: {losses_text}." if losses_text else "")
        ).strip(),
        (
            "- **Why it matters**: the current pattern is mainly **category-level demand rotation** "
            "(seasonality/event timing) rather than broad SEO efficiency deterioration, "
            "so decisions should prioritize demand timing and paid/campaign impact quantification before heavy technical escalation."
        ),
    ]


def _confidence_for_category(
    hypotheses: list[dict[str, object]],
    category_tokens: tuple[str, ...],
) -> int | None:
    for row in hypotheses:
        category = _normalize_text(str(row.get("category", "")))
        if any(token in category for token in category_tokens):
            try:
                return int(row.get("confidence", 0))
            except (TypeError, ValueError):
                return None
    return None


def _latest_google_update_signal(
    external_signals: list[ExternalSignal],
) -> ExternalSignal | None:
    updates = [
        signal
        for signal in external_signals
        if signal.source in {"Google Search Status", "Google Search Central Blog"}
        and "update" in _normalize_text(f"{signal.title} {signal.details}")
    ]
    if not updates:
        return None
    return max(updates, key=lambda signal: signal.day)


def _build_what_is_happening_lines(
    totals: dict[str, MetricSummary],
    windows: dict[str, DateWindow],
    scope_results: list[tuple[str, AnalysisResult]],
    hypotheses: list[dict[str, object]],
    external_signals: list[ExternalSignal],
    weather_summary: dict[str, float],
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
    additional_context: dict[str, object] | None,
) -> list[str]:
    current = totals["current_28d"]
    previous = totals["previous_28d"]
    yoy = totals["yoy_52w"]
    current_window = windows.get("current_28d")
    previous_window = windows.get("previous_28d")
    query_scope = _find_scope(scope_results, "query")
    cluster_rows = _query_cluster_rows(query_scope)

    seasonality_conf = _confidence_for_category(hypotheses, ("seasonality",))
    events_conf = _confidence_for_category(hypotheses, ("events",))
    template_conf = _confidence_for_category(hypotheses, ("template",))
    algorithm_conf = _confidence_for_category(hypotheses, ("algorithm",))
    weather_conf = _confidence_for_category(hypotheses, ("macro", "weather"))

    period_label = ""
    if isinstance(current_window, DateWindow) and isinstance(previous_window, DateWindow):
        period_label = (
            f"{current_window.start.isoformat()} to {current_window.end.isoformat()} "
            f"vs {previous_window.start.isoformat()} to {previous_window.end.isoformat()}"
        )

    def _cluster_change_text(row: dict[str, object], field: str = "delta_vs_previous") -> str:
        delta = float(row.get(field, 0.0))
        current_clicks = float(row.get("current_clicks", 0.0))
        baseline = current_clicks - delta if field == "delta_vs_previous" else current_clicks - delta
        pct = _ratio_delta(current_clicks, baseline) if baseline else 0.0
        return f"{str(row.get('cluster', '')).strip()} ({_fmt_signed_compact(delta)}; {_signed_pct(pct)})"

    gains = sorted(
        [row for row in cluster_rows if float(row.get("delta_vs_previous", 0.0)) > 0],
        key=lambda row: float(row.get("delta_vs_previous", 0.0)),
        reverse=True,
    )[:3]
    losses = sorted(
        [row for row in cluster_rows if float(row.get("delta_vs_previous", 0.0)) < 0],
        key=lambda row: float(row.get("delta_vs_previous", 0.0)),
    )[:3]
    gain_text = "; ".join(_cluster_change_text(row) for row in gains if str(row.get("cluster", "")).strip())
    loss_text = "; ".join(_cluster_change_text(row) for row in losses if str(row.get("cluster", "")).strip())

    winter_cluster = next(
        (row for row in cluster_rows if str(row.get("cluster", "")).strip() == "Winter & weather demand"),
        None,
    )
    event_cluster = next(
        (row for row in cluster_rows if str(row.get("cluster", "")).strip() == "WOŚP & charity events"),
        None,
    )
    non_brand_cluster = next(
        (row for row in cluster_rows if str(row.get("cluster", "")).strip() == "Other non-brand demand"),
        None,
    )
    winter_yoy_delta = float(winter_cluster.get("delta_vs_yoy", 0.0)) if isinstance(winter_cluster, dict) else 0.0
    event_yoy_delta = float(event_cluster.get("delta_vs_yoy", 0.0)) if isinstance(event_cluster, dict) else 0.0
    non_brand_yoy_delta = float(non_brand_cluster.get("delta_vs_yoy", 0.0)) if isinstance(non_brand_cluster, dict) else 0.0

    total_yoy_delta = float(current.clicks - yoy.clicks)
    non_brand_share = 0.0
    if total_yoy_delta != 0:
        non_brand_share = (non_brand_yoy_delta / total_yoy_delta) * 100.0

    home_wow = 0.0
    home_yoy = 0.0
    home_wow_pct = ""
    home_yoy_pct = ""
    page_name_line = ""
    template_rows = (segment_diagnostics or {}).get("page_template")
    if template_rows:
        home_row = next(
            (
                row
                for row in template_rows
                if str(row.get("segment", "")).strip().lower() == "home"
            ),
            None,
        )
        if home_row and float(home_row.get("delta_vs_previous", 0.0)) < 0:
            home_wow = float(home_row.get("delta_vs_previous", 0.0))
            home_yoy = float(home_row.get("delta_vs_yoy", 0.0))
            home_prev = float(home_row.get("previous_clicks", 0.0))
            home_yoy_clicks = float(home_row.get("yoy_clicks", 0.0))
            home_wow_pct = _signed_pct(_ratio_delta(home_prev + home_wow, home_prev))
            home_yoy_pct = _signed_pct(_ratio_delta(home_yoy_clicks + home_yoy, home_yoy_clicks))
            page_name_line = (
                f"home is down {_fmt_signed_compact(home_wow)} WoW ({home_wow_pct}) "
                f"and {_fmt_signed_compact(home_yoy)} YoY ({home_yoy_pct})."
            )

    brand_note = ""
    brand_wow = 0.0
    brand_yoy = 0.0
    brand_enabled = False
    brand_context = (additional_context or {}).get("google_trends_brand", {})
    brand_summary = brand_context.get("summary", {}) if isinstance(brand_context, dict) else {}
    if isinstance(brand_summary, dict):
        brand_wow = float(brand_summary.get("delta_pct_vs_previous", 0.0))
        brand_yoy = float(brand_summary.get("delta_pct_vs_yoy", 0.0))
        brand_enabled = bool(brand_context.get("enabled")) if isinstance(brand_context, dict) else False
        if brand_wow > 0:
            brand_note = (
                f"Google Trends brand demand is up WoW ({brand_wow:+.2f}%), "
                "which strengthens routing/SERP-mix hypothesis over pure demand collapse."
            )
        elif brand_wow < 0:
            brand_note = (
                f"Google Trends brand demand is down WoW ({brand_wow:+.2f}%), "
                "so demand-side softness is part of the homepage decline."
            )
        if home_yoy < 0 and brand_yoy >= 0:
            brand_note += " YoY brand interest is stable/up, so URL allocation remains a high-priority check."

    wow_pct = _signed_pct(_ratio_delta(current.clicks, previous.clicks))
    yoy_pct = _signed_pct(_ratio_delta(current.clicks, yoy.clicks))
    ctr_wow_pp = (current.ctr - previous.ctr) * 100.0
    pos_wow_delta = current.position - previous.position

    lines: list[str] = []
    lines.append(
        "In summary, this is a category-level demand rotation week: seasonal normalization and event-driven spikes explain most movement, "
        "while core SEO efficiency remains stable. Current data points to a demand/exposure effect first, so technical escalation should be secondary unless efficiency weakens."
    )

    lines.append(
        (
            "**WoW diagnosis**: **clicks changed by "
            f"{_fmt_signed_compact(current.clicks - previous.clicks)} ({wow_pct})**, with CTR {ctr_wow_pp:+.2f} pp and avg position {pos_wow_delta:+.2f}. "
            "This pattern indicates category-level demand rotation, not broad quality collapse. "
            + (f"Growth clusters: {gain_text}. " if gain_text else "")
            + (f"Decline clusters: {loss_text}. " if loss_text else "")
            + (f"Window: {period_label}." if period_label else "")
        ).strip()
    )

    yoy_line = (
        f"**YoY diagnosis**: **clicks are {_fmt_signed_compact(current.clicks - yoy.clicks)} ({yoy_pct})**. "
        f"Winter cluster contribution: {_fmt_signed_compact(winter_yoy_delta)}; event cluster contribution: {_fmt_signed_compact(event_yoy_delta)}. "
    )
    if total_yoy_delta < 0 and non_brand_yoy_delta > 0:
        yoy_line += (
            f"Other non-brand demand adds {_fmt_signed_compact(non_brand_yoy_delta)} and **offsets {abs(non_brand_share):.1f}% of total YoY decline**."
        )
    else:
        yoy_line += (
            f"Other non-brand demand contributes {_fmt_signed_compact(non_brand_yoy_delta)} "
            f"({non_brand_share:+.1f}% of total YoY click delta)."
        )
    lines.append(yoy_line.strip())
    brand_proxy = _brand_proxy_from_gsc(segment_diagnostics)
    if brand_enabled and brand_proxy:
        lines.append(
            "**Brand demand context (Google Trends + GSC)**: Google Trends branded-search interest is "
            f"{brand_wow:+.2f}% WoW and {brand_yoy:+.2f}% YoY; "
            f"GSC brand clicks are {brand_proxy.get('delta_pct_vs_previous', 0.0):+.2f}% WoW "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_previous', 0.0))}) and "
            f"{brand_proxy.get('delta_pct_vs_yoy', 0.0):+.2f}% YoY "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_yoy', 0.0))}); "
            f"GSC brand impressions are {brand_proxy.get('impressions_delta_pct_vs_previous', 0.0):+.2f}% WoW "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_previous', 0.0))}) and "
            f"{brand_proxy.get('impressions_delta_pct_vs_yoy', 0.0):+.2f}% YoY "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_yoy', 0.0))})."
        )
    elif brand_enabled:
        lines.append(
            "**Brand demand context**: Google Trends branded-search interest is "
            f"{brand_wow:+.2f}% WoW and {brand_yoy:+.2f}% YoY, "
            "and GSC brand-proxy segment is unavailable in this run."
        )
    elif brand_proxy:
        lines.append(
            "**Brand demand context (GSC proxy)**: Google Trends brand series is unavailable in this run, "
            f"so we use GSC brand-query proxy: WoW {brand_proxy.get('delta_pct_vs_previous', 0.0):+.2f}% "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_previous', 0.0))}), "
            f"impressions WoW {brand_proxy.get('impressions_delta_pct_vs_previous', 0.0):+.2f}% "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_previous', 0.0))}), "
            f"YoY {brand_proxy.get('delta_pct_vs_yoy', 0.0):+.2f}% "
            f"({_fmt_signed_compact(brand_proxy.get('delta_vs_yoy', 0.0))}), "
            f"impressions YoY {brand_proxy.get('impressions_delta_pct_vs_yoy', 0.0):+.2f}% "
            f"({_fmt_signed_compact(brand_proxy.get('impressions_delta_vs_yoy', 0.0))})."
        )
    else:
        lines.append(
            "**Brand demand context**: unavailable in this run (Google Trends brand time-series missing), "
            "and no GSC brand-proxy segment was available."
        )

    country_code = str((additional_context or {}).get("country_code", "")).strip().upper()
    brand_ctr_proxy = _brand_ctr_proxy_from_gsc(segment_diagnostics)
    if (
        country_code in {"CZ", "SK"}
        and _has_allegro_days_like_signal(external_signals, country_code=country_code)
        and isinstance(brand_ctr_proxy, dict)
        and float(brand_ctr_proxy.get("delta_pp_vs_previous", 0.0)) <= -0.10
    ):
        lines.append(
            "**Brand CTR during Allegro Days (CZ/SK)**: brand CTR fell "
            f"{float(brand_ctr_proxy.get('delta_pp_vs_previous', 0.0)):+.2f} pp WoW "
            f"(YoY {float(brand_ctr_proxy.get('delta_pp_vs_yoy', 0.0)):+.2f} pp) while campaign signals were active. "
            "Interpretation: part of brand demand may be reallocated to paid brand ads in SERP, so organic brand CTR softening is not automatically a technical SEO issue."
        )
    brand_ads_hyp = _brand_ads_hypothesis(
        segment_diagnostics=segment_diagnostics,
        additional_context=additional_context,
        external_signals=external_signals,
    )
    if isinstance(brand_ads_hyp, dict):
        lines.append(
            "**Brand Ads overlap hypothesis (WoW)**: "
            f"{str(brand_ads_hyp.get('statement', '')).strip()} "
            f"Evidence: {str(brand_ads_hyp.get('evidence', '')).strip()}. "
            f"Confidence: {int(brand_ads_hyp.get('confidence', 0) or 0)}/100."
        )

    trend_summary = _build_product_trend_summary(
        scope_results=scope_results,
        external_signals=external_signals,
        additional_context=additional_context,
    )
    concise_trend = trend_summary.get("executive_line", "").strip()
    if concise_trend:
        lines.append("Non-brand trend conclusion: " + concise_trend)

    causal_chain = (
        "**Causal chain**: Observation -> traffic is down while efficiency is stable. "
        "Evidence -> cluster deltas and stable CTR/position indicate a theme-level demand rotation. "
        "Hypothesis -> seasonality/event timing plus paid/campaign timing explain most movement. "
        "Decision -> treat current decline as demand/exposure-led first and quantify paid/campaign incremental effect before escalating technical SEO."
    )
    if page_name_line:
        causal_chain += f" {page_name_line}"
    if brand_note:
        causal_chain += f" {brand_note}"
    if template_conf is not None:
        causal_chain += f" Confidence (template/routing): {template_conf}/100."
    lines.append(causal_chain)

    update_signal = _latest_google_update_signal(external_signals)
    if update_signal:
        update_line = (
            f"Algorithm context: Google update published on {update_signal.day.isoformat()} (`{update_signal.title}`). "
            "Treat as a contributing context signal and validate only where cluster/page behavior changed after the update date."
        )
        if algorithm_conf is not None:
            update_line += f" Confidence: {algorithm_conf}/100."
        lines.append(update_line)

    ddg_context = (additional_context or {}).get("duckduckgo_context", {})
    if isinstance(ddg_context, dict) and ddg_context.get("enabled"):
        ddg_rows = ddg_context.get("rows", [])
        if isinstance(ddg_rows, list) and ddg_rows:
            headings: list[str] = []
            for row in ddg_rows[:2]:
                if not isinstance(row, dict):
                    continue
                heading = str(row.get("heading", "")).strip() or str(row.get("query", "")).strip()
                if heading:
                    headings.append(f"`{heading}`")
            if headings:
                lines.append(
                    "DuckDuckGo context scan: external cause-oriented signals were found and should be used as supporting hypotheses, "
                    "not as standalone evidence. Top hints: " + "; ".join(headings) + "."
                )

    campaign_context = _campaign_event_context(external_signals=external_signals, query_scope=query_scope)
    allegro_campaign_events = campaign_context.get("allegro_events", [])
    competitor_campaign_events = campaign_context.get("competitor_events", [])
    campaign_query_events = campaign_context.get("query_events", [])
    allegro_count = len(allegro_campaign_events) if isinstance(allegro_campaign_events, list) else 0
    competitor_count = len(competitor_campaign_events) if isinstance(competitor_campaign_events, list) else 0
    query_count = len(campaign_query_events) if isinstance(campaign_query_events, list) else 0
    if allegro_count >= 3 or competitor_count > 0 or query_count > 0:
        campaign_days: list[date] = []
        if isinstance(allegro_campaign_events, list):
            campaign_days.extend(row.day for row in allegro_campaign_events if isinstance(row, ExternalSignal))
        if isinstance(competitor_campaign_events, list):
            campaign_days.extend(
                row[0].day
                for row in competitor_campaign_events
                if isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], ExternalSignal)
            )
        period = ""
        if campaign_days:
            period = f" Window: {min(campaign_days).isoformat()} to {max(campaign_days).isoformat()}."
        lines.append(
            "Campaign context: active campaign signals exist and can influence weekly demand allocation. "
            f"(Allegro={_fmt_int(allegro_count)}, competitors={_fmt_int(competitor_count)}, query movers={_fmt_int(query_count)}).{period}"
        )

    pulse_line = _top_platform_pulse_line(additional_context)
    if pulse_line:
        lines.append(pulse_line)

    trade_plan_signal = _trade_plan_signal(additional_context)
    trade_plan = (additional_context or {}).get("trade_plan", {})
    if isinstance(trade_plan, dict) and trade_plan.get("enabled"):
        if isinstance(trade_plan_signal, dict):
            summary = str(trade_plan_signal.get("summary", "")).strip()
            statement = str(trade_plan_signal.get("statement", "")).strip()
            confidence = int(trade_plan_signal.get("confidence", 0) or 0)
            if summary:
                lines.append("Trade-plan summary: " + summary)
            if statement:
                lines.append(
                    "Trade-plan hypothesis: "
                    + statement
                    + f" Confidence: {confidence}/100."
                )
        channel_split = trade_plan.get("channel_split", [])
        if isinstance(channel_split, list) and channel_split:
            top_channel = next((row for row in channel_split if isinstance(row, dict)), None)
            if isinstance(top_channel, dict):
                channel_name = str(top_channel.get("channel", "")).strip() or "channel"
                delta_pct = top_channel.get("delta_spend_pct")
                if (
                    channel_name
                    and channel_name.strip() not in {"-", "Unknown channel"}
                    and isinstance(delta_pct, (int, float))
                ):
                    lines.append(
                        "Channel pressure signal: "
                        f"strongest shift in **{channel_name}** ({float(delta_pct):+.2f}% planned spend vs previous week)."
                    )
    elif isinstance(trade_plan, dict):
        errors = trade_plan.get("errors", [])
        if isinstance(errors, list) and errors:
            lines.append("Trade plan context unavailable in this run: " + str(errors[0]).strip())
        elif trade_plan:
            lines.append(
                "Trade plan context unavailable in this run: no rows matched current/previous weekly windows."
            )

    feature_split = (additional_context or {}).get("gsc_feature_split", {})
    if isinstance(feature_split, dict) and feature_split.get("enabled"):
        rows = feature_split.get("rows", [])
        if isinstance(rows, list) and rows:
            movers = [row for row in rows if isinstance(row, dict)]
            gains, losses = _top_unique_feature_movers(movers, limit=2)
            if gains or losses:
                gain_text = "; ".join(
                    f"{str(row.get('feature', '')).strip()} ({_fmt_signed_compact(row.get('delta_clicks_vs_previous', 0.0))})"
                    for row in gains
                    if str(row.get("feature", "")).strip()
                )
                loss_text = "; ".join(
                    f"{str(row.get('feature', '')).strip()} ({_fmt_signed_compact(row.get('delta_clicks_vs_previous', 0.0))})"
                    for row in losses
                    if str(row.get("feature", "")).strip()
                )
                lines.append(
                    "GSC feature split indicates distribution changes across SERP features rather than one uniform drop. "
                    + (f"Feature gains: {gain_text}. " if gain_text else "")
                    + (f"Feature declines: {loss_text}. " if loss_text else "")
                    + "Use this to verify whether clicks moved between feature types."
                )

    weekly_news = (additional_context or {}).get("weekly_news_digest", {})
    if isinstance(weekly_news, dict) and weekly_news.get("enabled"):
        seo_count = int(weekly_news.get("seo_count", 0) or 0)
        geo_count = int(weekly_news.get("geo_count", 0) or 0)
        lines.append(
            "Weekly SEO/GEO context: external publications were reviewed for potential causal support "
            f"(SEO items={_fmt_int(seo_count)}, GEO items={_fmt_int(geo_count)}). "
            "Interpretation uses this only as secondary evidence."
        )

    temp_diff = float(weather_summary.get("avg_temp_diff_c", 0.0))
    precip_change = float(weather_summary.get("precip_change_pct", 0.0))
    lines.append(
        "Weather context: week-over-week weather moved "
        f"({temp_diff:+.1f}C temperature delta; {precip_change:+.1f}% precipitation delta), "
        "which supports demand-timing interpretation, not direct technical SEO impact."
        + (f" Confidence: {weather_conf}/100." if weather_conf is not None else "")
    )

    macro = (additional_context or {}).get("macro", {})
    if isinstance(macro, dict):
        nbp = macro.get("nbp_fx", {})
        if isinstance(nbp, dict):
            eur = nbp.get("eur_pln", {})
            usd = nbp.get("usd_pln", {})
            if isinstance(eur, dict) and isinstance(usd, dict):
                eur_delta = float(eur.get("delta_pct_vs_previous", 0.0) or 0.0)
                usd_delta = float(usd.get("delta_pct_vs_previous", 0.0) or 0.0)
                if abs(eur_delta) >= 1.0 or abs(usd_delta) >= 1.0:
                    lines.append(
                        "Macro context: FX moved materially vs previous week "
                        f"(EUR/PLN {eur_delta:+.2f}%, USD/PLN {usd_delta:+.2f}%). "
                        "This can affect price competitiveness and conversion in import-heavy categories."
                    )

    forecast_start = str(weather_summary.get("forecast_start", "")).strip()
    forecast_end = str(weather_summary.get("forecast_end", "")).strip()
    if forecast_start and forecast_end:
        lines.append(
            "**Forward 7d**: weather forecast "
            f"{forecast_start} to {forecast_end} (avg {float(weather_summary.get('forecast_avg_temp_c', 0.0)):+.1f}C, "
            f"precip {float(weather_summary.get('forecast_precip_mm', 0.0)):.1f}mm). "
            "Use this as timing context for near-term category demand."
        )

    confirmed_points: list[str] = []
    hypothesis_points: list[str] = []
    confirmed_points.append(
        f"GSC clicks {_fmt_signed_compact(current.clicks - previous.clicks)} WoW ({wow_pct}) with near-stable CTR ({ctr_wow_pp:+.2f} pp)."
    )
    if page_name_line:
        confirmed_points.append(page_name_line)
    if isinstance(brand_proxy, dict):
        confirmed_points.append(
            "Brand in GSC: clicks "
            f"{float(brand_proxy.get('delta_pct_vs_previous', 0.0)):+.2f}% WoW, "
            f"{float(brand_proxy.get('delta_pct_vs_yoy', 0.0)):+.2f}% YoY."
        )
    if brand_note:
        hypothesis_points.append(brand_note)
    if isinstance(brand_ads_hyp, dict):
        hypothesis_points.append(
            str(brand_ads_hyp.get("statement", "")).strip()
            + f" (confidence {int(brand_ads_hyp.get('confidence', 0) or 0)}/100)."
        )
    if update_signal:
        hypothesis_points.append(
            f"Post-update SERP behavior check after {update_signal.day.isoformat()} is required before root-cause lock."
        )
    if isinstance(trade_plan_signal, dict):
        hypothesis_points.append(
            "Paid-channel timing may reallocate brand/category clicks between organic and ads."
        )

    if confirmed_points or hypothesis_points:
        lines.append("")
        lines.append("**Confirmed vs hypothesis**")
        if confirmed_points:
            lines.append("Confirmed by data:")
            for item in confirmed_points[:3]:
                lines.append(f"- {item}")
        if hypothesis_points:
            lines.append("Working hypotheses:")
            for item in hypothesis_points[:3]:
                lines.append(f"- {item}")

    # Dynamic driver table (only active/contextual drivers for this run).
    driver_rows: list[tuple[str, str, str, str, str]] = []
    demand_direction = "Down" if (current.clicks - previous.clicks) < 0 else "Up"
    demand_conf = seasonality_conf if seasonality_conf is not None else 70
    demand_impact = "High" if abs(_ratio_delta(current.clicks, previous.clicks)) >= 0.03 else "Medium"
    driver_rows.append(
        (
            "Demand mix",
            demand_direction,
            f"Organic clicks {wow_pct} WoW; efficiency change limited (CTR {ctr_wow_pp:+.2f} pp).",
            f"{demand_conf}/100",
            demand_impact,
        )
    )
    if page_name_line:
        driver_rows.append(
            (
                "Page routing",
                "Down" if home_wow < 0 else "Up",
                page_name_line,
                f"{template_conf if template_conf is not None else 72}/100",
                "High",
            )
        )
    if isinstance(brand_ads_hyp, dict):
        driver_rows.append(
            (
                "Brand paid overlap",
                "Mixed",
                str(brand_ads_hyp.get("statement", "")).strip(),
                f"{int(brand_ads_hyp.get('confidence', 0) or 0)}/100",
                "Medium",
            )
        )
    if isinstance(trade_plan_signal, dict):
        driver_rows.append(
            (
                "Trade plan timing",
                "Mixed",
                str(trade_plan_signal.get("statement", "")).strip(),
                f"{int(trade_plan_signal.get('confidence', 0) or 0)}/100",
                "Medium",
            )
        )
    if update_signal:
        driver_rows.append(
            (
                "Algorithm context",
                "Mixed",
                f"Post-update monitoring needed after {update_signal.day.isoformat()}.",
                f"{algorithm_conf if algorithm_conf is not None else 68}/100",
                "Medium",
            )
        )
    if abs(temp_diff) >= 2.0 or abs(precip_change) >= 20.0:
        driver_rows.append(
            (
                "Weather timing",
                "Mixed",
                f"Weather shift vs previous week: temp {temp_diff:+.1f}C, precip {precip_change:+.1f}%.",
                f"{weather_conf if weather_conf is not None else 64}/100",
                "Low-Med",
            )
        )

    lines.append("")
    lines.append("**Driver Scoreboard**")
    lines.append("| Driver | Direction | Evidence | Confidence | Expected impact on traffic |")
    lines.append("|---|---|---|---:|---|")
    for row in driver_rows[:6]:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

    senuto_intelligence = (additional_context or {}).get("senuto_intelligence", {})
    if isinstance(senuto_intelligence, dict) and senuto_intelligence.get("enabled"):
        competitors = senuto_intelligence.get("competitors_overview", [])
        if isinstance(competitors, list) and competitors:
            top_comp = next((row for row in competitors if isinstance(row, dict)), None)
            if isinstance(top_comp, dict):
                lines.append(
                    "Competitive context: Senuto overlap shows strongest competitor pressure from "
                    f"`{top_comp.get('domain', '')}` ({_fmt_int(top_comp.get('common_keywords', 0.0))} common keywords)."
                )

    return lines


def _focus_terms_from_query_scope(query_scope: AnalysisResult | None) -> list[str]:
    if query_scope is None:
        return []

    stop_words = {
        "allegro",
        "allegropl",
        "listing",
        "kategoria",
        "oferta",
        "www",
        "https",
        "query",
        "smart",
        "strona",
        "glowna",
    }

    counts: dict[str, int] = {}
    for row in query_scope.top_losers + query_scope.top_winners:
        text = _normalize_text(row.key)
        for token in re.findall(r"[a-z0-9]{4,}", text):
            if token in stop_words:
                continue
            if token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [token for token, _ in ranked[:20]]


def _collect_context_notes(rows: list[dict[str, object]], fallback_key: str = "excerpt") -> list[str]:
    notes: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        highlights = row.get("highlights", [])
        if isinstance(highlights, list):
            for note in highlights:
                text = str(note).strip()
                if text:
                    notes.append(text)
        if not highlights:
            fallback = str(row.get(fallback_key, "")).strip()
            if fallback:
                notes.append(fallback)
    return notes


def _find_overlap_terms(texts: list[str], terms: list[str]) -> list[str]:
    if not texts or not terms:
        return []
    blob = " ".join(_normalize_text(text) for text in texts if text)
    overlaps = [term for term in terms if term in blob]
    return overlaps[:6]


def _build_reasoning_hypotheses(
    totals: dict[str, MetricSummary],
    scope_results: list[tuple[str, AnalysisResult]],
    external_signals: list[ExternalSignal],
    weather_summary: dict[str, float],
    ferie_context: dict[str, object],
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
    additional_context: dict[str, object] | None,
    senuto_summary: dict[str, float] | None,
    senuto_error: str | None,
) -> list[dict[str, object]]:
    current = totals["current_28d"]
    previous = totals["previous_28d"]
    hypotheses: list[dict[str, object]] = []
    query_scope = _find_scope(scope_results, "query")

    if query_scope:
        winter_terms = (
            "fajerwerk",
            "petard",
            "sanki",
            "odsniez",
            "snieg",
            "kalendarz 2026",
            "dzien babci",
            "dziadka",
        )
        event_terms = (
            "wosp",
            "wieniawa",
            "nocowanka",
            "licytac",
            "walentyn",
        )
        winter_drop = _sum_losses(query_scope.top_losers, winter_terms)
        event_gain = _sum_gains(query_scope.top_winners, event_terms)
        ctr_up = current.ctr >= previous.ctr
        position_better = current.position <= previous.position

        if winter_drop >= 5000:
            score = 86 if (ctr_up and position_better) else 73
            evidence = [
                f"Drop in winter-intent query clicks vs previous week: ~{_fmt_int(winter_drop)}.",
                f"CTR current vs prev: {_pct(current.ctr)} vs {_pct(previous.ctr)}.",
                f"Avg position current vs prev: {current.position:.2f} vs {previous.position:.2f}.",
            ]
            hypotheses.append(
                {
                    "category": "Seasonality",
                    "confidence": score,
                    "thesis": (
                        "Declines are concentrated in winter-intent demand clusters (for example sled/firework-type intents), "
                        "while CTR and average position improved. This pattern is consistent with post-peak seasonality "
                        "and does not indicate broad ranking/indexing degradation."
                    ),
                    "evidence": evidence,
                    "owner": "SEO + Merchandising",
                }
            )

        if event_gain >= 3000:
            hypotheses.append(
                {
                    "category": "Events",
                    "confidence": 82,
                    "thesis": "Event-driven demand topics are offsetting part of the natural seasonal declines.",
                    "evidence": [
                        f"Increase in event-query clicks vs previous week: ~{_fmt_int(event_gain)}.",
                        "Top winners include queries tied to current events.",
                    ],
                    "owner": "SEO + Content",
                }
            )

    yoy_ferie = ferie_context.get("yoy_comparison", {})
    ferie_delta_pp = float(yoy_ferie.get("avg_daily_delta_pp", 0.0))
    if abs(ferie_delta_pp) >= 0.3:
        top_region = ""
        yoy_rows = yoy_ferie.get("rows")
        if isinstance(yoy_rows, list) and yoy_rows:
            row = yoy_rows[0]
            if isinstance(row, dict):
                top_region = (
                    f"{row.get('name', 'Region')}: {int(row.get('current_days', 0))} days vs "
                    f"{int(row.get('yoy_days', 0))} days ({float(row.get('contribution_pp', 0.0)):+.2f} pp)."
                )
        hypotheses.append(
            {
                "category": "Seasonality",
                "confidence": 74,
                "thesis": "Regional winter-break distribution changes YoY comparability for winter demand.",
                "evidence": [
                    f"YoY delta in winter-break exposure (avg daily GMV proxy): {ferie_delta_pp:+.2f} pp.",
                    top_region or "Material regional winter-break difference in the YoY window.",
                ],
                "owner": "SEO + BI",
            }
        )

    temp_diff = float(weather_summary.get("avg_temp_diff_c", 0.0))
    if abs(temp_diff) >= 2.0:
        hypotheses.append(
            {
                "category": "Macro/Weather",
                "confidence": 68,
                "thesis": "Weather anomaly may have shifted demand between product categories.",
                "evidence": [
                    f"Avg temperature diff current vs previous: {temp_diff:+.1f}C.",
                    f"Precipitation change current vs previous: {float(weather_summary.get('precip_change_pct', 0.0)):+.1f}%.",
                ],
                "owner": "SEO + Commercial",
            }
        )

    google_updates = [
        signal
        for signal in external_signals
        if signal.source in {"Google Search Status", "Google Search Central Blog"}
        and "update" in _normalize_text(f"{signal.title} {signal.details}")
    ]
    if google_updates:
        latest = max(google_updates, key=lambda row: row.day)
        hypotheses.append(
            {
                "category": "Algorithm",
                "confidence": 71,
                "thesis": "Google algorithm changes may have affected selected query/page clusters.",
                "evidence": [
                    f"Latest update: {latest.day.isoformat()} ({latest.title}).",
                    "Query/page segment monitoring is required after the update date.",
                ],
                "owner": "SEO",
            }
        )

    if senuto_summary:
        avg_delta = float(senuto_summary.get("avg_delta_pct", 0.0))
        if avg_delta < -5:
            hypotheses.append(
                {
                    "category": "SEO Visibility",
                    "confidence": 79,
                    "thesis": "Senuto visibility decline suggests a ranking component, not only seasonality.",
                    "evidence": [
                        f"Senuto avg delta vs previous week: {avg_delta:+.2f}%.",
                        f"Senuto latest delta vs previous week: {float(senuto_summary.get('latest_delta_pct', 0.0)):+.2f}%.",
                    ],
                    "owner": "SEO",
                }
            )
    elif senuto_error:
        hypotheses.append(
            {
                "category": "Data quality",
                "confidence": 92,
                "thesis": "Missing Senuto data lowers confidence in ranking diagnosis.",
                "evidence": [f"Senuto error: {senuto_error}"],
                "owner": "SEO Ops",
            }
        )

    if segment_diagnostics:
        non_brand = _top_negative_segment(segment_diagnostics.get("brand_non_brand"), min_clicks=3000.0)
        if non_brand and str(non_brand.get("segment")) == "non_brand":
            hypotheses.append(
                {
                    "category": "Demand mix",
                    "confidence": 76,
                    "thesis": "Non-brand decline is larger than brand decline, which may indicate competitive pressure beyond branded traffic.",
                    "evidence": [
                        f"Non-brand delta vs prev: {_fmt_signed_int(non_brand.get('delta_vs_previous', 0.0))} ({_signed_pct(float(non_brand.get('delta_pct_vs_previous', 0.0)))}).",
                    ],
                    "owner": "SEO + Acquisition",
                }
            )

        device_loser = _top_negative_segment(segment_diagnostics.get("device"), min_clicks=3000.0)
        if device_loser:
            hypotheses.append(
                {
                    "category": "Technical/UX",
                    "confidence": 67,
                    "thesis": "One device segment drives the decline more than others and needs targeted UX/SEO audit.",
                    "evidence": [
                        f"Device {device_loser.get('segment')}: delta vs prev {_fmt_signed_int(device_loser.get('delta_vs_previous', 0.0))} ({_signed_pct(float(device_loser.get('delta_pct_vs_previous', 0.0)))}).",
                    ],
                    "owner": "SEO + Web Performance",
                }
            )

        template_loser = _top_negative_segment(segment_diagnostics.get("page_template"), min_clicks=3000.0)
        if template_loser:
            hypotheses.append(
                {
                    "category": "Page Name performance",
                    "confidence": 72,
                    "thesis": "Decline concentrated in one page-name segment suggests page-level issue or section-specific demand drop.",
                    "evidence": [
                        f"Page Name segment {template_loser.get('segment')}: delta vs prev {_fmt_signed_int(template_loser.get('delta_vs_previous', 0.0))} ({_signed_pct(float(template_loser.get('delta_pct_vs_previous', 0.0)))}).",
                    ],
                    "owner": "SEO + Product",
                }
            )

    if additional_context:
        focus_terms = _focus_terms_from_query_scope(query_scope)
        pagespeed = additional_context.get("pagespeed", {})
        if isinstance(pagespeed, dict):
            mobile = pagespeed.get("mobile", {})
            if isinstance(mobile, dict):
                lcp = float(mobile.get("lcp_ms", 0.0))
                inp = float(mobile.get("inp_ms", 0.0))
                cls = float(mobile.get("cls", 0.0))
                if lcp >= 2500 or inp >= 200 or cls >= 0.1:
                    hypotheses.append(
                        {
                            "category": "Technical/UX",
                            "confidence": 66,
                            "thesis": "Mobile CWV are near/below \"good\" thresholds and can depress organic performance.",
                            "evidence": [
                                f"PageSpeed mobile LCP={lcp:.0f}ms, INP={inp:.0f}ms, CLS={cls:.2f}.",
                            ],
                            "owner": "Web Performance",
                        }
                    )

        macro = additional_context.get("macro", {})
        if isinstance(macro, dict):
            nbp = macro.get("nbp_fx", {})
            if isinstance(nbp, dict):
                for code_key, label in (("eur_pln", "EUR/PLN"), ("usd_pln", "USD/PLN")):
                    row = nbp.get(code_key)
                    if isinstance(row, dict):
                        delta = float(row.get("delta_pct_vs_previous", 0.0))
                        if abs(delta) >= 2.0:
                            hypotheses.append(
                                {
                                    "category": "Macro",
                                    "confidence": 63,
                                    "thesis": "FX changes can affect pricing/offers and demand in selected categories.",
                                    "evidence": [
                                        f"{label} avg delta vs previous week: {delta:+.2f}%.",
                                    ],
                                    "owner": "Commercial + Pricing",
                                }
                            )
                            break

        product_trends = additional_context.get("product_trends", {})
        if isinstance(product_trends, dict):
            trend_rows = product_trends.get("top_yoy_non_brand", [])
            fallback_source = False
            if not (isinstance(trend_rows, list) and trend_rows):
                trend_rows = _fallback_non_brand_yoy_from_gsc(
                    query_scope=query_scope,
                    top_rows=10,
                )
                fallback_source = bool(trend_rows)
            if isinstance(trend_rows, list) and trend_rows:
                current_sum = 0.0
                previous_sum = 0.0
                for row in trend_rows:
                    if not isinstance(row, dict):
                        continue
                    current_sum += float(row.get("current_value", 0.0))
                    previous_sum += float(row.get("previous_value", 0.0))
                delta = current_sum - previous_sum
                delta_pct = (delta / previous_sum * 100.0) if previous_sum else 0.0

                lead = next((row for row in trend_rows if isinstance(row, dict)), None)
                lead_text = ""
                if isinstance(lead, dict):
                    lead_text = (
                        f"Top trend: {lead.get('trend', '')} "
                        f"({_fmt_int(lead.get('current_value', 0.0))} vs {_fmt_int(lead.get('previous_value', 0.0))})."
                    )
                confidence = 74 if abs(delta_pct) >= 10 else 66
                direction = "upward" if delta >= 0 else "downward"
                hypotheses.append(
                    {
                        "category": "Non-brand product trends",
                        "confidence": confidence,
                        "thesis": (
                            "Non-brand product trend sheets indicate "
                            f"{direction} demand pressure vs previous year, which should be considered in SEO diagnosis."
                        ),
                        "evidence": [
                            f"Top non-brand trends sum: current={_fmt_int(current_sum)}, previous={_fmt_int(previous_sum)}, delta={_fmt_signed_int(delta)} ({delta_pct:+.2f}%).",
                            "Source: GSC query fallback (sheet YoY unavailable)." if fallback_source else "Source: product trend sheets.",
                            lead_text or "Top non-brand trend rows were detected in the configured sheets.",
                        ],
                        "owner": "SEO + Category/Merchandising",
                    }
                )

        senuto_intelligence = additional_context.get("senuto_intelligence", {})
        if isinstance(senuto_intelligence, dict) and senuto_intelligence.get("enabled"):
            competitors = senuto_intelligence.get("competitors_overview", [])
            if isinstance(competitors, list) and competitors:
                top_comp = next((row for row in competitors if isinstance(row, dict)), None)
                if isinstance(top_comp, dict):
                    hypotheses.append(
                        {
                            "category": "Competitive pressure",
                            "confidence": 67,
                            "thesis": (
                                "Competitor overlap indicates elevated competitive pressure in shared non-brand SERP space."
                            ),
                            "evidence": [
                                f"Top competitor overlap domain: {top_comp.get('domain', '')}.",
                                f"Common keywords: {_fmt_int(top_comp.get('common_keywords', 0.0))}.",
                                f"Top10 keywords (competitor): {_fmt_int(top_comp.get('top10_current', 0.0))}.",
                            ],
                            "owner": "SEO + Category Owners",
                        }
                    )

            direct_answers = senuto_intelligence.get("direct_answers", [])
            if isinstance(direct_answers, list) and direct_answers:
                hypotheses.append(
                    {
                        "category": "SERP features",
                        "confidence": 64,
                        "thesis": (
                            "Direct-answer SERP features are present for tracked queries and may reduce classic blue-link CTR."
                        ),
                        "evidence": [
                            f"Direct-answer rows in Senuto snapshot: {_fmt_int(len(direct_answers))}.",
                        ],
                        "owner": "SEO",
                    }
                )

            seasonality = senuto_intelligence.get("seasonality", {})
            if isinstance(seasonality, dict):
                trend_values = seasonality.get("trend_values", [])
                if isinstance(trend_values, list) and trend_values:
                    month_idx = date.today().month - 1
                    current_idx = (
                        float(trend_values[month_idx])
                        if 0 <= month_idx < len(trend_values)
                        else 0.0
                    )
                    hypotheses.append(
                        {
                            "category": "Seasonality",
                            "confidence": 73,
                            "thesis": (
                                "Senuto monthly seasonality supports demand-timing interpretation and should be read together with weekly weather shifts."
                            ),
                            "evidence": [
                                f"Senuto current-month seasonality index: {current_idx:.2f}.",
                                f"Senuto peak month/value: M{int(seasonality.get('peak_month', 0) or 0)} / {float(seasonality.get('peak_value', 0.0) or 0.0):.2f}.",
                                f"Senuto low month/value: M{int(seasonality.get('low_month', 0) or 0)} / {float(seasonality.get('low_value', 0.0) or 0.0):.2f}.",
                            ],
                            "owner": "SEO + BI",
                        }
                    )

        seo_presentations = additional_context.get("seo_presentations", {})
        if isinstance(seo_presentations, dict) and seo_presentations.get("enabled"):
            highlights = seo_presentations.get("highlights", [])
            if isinstance(highlights, list) and highlights:
                top_notes = [
                    str(row.get("note", "")).strip()
                    for row in highlights[:2]
                    if isinstance(row, dict) and str(row.get("note", "")).strip()
                ]
                years_covered = [
                    str(row.get("year", "")).strip()
                    for row in seo_presentations.get("years", [])
                    if isinstance(row, dict) and str(row.get("year", "")).strip()
                ]
                years_label = ", ".join(years_covered[:2]) if years_covered else "current and previous year"
                hypotheses.append(
                    {
                        "category": "Internal initiatives",
                        "confidence": 61,
                        "thesis": "Recent SEO team presentation topics may explain part of the observed movement in monitored segments.",
                        "evidence": [
                            f"Presentation archive analyzed for years: {years_label}.",
                            top_notes[0] if top_notes else "Highlights extracted from team presentations.",
                        ],
                        "owner": "SEO Team",
                    }
                )

        historical_reports = additional_context.get("historical_reports", {})
        if isinstance(historical_reports, dict) and historical_reports.get("enabled"):
            recent_reports = historical_reports.get("recent_reports", [])
            recent_notes = []
            if isinstance(recent_reports, list):
                recent_notes = _collect_context_notes(recent_reports)
            yoy_report = historical_reports.get("yoy_report", {})
            if isinstance(yoy_report, dict):
                recent_notes.extend(
                    _collect_context_notes([yoy_report], fallback_key="excerpt")
                )

            overlaps = _find_overlap_terms(recent_notes, focus_terms)
            if recent_notes:
                thesis = (
                    "Current query/page shifts are consistent with themes highlighted in prior weekly reports."
                    if overlaps
                    else "Historical reports add context, but overlap with current movers is limited."
                )
                confidence = 72 if overlaps else 58
                evidence = [
                    f"Recent reports analyzed: {len(recent_reports)}.",
                ]
                if isinstance(yoy_report, dict) and yoy_report.get("id"):
                    evidence.append(
                        f"YoY report loaded: {str(yoy_report.get('date', '')).strip() or 'available'}."
                    )
                if overlaps:
                    evidence.append(
                        f"Shared terms between history and current movers: {', '.join(overlaps[:5])}."
                    )
                hypotheses.append(
                    {
                        "category": "Continuity",
                        "confidence": confidence,
                        "thesis": thesis,
                        "evidence": evidence,
                        "owner": "SEO Team",
                    }
                )

        status_log = additional_context.get("status_log", {})
        if isinstance(status_log, dict) and status_log.get("enabled"):
            status_entries = status_log.get("entries", [])
            notes: list[str] = []
            latest_date = ""
            if isinstance(status_entries, list):
                for row in status_entries[:8]:
                    if not isinstance(row, dict):
                        continue
                    topic = str(row.get("topic", "")).strip()
                    summary = str(row.get("summary", "")).strip()
                    if topic:
                        notes.append(topic)
                    if summary:
                        notes.append(summary)
                    if not latest_date:
                        latest_date = str(row.get("date", "")).strip()
            overlaps = _find_overlap_terms(notes, focus_terms)
            if notes:
                confidence = 70 if overlaps else 55
                thesis = (
                    "Recently discussed status topics align with this week's winners/losers."
                    if overlaps
                    else "Status updates are available, but direct linkage to current movers is weak."
                )
                evidence = [
                    f"Status entries analyzed: {len(status_entries) if isinstance(status_entries, list) else 0}.",
                ]
                if latest_date:
                    evidence.append(f"Latest status date in source: {latest_date}.")
                if overlaps:
                    evidence.append(f"Overlap terms: {', '.join(overlaps[:5])}.")
                hypotheses.append(
                    {
                        "category": "Execution continuity",
                        "confidence": confidence,
                        "thesis": thesis,
                        "evidence": evidence,
                        "owner": "SEO Team + PM",
                    }
                )

        campaign_context = _campaign_event_context(
            external_signals=external_signals,
            query_scope=query_scope,
        )
        allegro_events = campaign_context.get("allegro_events", [])
        competitor_events = campaign_context.get("competitor_events", [])
        query_events = campaign_context.get("query_events", [])
        allegro_count = len(allegro_events) if isinstance(allegro_events, list) else 0
        competitor_count = len(competitor_events) if isinstance(competitor_events, list) else 0
        query_count = len(query_events) if isinstance(query_events, list) else 0
        if allegro_count or competitor_count or query_count:
            evidence: list[str] = [
                f"Campaign signals: allegro={allegro_count}, competitors={competitor_count}, query_movers={query_count}.",
            ]
            if isinstance(allegro_events, list) and allegro_events:
                first = allegro_events[0]
                evidence.append(
                    f"Latest Allegro campaign mention: {first.day.isoformat()} | {first.title}."
                )
            if isinstance(competitor_events, list) and competitor_events:
                first_signal, first_competitor = competitor_events[0]
                evidence.append(
                    f"Latest competitor campaign mention: {first_signal.day.isoformat()} | {first_competitor} | {first_signal.title}."
                )
            if isinstance(query_events, list) and query_events:
                top = query_events[0]
                evidence.append(
                    f"Top campaign query mover: {top.key} ({_fmt_signed_int(top.click_delta_vs_previous)} vs prev)."
                )
            confidence = 76 if (allegro_count and competitor_count) else 69
            hypotheses.append(
                {
                    "category": "Campaign events",
                    "confidence": confidence,
                    "thesis": (
                        "Observed changes may be influenced by marketplace campaign periods on Allegro and/or competitors, "
                        "not only by technical SEO factors."
                    ),
                    "evidence": evidence,
                    "owner": "SEO + Commercial",
                }
            )

    if not hypotheses:
        hypotheses.append(
            {
                "category": "Unknown",
                "confidence": 40,
                "thesis": "No strong signals for a single dominant thesis; deeper segmentation and monitoring required.",
                "evidence": ["No dominant signals above heuristic thresholds."],
                "owner": "SEO",
            }
        )

    hypotheses.sort(key=lambda row: int(row.get("confidence", 0)), reverse=True)
    return hypotheses


def _build_integrated_reasoning(
    totals: dict[str, MetricSummary],
    scope_results: list[tuple[str, AnalysisResult]],
    external_signals: list[ExternalSignal],
    weather_summary: dict[str, float],
    ferie_context: dict[str, object],
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None,
    additional_context: dict[str, object] | None,
    senuto_summary: dict[str, float] | None,
    senuto_error: str | None,
) -> list[str]:
    precomputed = (additional_context or {}).get("precomputed_hypotheses", [])
    if isinstance(precomputed, list) and precomputed:
        hypotheses = [row for row in precomputed if isinstance(row, dict)]
    else:
        hypotheses = _build_reasoning_hypotheses(
            totals=totals,
            scope_results=scope_results,
            external_signals=external_signals,
            weather_summary=weather_summary,
            ferie_context=ferie_context,
            segment_diagnostics=segment_diagnostics,
            additional_context=additional_context,
            senuto_summary=senuto_summary,
            senuto_error=senuto_error,
        )

    lines: list[str] = []
    for row in hypotheses:
        score = int(row.get("confidence", 0))
        category = str(row.get("category", "Unknown"))
        thesis = str(row.get("thesis", ""))
        evidence = row.get("evidence")
        evidence_text = ""
        if isinstance(evidence, list):
            evidence_text = " | ".join(str(item) for item in evidence[:3])
        lines.append(f"- [{score}/100] {category}: {thesis} Evidence: {evidence_text}")
    return lines


def _build_upcoming_trends(
    run_date: date,
    external_signals: list[ExternalSignal],
    ferie_trends: list[tuple[date, str, str, str, str]] | None = None,
    additional_context: dict[str, object] | None = None,
) -> list[tuple[str, str, str, str]]:
    horizon = run_date + timedelta(days=60)
    rows: list[tuple[date, str, str, str, str]] = []

    if ferie_trends:
        rows.extend(ferie_trends)

    for signal in external_signals:
        if signal.source == "Public Holidays" and run_date < signal.day <= horizon:
            rows.append(
                (
                    signal.day,
                    signal.day.isoformat(),
                    signal.title,
                    "Holiday timing can shift purchase intent and weekly traffic distribution.",
                    "Validate campaign calendar and traffic expectations for days before/after the holiday.",
                )
            )

    product_trends = (additional_context or {}).get("product_trends", {})
    if isinstance(product_trends, dict) and product_trends.get("enabled"):
        upcoming_rows = product_trends.get("upcoming_31d", [])
        if isinstance(upcoming_rows, list):
            for row in upcoming_rows[:10]:
                if not isinstance(row, dict):
                    continue
                day_label = str(row.get("date", "")).strip()
                trend_day: date
                if day_label:
                    try:
                        trend_day = date.fromisoformat(day_label[:10])
                    except ValueError:
                        continue
                    if not (run_date < trend_day <= horizon):
                        continue
                else:
                    # next31D sheet can have no explicit date; treat as horizon-level signal
                    trend_day = run_date + timedelta(days=1)
                trend_name = str(row.get("trend", "")).strip()
                if not trend_name:
                    continue
                value = float(row.get("value", 0.0))
                rows.append(
                    (
                        trend_day,
                        trend_day.isoformat(),
                        f"Product trend window: {trend_name}",
                        (
                            f"Upcoming non-brand product trend in next {int(product_trends.get('horizon_days', 31))} days "
                            f"(score/value: {value:.2f})."
                        ),
                        "Prepare category pages/content and campaign timing before the trend window starts.",
                    )
                )

    deduped: list[tuple[date, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: item[0]):
        dedupe_key = (row[1], row[2])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(row)

    return [(label, title, impact, action) for _, label, title, impact, action in deduped[:10]]


def build_markdown_report(
    run_date: date,
    report_country_code: str,
    windows: dict[str, DateWindow],
    totals: dict[str, MetricSummary],
    scope_results: list[tuple[str, AnalysisResult]],
    external_signals: list[ExternalSignal],
    weather_summary: dict[str, float],
    ferie_context: dict[str, object] | None = None,
    upcoming_ferie_trends: list[tuple[date, str, str, str, str]] | None = None,
    segment_diagnostics: dict[str, list[dict[str, float | str]]] | None = None,
    additional_context: dict[str, object] | None = None,
    senuto_summary: dict[str, float] | None = None,
    senuto_error: str | None = None,
    query_filter_stats: dict[str, dict[str, int]] | None = None,
) -> str:
    if ferie_context is None:
        ferie_context = {
            "source": "OpenHolidays API (SchoolHolidays + Subdivisions)",
            "source_url": "",
            "profiles_ranked": [],
            "window_stats": {},
            "missing_years": [],
            "yoy_comparison": {"avg_daily_delta_pp": 0.0, "rows": []},
            "errors": ["Ferie context was not provided."],
        }

    current = totals["current_28d"]
    previous = totals["previous_28d"]
    yoy = totals["yoy_52w"]

    click_delta_prev = current.clicks - previous.clicks
    click_delta_prev_pct = _ratio_delta(current.clicks, previous.clicks)
    click_delta_yoy = current.clicks - yoy.clicks
    click_delta_yoy_pct = _ratio_delta(current.clicks, yoy.clicks)

    impression_delta_prev = current.impressions - previous.impressions
    impression_delta_prev_pct = _ratio_delta(current.impressions, previous.impressions)

    hypotheses = _build_reasoning_hypotheses(
        totals=totals,
        scope_results=scope_results,
        external_signals=external_signals,
        weather_summary=weather_summary,
        ferie_context=ferie_context,
        segment_diagnostics=segment_diagnostics,
        additional_context=additional_context,
        senuto_summary=senuto_summary,
        senuto_error=senuto_error,
    )

    lines: list[str] = []
    country_label = report_country_code.strip().upper() or "PL"
    lines.append(f"# Weekly SEO Intelligence Report ({run_date.isoformat()} | {country_label})")
    lines.append("")
    lines.append("## Executive summary")
    lines.extend(
        _build_executive_summary_lines(
            totals=totals,
            scope_results=scope_results,
            hypotheses=hypotheses,
            external_signals=external_signals,
            segment_diagnostics=segment_diagnostics,
            additional_context=additional_context,
            senuto_summary=senuto_summary,
            senuto_error=senuto_error,
        )
    )
    lines.append("")
    lines.append("## What is happening and why")
    lines.extend(
        _build_what_is_happening_lines(
            totals=totals,
            windows=windows,
            scope_results=scope_results,
            hypotheses=hypotheses,
            external_signals=external_signals,
            weather_summary=weather_summary,
            segment_diagnostics=segment_diagnostics,
            additional_context=additional_context,
        )
    )

    # Decision-layer report only (Executive summary + narrative).
    if not INCLUDE_APPENDIX_IN_REPORT:
        return "\n".join(lines).strip() + "\n"

    lines.append("")
    lines.append("## Appendix")
    lines.append("Supporting data tables and diagnostics start below.")
    lines.append("")
    lines.append("## Date windows")
    lines.append("| Window | Start | End | Days |")
    lines.append("|---|---:|---:|---:|")
    for key in [
        "current_28d",
        "previous_28d",
        "yoy_52w",
        "current_28d_context",
        "previous_28d_context",
        "yoy_28d_context_52w",
    ]:
        window = windows.get(key)
        if not isinstance(window, DateWindow):
            continue
        lines.append(
            f"| {window.name} | {window.start.isoformat()} | {window.end.isoformat()} | {window.days} |"
        )

    freshness_rows = _source_freshness_rows(additional_context=additional_context)
    if freshness_rows:
        lines.append("")
        lines.append("## Source freshness and fallback status")
        lines.append("| Source | Status | Latest data day | TTL (hours) | Cache mode | Notes |")
        lines.append("|---|---|---|---:|---|---|")
        for row in freshness_rows:
            lines.append(
                f"| {row.get('source', '')} | {str(row.get('status', '')).upper()} | "
                f"{row.get('last_day', '-') or '-'} | {float(row.get('ttl_hours', 0.0)):.1f} | "
                f"{row.get('cache_mode', '-')} | {row.get('note', '-')} |"
            )

    lines.append("")
    lines.append("## KPI summary")
    lines.append("| Metric | Current week | Previous week | YoY (52w) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Clicks | {_fmt_int(current.clicks)} | {_fmt_int(previous.clicks)} | {_fmt_int(yoy.clicks)} |")
    lines.append(
        f"| Impressions | {_fmt_int(current.impressions)} | {_fmt_int(previous.impressions)} | {_fmt_int(yoy.impressions)} |"
    )
    lines.append(f"| CTR | {_pct(current.ctr)} | {_pct(previous.ctr)} | {_pct(yoy.ctr)} |")
    lines.append(
        f"| Avg position | {current.position:.2f} | {previous.position:.2f} | {yoy.position:.2f} |"
    )

    lines.append("")
    lines.append("## Main deltas")
    lines.append("| Metric | WoW | YoY (52w) |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Clicks | {_fmt_signed_int(click_delta_prev)} ({_signed_pct(click_delta_prev_pct)}) | {_fmt_signed_int(click_delta_yoy)} ({_signed_pct(click_delta_yoy_pct)}) |"
    )
    lines.append(
        f"| Impressions | {_fmt_signed_int(impression_delta_prev)} ({_signed_pct(impression_delta_prev_pct)}) | {_fmt_signed_int(current.impressions - yoy.impressions)} ({_signed_pct(_ratio_delta(current.impressions, yoy.impressions))}) |"
    )

    lines.append("")
    lines.append("## Decision one-pager")
    lines.append("| What changed | Why it matters |")
    lines.append("|---|---|")
    for row in _decision_one_pager_rows(hypotheses, limit=6):
        lines.append(f"| {row.get('what_changed', '')} | {row.get('why_it_matters', '')} |")

    impact_rows = _impact_attribution_rows(
        totals=totals,
        scope_results=scope_results,
        segment_diagnostics=segment_diagnostics,
        limit=10,
    )
    lines.append("")
    lines.append("## Impact attribution (share of WoW click movement)")
    lines.append("| Dimension | Segment | Delta vs WoW | Share of total WoW delta |")
    lines.append("|---|---|---:|---:|")
    if impact_rows:
        for row in impact_rows:
            lines.append(
                f"| {row.get('dimension', '')} | {row.get('name', '')} | "
                f"{_fmt_signed_int(row.get('delta', 0.0))} | {float(row.get('share_of_total', 0.0)):+.1f}% |"
            )
    else:
        lines.append("| - | - | 0 | 0.0% |")

    lines.append("")
    lines.append("## Query anomaly detection (WoW)")
    lines.append("| Query | Delta vs WoW | Delta % vs WoW | Current clicks |")
    lines.append("|---|---:|---:|---:|")
    anomaly_rows = _query_anomaly_rows(_find_scope(scope_results, "query"), limit=6)
    if anomaly_rows:
        for row in anomaly_rows:
            lines.append(
                f"| `{row.get('query', '')}` | {_fmt_signed_int(row.get('delta', 0.0))} | "
                f"{float(row.get('delta_pct', 0.0)):+.2f}% | {_fmt_int(row.get('current_clicks', 0.0))} |"
            )
    else:
        lines.append("| (no outlier anomalies) | 0 | 0.00% | 0 |")

    data_quality_score, data_quality_notes = _data_quality_score(
        additional_context=additional_context,
        external_signals=external_signals,
        senuto_error=senuto_error,
    )
    lines.append("")
    lines.append("## Data quality score")
    lines.append(f"- Run data-quality score: **{data_quality_score}/100**.")
    for note in data_quality_notes:
        lines.append(f"- {note}")

    long_window = (additional_context or {}).get("long_window_context", {})
    if isinstance(long_window, dict) and long_window.get("enabled"):
        kpi = long_window.get("kpi", {})
        windows_map = long_window.get("windows", {})
        lines.append("")
        lines.append("## Long-window context (last 28 days overlay)")
        if isinstance(windows_map, dict):
            current_ctx = windows_map.get("current_28d_context", {})
            previous_ctx = windows_map.get("previous_28d_context", {})
            if isinstance(current_ctx, dict) and isinstance(previous_ctx, dict):
                lines.append(
                    "- Window overlay: "
                    f"current {current_ctx.get('start', '')} to {current_ctx.get('end', '')} "
                    f"vs previous {previous_ctx.get('start', '')} to {previous_ctx.get('end', '')}."
                )
        if isinstance(kpi, dict):
            lines.append(
                f"- Clicks 28d: {_fmt_signed_int(kpi.get('clicks_delta_vs_previous', 0.0))} "
                f"({_signed_pct(float(kpi.get('clicks_delta_pct_vs_previous', 0.0) or 0.0) / 100.0)}) vs previous 28d; "
                f"{_fmt_signed_int(kpi.get('clicks_delta_vs_yoy', 0.0))} "
                f"({_signed_pct(float(kpi.get('clicks_delta_pct_vs_yoy', 0.0) or 0.0) / 100.0)}) vs 52W."
            )
            lines.append(
                f"- Impressions 28d: {_fmt_signed_int(kpi.get('impressions_delta_vs_previous', 0.0))} "
                f"({_signed_pct(float(kpi.get('impressions_delta_pct_vs_previous', 0.0) or 0.0) / 100.0)}) vs previous 28d; "
                f"{_fmt_signed_int(kpi.get('impressions_delta_vs_yoy', 0.0))} "
                f"({_signed_pct(float(kpi.get('impressions_delta_pct_vs_yoy', 0.0) or 0.0) / 100.0)}) vs 52W."
            )
        movers = long_window.get("query_movers", {})
        if isinstance(movers, dict):
            winners = movers.get("winners", [])
            losers = movers.get("losers", [])
            if isinstance(winners, list) and winners:
                top = winners[:5]
                lines.append(
                    "- Top 28d query winners: "
                    + "; ".join(
                        f"`{row.get('key', '')}` ({_fmt_signed_int(row.get('delta_vs_previous', 0.0))} vs prev28d)"
                        for row in top
                        if isinstance(row, dict)
                    )
                    + "."
                )
            if isinstance(losers, list) and losers:
                top = losers[:5]
                lines.append(
                    "- Top 28d query losers: "
                    + "; ".join(
                        f"`{row.get('key', '')}` ({_fmt_signed_int(row.get('delta_vs_previous', 0.0))} vs prev28d)"
                        for row in top
                        if isinstance(row, dict)
                    )
                    + "."
                )

    lines.append("")
    lines.append("## YoY click contribution by query cluster")
    contribution_rows = _query_cluster_contribution_rows(
        query_scope=_find_scope(scope_results, "query"),
        total_click_delta_yoy=click_delta_yoy,
        limit=8,
    )
    lines.append(
        "| Cluster | Current clicks | Delta vs YoY | Contribution vs total YoY click delta | Movers | Example queries |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    if contribution_rows:
        for row in contribution_rows:
            samples = row.get("samples", [])
            sample_text = (
                ", ".join(f"`{str(item).strip()}`" for item in samples[:2] if str(item).strip())
                if isinstance(samples, list)
                else ""
            )
            lines.append(
                f"| {row.get('cluster', '')} | {_fmt_int(row.get('current_clicks', 0.0))} | {_fmt_signed_int(row.get('delta_vs_yoy', 0.0))} | {float(row.get('contribution_pct', 0.0)):+.2f}% | {int(row.get('rows', 0))} | {sample_text or '-'} |"
            )
    else:
        lines.append("| (no query clusters) | 0 | +0 | +0.00% | 0 | - |")

    lines.append("")
    lines.append("## Additional source snapshots")
    pagespeed = (additional_context or {}).get("pagespeed", {})
    lines.append("### CrUX/PageSpeed (origin field data)")
    if isinstance(pagespeed, dict) and pagespeed:
        lines.append("| Strategy | LCP (ms) | INP (ms) | CLS | Category |")
        lines.append("|---|---:|---:|---:|---|")
        for strategy in ("mobile", "desktop"):
            row = pagespeed.get(strategy, {})
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {strategy} | {float(row.get('lcp_ms', 0.0)):.0f} | {float(row.get('inp_ms', 0.0)):.0f} | {float(row.get('cls', 0.0)):.2f} | {row.get('overall_category', '')} |"
            )
    else:
        lines.append("- No PageSpeed/CrUX data in this run.")

    ga4 = (additional_context or {}).get("ga4", {})
    lines.append("")
    lines.append("### Google Analytics 4 (country slice)")
    if isinstance(ga4, dict) and ga4.get("enabled"):
        lines.append(
            f"- Property: {str(ga4.get('property_id', '')).strip() or '(configured)'} | Country: {str(ga4.get('country_code', '')).strip() or '-'}"
        )
        summary = ga4.get("summary", {})
        if isinstance(summary, dict):
            current_ga4 = summary.get("current", {})
            previous_ga4 = summary.get("previous", {})
            yoy_ga4 = summary.get("yoy", {})
            if (
                isinstance(current_ga4, dict)
                and isinstance(previous_ga4, dict)
                and isinstance(yoy_ga4, dict)
            ):
                metric_rows = [
                    ("sessions", "Sessions"),
                    ("users", "Users"),
                    ("engaged_sessions", "Engaged sessions"),
                    ("transactions", "Transactions"),
                    ("revenue", "Purchase revenue"),
                ]
                lines.append("| Metric | Current week | Previous week | YoY (52w) |")
                lines.append("|---|---:|---:|---:|")
                added_rows = 0
                for metric, label in metric_rows:
                    if not _ga4_metric_available(ga4, metric):
                        continue
                    current_value = _ga4_num(current_ga4, metric)
                    previous_value = _ga4_num(previous_ga4, metric)
                    yoy_value = _ga4_num(yoy_ga4, metric)
                    if current_value is None or previous_value is None or yoy_value is None:
                        continue
                    lines.append(
                        f"| {label} | {_fmt_int(current_value)} | {_fmt_int(previous_value)} | {_fmt_int(yoy_value)} |"
                    )
                    added_rows += 1
                if added_rows == 0:
                    lines.append("| (no complete GA4 metrics in this run) | - | - | - |")

        channels = ga4.get("channels", {})
        yoy_deltas = channels.get("yoy_deltas", []) if isinstance(channels, dict) else []
        lines.append("")
        lines.append("| Channel | Sessions current | Sessions YoY | Delta vs YoY | Delta % vs YoY |")
        lines.append("|---|---:|---:|---:|---:|")
        if isinstance(yoy_deltas, list) and yoy_deltas:
            for row in yoy_deltas[: int(ga4.get("top_rows", 10) or 10)]:
                if not isinstance(row, dict):
                    continue
                delta_pct = row.get("delta_vs_yoy_pct")
                delta_pct_text = (
                    f"{float(delta_pct):+.2f}%"
                    if isinstance(delta_pct, (float, int))
                    else "-"
                )
                lines.append(
                    f"| {row.get('channel', '')} | {_fmt_int(row.get('sessions_current', 0.0))} | {_fmt_int(row.get('sessions_yoy', 0.0))} | {_fmt_signed_int(row.get('delta_vs_yoy', 0.0))} | {delta_pct_text} |"
                )
        else:
            lines.append("| (no channel rows) | 0 | 0 | +0 | - |")

        if isinstance(yoy_deltas, list) and yoy_deltas:
            rising = sorted(
                [row for row in yoy_deltas if isinstance(row, dict)],
                key=lambda row: float(row.get("delta_vs_yoy", 0.0) or 0.0),
                reverse=True,
            )[:5]
            falling = sorted(
                [row for row in yoy_deltas if isinstance(row, dict)],
                key=lambda row: float(row.get("delta_vs_yoy", 0.0) or 0.0),
            )[:5]
            if rising:
                lines.append(
                    "- Top channel growth YoY: "
                    + "; ".join(
                        f"{row.get('channel', '')} ({_fmt_signed_int(row.get('delta_vs_yoy', 0.0))})"
                        for row in rising
                    )
                    + "."
                )
            if falling:
                lines.append(
                    "- Top channel decline YoY: "
                    + "; ".join(
                        f"{row.get('channel', '')} ({_fmt_signed_int(row.get('delta_vs_yoy', 0.0))})"
                        for row in falling
                    )
                    + "."
                )

        cannibalization = ga4.get("cannibalization", {})
        if isinstance(cannibalization, dict):
            lines.append(
                "- Potential channel cannibalization (GA4): "
                f"{str(cannibalization.get('note', '')).strip()}"
            )

        top_pages = ga4.get("top_landing_pages", [])
        lines.append("")
        show_transactions = _ga4_metric_available(ga4, "transactions")
        show_revenue = _ga4_metric_available(ga4, "revenue")
        page_header = "| Landing page | Sessions |"
        page_divider = "|---|---:|"
        if show_transactions:
            page_header += " Transactions |"
            page_divider += "---:|"
        if show_revenue:
            page_header += " Revenue |"
            page_divider += "---:|"
        lines.append(page_header)
        lines.append(page_divider)
        if isinstance(top_pages, list) and top_pages:
            for row in top_pages[: int(ga4.get("top_rows", 10) or 10)]:
                if not isinstance(row, dict):
                    continue
                line = (
                    f"| {row.get('landing_page', '')} | {_fmt_int(row.get('sessions', 0.0))} |"
                )
                if show_transactions:
                    line += f" {_fmt_int(row.get('transactions', 0.0))} |"
                if show_revenue:
                    line += f" {_fmt_int(row.get('revenue', 0.0))} |"
                lines.append(line)
        else:
            empty = "| (no rows) | 0 |"
            if show_transactions:
                empty += " 0 |"
            if show_revenue:
                empty += " 0 |"
            lines.append(empty)
    else:
        lines.append("- GA4 not configured or unavailable in this run.")
        if isinstance(ga4, dict):
            errors = ga4.get("errors", [])
            if isinstance(errors, list):
                for err in errors[:3]:
                    lines.append(f"- GA4 warning: {err}")

    allegro_trends = (additional_context or {}).get("allegro_trends", {})
    lines.append("")
    lines.append("### Allegro Trends API (marketplace demand)")
    if isinstance(allegro_trends, dict) and allegro_trends.get("enabled"):
        source_label = str(allegro_trends.get("source", "Allegro Trends API")).strip()
        window_start = str(allegro_trends.get("from", "")).strip()
        window_end = str(allegro_trends.get("till", "")).strip()
        country_trends = str(allegro_trends.get("country_code", country_label)).strip() or country_label
        interval_label = str(allegro_trends.get("interval", "day")).strip() or "day"
        lines.append(
            f"- Source: {source_label} | Country: {country_trends} | Window: {window_start} to {window_end} | Interval: {interval_label}"
        )
        rows = allegro_trends.get("rows", [])
        top_rows = int(allegro_trends.get("top_rows", 10) or 10)
        lines.append("| Query | Visits | PV | Offers | Deals | GMV | Data points | HTTP |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        if isinstance(rows, list) and rows:
            total_visits = 0.0
            total_pv = 0.0
            total_gmv = 0.0
            for row in rows[:max(1, top_rows)]:
                if not isinstance(row, dict):
                    continue
                visits = float(row.get("visit", 0.0) or 0.0)
                pv = float(row.get("pv", 0.0) or 0.0)
                offers = float(row.get("offers", 0.0) or 0.0)
                deals = float(row.get("deals", 0.0) or 0.0)
                gmv = float(row.get("gmv", 0.0) or 0.0)
                points = int(float(row.get("points", 0.0) or 0.0))
                http_code = int(float(row.get("http_code", 0.0) or 0.0))
                total_visits += visits
                total_pv += pv
                total_gmv += gmv
                lines.append(
                    f"| {row.get('query', '')} | {_fmt_int(visits)} | {_fmt_int(pv)} | {_fmt_int(offers)} | {_fmt_int(deals)} | {_fmt_int(gmv)} | {_fmt_int(points)} | {_fmt_int(http_code)} |"
                )
            lines.append(
                f"- Aggregated tracked-query totals: visits {_fmt_int(total_visits)}, PV {_fmt_int(total_pv)}, GMV {_fmt_int(total_gmv)}."
            )
        else:
            lines.append("| (no rows) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
        errors = allegro_trends.get("errors", [])
        if isinstance(errors, list):
            for err in errors[:3]:
                lines.append(f"- Allegro Trends warning: {err}")
    else:
        lines.append("- Allegro Trends API not configured or unavailable in this run.")
        if isinstance(allegro_trends, dict):
            errors = allegro_trends.get("errors", [])
            if isinstance(errors, list):
                for err in errors[:3]:
                    lines.append(f"- Allegro Trends warning: {err}")

    trends = (additional_context or {}).get("google_trends", [])
    lines.append("")
    lines.append(f"### Google Trends ({country_label})")
    if isinstance(trends, list) and trends:
        lines.append("| Date | Topic | Approx traffic |")
        lines.append("|---|---|---:|")
        for row in trends[:12]:
            if not isinstance(row, dict):
                continue
            day = row.get("day")
            day_label = day.isoformat() if isinstance(day, date) else str(day or "")
            lines.append(
                f"| {day_label} | {row.get('topic', '')} | {int(row.get('approx_traffic', 0))} |"
            )
    else:
        lines.append("- No Google Trends items in this run.")
    brand_trends = (additional_context or {}).get("google_trends_brand", {})
    if isinstance(brand_trends, dict) and brand_trends.get("enabled"):
        rows = brand_trends.get("rows", [])
        summary = brand_trends.get("summary", {})
        lines.append("")
        lines.append("#### Brand Search Interest (Google Trends API)")
        lines.append("| Keyword | Avg interest (current) | Avg interest (previous) | Avg interest (YoY) | WoW delta % | YoY delta % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        if isinstance(rows, list) and rows:
            for row in rows[:6]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('keyword', '')} | {float(row.get('current_avg', 0.0)):.1f} | {float(row.get('previous_avg', 0.0)):.1f} | {float(row.get('yoy_avg', 0.0)):.1f} | {float(row.get('delta_pct_vs_previous', 0.0)):+.2f}% | {float(row.get('delta_pct_vs_yoy', 0.0)):+.2f}% |"
                )
        else:
            lines.append("| (no rows) | 0.0 | 0.0 | 0.0 | +0.00% | +0.00% |")
        if isinstance(summary, dict):
            lines.append(
                "- Blended brand-interest trend: "
                f"WoW {float(summary.get('delta_pct_vs_previous', 0.0)):+.2f}%, "
                f"YoY {float(summary.get('delta_pct_vs_yoy', 0.0)):+.2f}%."
            )
        errors = brand_trends.get("errors", [])
        if isinstance(errors, list):
            for err in errors[:3]:
                lines.append(f"- Google Trends brand warning: {err}")

    lines.append("")
    lines.append("### DuckDuckGo context scan")
    ddg_context = (additional_context or {}).get("duckduckgo_context", {})
    if isinstance(ddg_context, dict) and ddg_context.get("enabled"):
        ddg_rows = ddg_context.get("rows", [])
        lines.append("| Query | Heading | Abstract/Hint |")
        lines.append("|---|---|---|")
        if isinstance(ddg_rows, list) and ddg_rows:
            for row in ddg_rows[:8]:
                if not isinstance(row, dict):
                    continue
                query = str(row.get("query", "")).strip()
                heading = str(row.get("heading", "")).strip()
                abstract = str(row.get("abstract", "")).strip()
                if len(abstract) > 220:
                    abstract = abstract[:217] + "..."
                related = row.get("related_topics", [])
                if not abstract and isinstance(related, list) and related:
                    abstract = "; ".join(str(item) for item in related[:2])
                lines.append(f"| {query} | {heading or '-'} | {abstract or '-'} |")
        else:
            lines.append("| (no rows) | - | - |")
        errors = ddg_context.get("errors", [])
        if isinstance(errors, list):
            for err in errors[:3]:
                lines.append(f"- DuckDuckGo warning: {err}")
    else:
        lines.append("- DuckDuckGo context not available in this run.")

    lines.append("")
    lines.append("### Market event calendar (API)")
    market_events = (additional_context or {}).get("market_event_calendar", {})
    if isinstance(market_events, dict) and market_events.get("enabled"):
        lines.append(
            f"- Source: {market_events.get('source', 'Market Events API')} | Country: {market_events.get('country_code', country_label)}"
        )
        rows = market_events.get("events", [])
        lines.append("| Date | Event type | GMV impact | Source | Event | Why it may affect GMV |")
        lines.append("|---|---|---|---|---|---|")
        if isinstance(rows, list) and rows:
            max_rows = int(market_events.get("top_rows", 12) or 12)
            for row in rows[:max_rows]:
                if not isinstance(row, dict):
                    continue
                impact = (
                    f"{row.get('impact_level', 'LOW')} / {row.get('impact_direction', 'Mixed')} "
                    f"(conf {int(row.get('confidence', 0) or 0)}/100)"
                )
                lines.append(
                    f"| {row.get('date', '')} | {row.get('event_type', '')} | {impact} | {row.get('source', '')} | {row.get('title', '')} | {row.get('gmv_reason', '')} |"
                )
        else:
            lines.append("| (no API events) | - | - | - | - | - |")
        errors = market_events.get("errors", [])
        if isinstance(errors, list):
            for err in errors[:3]:
                lines.append(f"- Market event calendar warning: {err}")
    else:
        lines.append("- Market event calendar not configured or unavailable in this run.")

    lines.append("")
    lines.append("### SEO update early analyses (external SEO media)")
    seo_update_rows = [
        signal
        for signal in external_signals
        if "seo update analysis" in _normalize_text(signal.source)
    ]
    if seo_update_rows:
        lines.append("| Date | Source | Update article | Early takeaways |")
        lines.append("|---|---|---|---|")
        for signal in seo_update_rows[:8]:
            details = str(signal.details or "").strip()
            if len(details) > 220:
                details = details[:217] + "..."
            lines.append(
                f"| {signal.day.isoformat()} | {signal.source} | {signal.title} | {details} |"
            )
    else:
        lines.append("- No SEO-media early analyses detected in this run window.")

    product_trends = (additional_context or {}).get("product_trends", {})
    lines.append("")
    lines.append("### Product trends (non-brand, Sheets)")
    if isinstance(product_trends, dict) and product_trends.get("enabled"):
        lines.append(f"- Source: {product_trends.get('source', 'Google Sheets product trend trackers')}")
        lines.append(f"- Top rows used: {int(product_trends.get('top_rows', 0))}")
        lines.append(f"- Upcoming horizon: {int(product_trends.get('horizon_days', 0))} days")

        yoy_rows = product_trends.get("top_yoy_non_brand", [])
        if not (isinstance(yoy_rows, list) and yoy_rows):
            yoy_rows = _fallback_non_brand_yoy_from_gsc(
                query_scope=_find_scope(scope_results, "query"),
                top_rows=int(product_trends.get("top_rows", 12)),
            )
            if yoy_rows:
                lines.append("- YoY trend source fallback: GSC non-brand queries (sheet YoY snapshot unavailable).")
        lines.append("")
        if isinstance(yoy_rows, list) and yoy_rows:
            first_yoy_row = next((row for row in yoy_rows if isinstance(row, dict)), None)
            if isinstance(first_yoy_row, dict):
                current_snapshot = str(first_yoy_row.get("current_snapshot_date", "")).strip()
                previous_snapshot = str(first_yoy_row.get("previous_snapshot_date", "")).strip()
                if current_snapshot or previous_snapshot:
                    lines.append(
                        f"- YoY snapshot dates: current={current_snapshot or '-'} | previous={previous_snapshot or '-'}"
                    )
            lines.append("| Trend | Current year | Previous year | Delta | Delta % | Sheet |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for row in yoy_rows[: int(product_trends.get("top_rows", 12))]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('trend', '')} | {_fmt_int(row.get('current_value', 0.0))} | {_fmt_int(row.get('previous_value', 0.0))} | {_fmt_signed_int(row.get('delta_value', 0.0))} | {float(row.get('delta_pct', 0.0)):+.2f}% | {row.get('sheet', '')} |"
                )
        else:
            lines.append("| Trend | Current year | Previous year | Delta | Delta % | Sheet |")
            lines.append("|---|---:|---:|---:|---:|---|")
            lines.append("| (no data) | 0 | 0 | +0 | +0.00% | - |")

        upcoming_rows = product_trends.get("upcoming_31d", [])
        lines.append("")
        lines.append("| Upcoming date | Trend | Value | Sheet |")
        lines.append("|---|---|---:|---|")
        if isinstance(upcoming_rows, list) and upcoming_rows:
            for row in upcoming_rows[: int(product_trends.get("top_rows", 12))]:
                if not isinstance(row, dict):
                    continue
                day_label = str(row.get("date", "")).strip() or "(next31d)"
                lines.append(
                    f"| {day_label} | {row.get('trend', '')} | {float(row.get('value', 0.0)):.2f} | {row.get('sheet', '')} |"
                )
        else:
            lines.append("| (no data) | - | 0.00 | - |")

        current_rows = product_trends.get("current_non_brand", [])
        lines.append("")
        lines.append("| Trend date | Current trend | Value | Sheet |")
        lines.append("|---|---|---:|---|")
        if isinstance(current_rows, list) and current_rows:
            for row in current_rows[: int(product_trends.get("top_rows", 12))]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('date', '')} | {row.get('trend', '')} | {float(row.get('value', 0.0)):.2f} | {row.get('sheet', '')} |"
                )
        else:
            lines.append("| (no data) | - | 0.00 | - |")

        trend_errors = product_trends.get("errors", [])
        if isinstance(trend_errors, list):
            for err in trend_errors[:5]:
                lines.append(f"- Product trend warning: {err}")
    else:
        lines.append("- No product trend sheet data in this run.")

    senuto_intelligence = (additional_context or {}).get("senuto_intelligence", {})
    lines.append("")
    lines.append("### Senuto competitor and SERP intelligence")
    if isinstance(senuto_intelligence, dict) and senuto_intelligence.get("enabled"):
        lines.append(
            f"- Country id: {int(senuto_intelligence.get('country_id', 0) or 0)} | Top rows: {int(senuto_intelligence.get('top_rows', 10) or 10)}"
        )

        competitors_overview = senuto_intelligence.get("competitors_overview", [])
        if isinstance(competitors_overview, list) and competitors_overview:
            lines.append("| Competitor domain | Common keywords | Visibility (current) | Visibility delta % | Top10 keywords | Domain rank |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in competitors_overview[: int(senuto_intelligence.get("top_rows", 10) or 10)]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('domain', '')} | {_fmt_int(row.get('common_keywords', 0.0))} | {_fmt_int(row.get('visibility_current', 0.0))} | {float(row.get('visibility_diff_pct', 0.0)):+.2f}% | {_fmt_int(row.get('top10_current', 0.0))} | {_fmt_int(row.get('domain_rank_current', 0.0))} |"
                )

        wins_losses = senuto_intelligence.get("wins_losses", {})
        wins_rows = wins_losses.get("wins", []) if isinstance(wins_losses, dict) else []
        losses_rows = wins_losses.get("losses", []) if isinstance(wins_losses, dict) else []
        if isinstance(wins_rows, list) and wins_rows:
            lines.append("")
            lines.append("| Keyword movers | Visibility current | Visibility previous | Visibility diff | Visibility diff % |")
            lines.append("|---|---:|---:|---:|---:|")
            for row in wins_rows[:5]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| WIN: {row.get('keyword', '')} | {_fmt_int(row.get('visibility_current', 0.0))} | {_fmt_int(row.get('visibility_previous', 0.0))} | {_fmt_signed_int(row.get('visibility_diff', 0.0))} | {float(row.get('visibility_diff_pct', 0.0)):+.2f}% |"
                )
        if isinstance(losses_rows, list) and losses_rows:
            if not (isinstance(wins_rows, list) and wins_rows):
                lines.append("")
                lines.append("| Keyword movers | Visibility current | Visibility previous | Visibility diff | Visibility diff % |")
                lines.append("|---|---:|---:|---:|---:|")
            for row in losses_rows[:5]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| LOSS: {row.get('keyword', '')} | {_fmt_int(row.get('visibility_current', 0.0))} | {_fmt_int(row.get('visibility_previous', 0.0))} | {_fmt_signed_int(row.get('visibility_diff', 0.0))} | {float(row.get('visibility_diff_pct', 0.0)):+.2f}% |"
                )

        acquired_lost = senuto_intelligence.get("acquired_lost", {})
        kw_acquired = acquired_lost.get("keywords_acquired", []) if isinstance(acquired_lost, dict) else []
        kw_lost = acquired_lost.get("keywords_lost", []) if isinstance(acquired_lost, dict) else []
        url_acquired = acquired_lost.get("urls_acquired", []) if isinstance(acquired_lost, dict) else []
        url_lost = acquired_lost.get("urls_lost", []) if isinstance(acquired_lost, dict) else []
        lines.append("")
        lines.append(
            f"- Acquired/Lost snapshot: keywords acquired={_fmt_int(len(kw_acquired) if isinstance(kw_acquired, list) else 0)}, "
            f"keywords lost={_fmt_int(len(kw_lost) if isinstance(kw_lost, list) else 0)}, "
            f"urls acquired={_fmt_int(len(url_acquired) if isinstance(url_acquired, list) else 0)}, "
            f"urls lost={_fmt_int(len(url_lost) if isinstance(url_lost, list) else 0)}."
        )

        direct_answers = senuto_intelligence.get("direct_answers", [])
        if isinstance(direct_answers, list) and direct_answers:
            lines.append("")
            lines.append("| Direct-answer keyword | Feature | Searches | Position |")
            lines.append("|---|---|---:|---:|")
            for row in direct_answers[: int(senuto_intelligence.get("top_rows", 10) or 10)]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('keyword', '')} | {row.get('feature', '') or '-'} | {_fmt_int(row.get('searches', 0.0))} | {float(row.get('position', 0.0)):.1f} |"
                )

        seasonality = senuto_intelligence.get("seasonality", {})
        if isinstance(seasonality, dict):
            trend_values = seasonality.get("trend_values", [])
            if isinstance(trend_values, list) and trend_values:
                lines.append("")
                lines.append("| Seasonality (month) | Index |")
                lines.append("|---|---:|")
                for idx, value in enumerate(trend_values, start=1):
                    lines.append(f"| M{idx:02d} | {float(value):.2f} |")
                lines.append(
                    f"- Peak month M{int(seasonality.get('peak_month', 0) or 0)} ({float(seasonality.get('peak_value', 0.0) or 0.0):.2f}), "
                    f"low month M{int(seasonality.get('low_month', 0) or 0)} ({float(seasonality.get('low_value', 0.0) or 0.0):.2f})."
                )

        market_ranking = senuto_intelligence.get("market_ranking", [])
        if isinstance(market_ranking, list) and market_ranking:
            lines.append("")
            lines.append("| Market ranking domain | Category | Rank current | Rank previous | Visibility current | Visibility previous | Share |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for row in market_ranking[: int(senuto_intelligence.get("top_rows", 10) or 10)]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('domain', '')} | {row.get('category', '')} | {_fmt_int(row.get('rank_current', 0.0))} | {_fmt_int(row.get('rank_previous', 0.0))} | {_fmt_int(row.get('visibility_current', 0.0))} | {_fmt_int(row.get('visibility_previous', 0.0))} | {float(row.get('share', 0.0)):.2f} |"
                )

        keyword_trending = senuto_intelligence.get("keyword_trending", [])
        if isinstance(keyword_trending, list) and keyword_trending:
            lines.append("")
            lines.append("| Senuto trending keyword | Searches | Growth metric |")
            lines.append("|---|---:|---:|")
            for row in keyword_trending[: int(senuto_intelligence.get("top_rows", 10) or 10)]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('keyword', '')} | {_fmt_int(row.get('searches', 0.0))} | {float(row.get('growth', 0.0)):.2f} |"
                )

        serp_volatility = senuto_intelligence.get("serp_volatility", [])
        if isinstance(serp_volatility, list) and serp_volatility:
            lines.append("")
            lines.append("| SERP volatility keyword | Domain | Volatility | Position points |")
            lines.append("|---|---|---:|---:|")
            for row in serp_volatility[: int(senuto_intelligence.get("top_rows", 10) or 10)]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('keyword', '')} | {row.get('domain', '')} | {float(row.get('volatility', 0.0)):.2f} | {_fmt_int(row.get('positions_points', 0.0))} |"
                )

        errors = senuto_intelligence.get("errors", [])
        if isinstance(errors, list):
            for err in errors[:8]:
                lines.append(f"- Senuto intelligence warning: {err}")
    else:
        lines.append("- Senuto intelligence not configured in this run.")

    lines.append("")
    lines.append("### Trend-to-page coverage check")
    coverage_rows = _trend_page_coverage_rows(
        scope_results=scope_results,
        additional_context=additional_context,
        limit=8,
    )
    lines.append(
        "| Trend | Landing page | Trend score/value | Median position (sheet) | Page delta WoW | Page delta YoY | Status | Gap interpretation |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    if coverage_rows:
        for row in coverage_rows:
            lines.append(
                f"| {row.get('trend', '')} | {row.get('page', '-')} | {float(row.get('sheet_value', 0.0)):.2f} | {float(row.get('median_position', 0.0)):.2f} | {_fmt_signed_int(row.get('page_delta_vs_prev', 0.0))} | {_fmt_signed_int(row.get('page_delta_vs_yoy', 0.0))} | {row.get('status', '')} | {row.get('gap_reason', '')} |"
            )
    else:
        lines.append("| (no trend coverage rows) | - | 0.00 | 0.00 | +0 | +0 | - | No trend rows with landing page metadata in this run. |")

    lines.append("")
    lines.append("### Query clusters (top movers grouped)")
    cluster_rows = _query_cluster_rows(_find_scope(scope_results, "query"))
    lines.append("| Cluster | Movers | Current clicks | Delta vs prev | Delta vs YoY | Example queries |")
    lines.append("|---|---:|---:|---:|---:|---|")
    if cluster_rows:
        for row in cluster_rows[:8]:
            samples = row.get("samples", [])
            sample_text = ""
            if isinstance(samples, list):
                sample_text = ", ".join(
                    f"`{str(item).strip()}`" for item in samples[:3] if str(item).strip()
                )
            lines.append(
                f"| {row.get('cluster', '')} | {int(row.get('rows', 0))} | {_fmt_int(row.get('current_clicks', 0.0))} | {_fmt_signed_int(row.get('delta_vs_previous', 0.0))} | {_fmt_signed_int(row.get('delta_vs_yoy', 0.0))} | {sample_text or '-'} |"
            )
    else:
        lines.append("| (no query clusters) | 0 | 0 | +0 | +0 | - |")

    campaign_context = _campaign_event_context(
        external_signals=external_signals,
        query_scope=_find_scope(scope_results, "query"),
    )
    allegro_campaign_events = campaign_context.get("allegro_events", [])
    competitor_campaign_events = campaign_context.get("competitor_events", [])
    campaign_query_events = campaign_context.get("query_events", [])
    lines.append("")
    lines.append("### Campaign events (Allegro vs competitors)")
    has_rows = False
    if isinstance(allegro_campaign_events, list) and allegro_campaign_events:
        lines.append("| Side | Date | Source | Event |")
        lines.append("|---|---|---|---|")
    if isinstance(allegro_campaign_events, list):
        for row in allegro_campaign_events[:8]:
            if not isinstance(row, ExternalSignal):
                continue
            has_rows = True
            lines.append(
                f"| Allegro | {row.day.isoformat()} | {row.source} | {row.title} |"
            )
    if isinstance(competitor_campaign_events, list):
        for row in competitor_campaign_events[:8]:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], ExternalSignal)
            ):
                continue
            if not has_rows:
                lines.append("| Side | Date | Source | Event |")
                lines.append("|---|---|---|---|")
            signal, competitor = row
            has_rows = True
            lines.append(
                f"| Competitor ({competitor}) | {signal.day.isoformat()} | {signal.source} | {signal.title} |"
            )
    if not has_rows:
        lines.append("- No detected campaign-event mentions in current source set.")

    lines.append("")
    if isinstance(campaign_query_events, list) and campaign_query_events:
        lines.append("| Campaign query | Delta vs prev | Delta vs YoY |")
        lines.append("|---|---:|---:|")
        for row in campaign_query_events[:8]:
            lines.append(
                f"| {row.key} | {_fmt_signed_int(row.click_delta_vs_previous)} ({_signed_pct(row.click_delta_pct_vs_previous)}) | {_fmt_signed_int(row.click_delta_vs_yoy)} ({_signed_pct(row.click_delta_pct_vs_yoy)}) |"
            )
    else:
        lines.append("- No campaign queries among top movers in this run.")

    macro = (additional_context or {}).get("macro", {})
    lines.append("")
    lines.append("### Macro context (NBP / IMGW)")
    if isinstance(macro, dict) and macro:
        note = str(macro.get("note", "")).strip()
        if note:
            lines.append(f"- {note}")
        nbp_fx = macro.get("nbp_fx", {})
        if isinstance(nbp_fx, dict):
            lines.append("| Pair | Avg current | Avg previous | Delta vs previous | Latest |")
            lines.append("|---|---:|---:|---:|---:|")
            for key, label in (("eur_pln", "EUR/PLN"), ("usd_pln", "USD/PLN")):
                row = nbp_fx.get(key, {})
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {label} | {float(row.get('avg_current', 0.0)):.4f} | {float(row.get('avg_previous', 0.0)):.4f} | {float(row.get('delta_pct_vs_previous', 0.0)):+.2f}% | {float(row.get('latest', 0.0)):.4f} |"
                )
        warnings = macro.get("imgw_warnings", [])
        if isinstance(warnings, list):
            total_warnings = int(macro.get("imgw_warnings_total", len(warnings)))
            high_warnings = int(
                macro.get(
                    "imgw_high_severity_count",
                    sum(
                        1
                        for row in warnings
                        if isinstance(row, dict)
                        and str(row.get("severity", "")).strip() in {"2", "3"}
                    ),
                )
            )
            lines.append(
                f"- Active IMGW warnings: {total_warnings} (showing {len(warnings)}), level 2-3: {high_warnings}."
            )
            if warnings:
                lines.append("| Severity | Event | Areas | Valid to |")
                lines.append("|---:|---|---|---|")
                for row in warnings[:8]:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        f"| {row.get('severity', '') or '-'} | {row.get('event', '') or '-'} | {row.get('areas', '') or '-'} | {row.get('to', '') or '-'} |"
                    )
    else:
        lines.append("- No macro context data in this run.")

    macro_backdrop = (additional_context or {}).get("macro_backdrop", {})
    lines.append("")
    lines.append("### Macro backdrop (country-level, World Bank)")
    if isinstance(macro_backdrop, dict) and macro_backdrop.get("rows"):
        lines.append(
            f"- Source: {macro_backdrop.get('source', 'World Bank API')} | Country: {macro_backdrop.get('country_code', country_label)}"
        )
        rows = macro_backdrop.get("rows", {})
        if isinstance(rows, dict):
            lines.append("| Indicator | Latest year | Latest value | Previous year | Previous value |")
            lines.append("|---|---:|---:|---:|---:|")
            for key, label in (
                ("inflation_cpi_pct", "Inflation CPI %"),
                ("unemployment_pct", "Unemployment %"),
                ("gdp_growth_pct", "GDP growth %"),
            ):
                row = rows.get(key, {})
                if not isinstance(row, dict):
                    continue
                latest_val = row.get("latest_value")
                prev_val = row.get("previous_value")
                latest_text = f"{float(latest_val):.2f}" if isinstance(latest_val, (int, float)) else "-"
                prev_text = f"{float(prev_val):.2f}" if isinstance(prev_val, (int, float)) else "-"
                lines.append(
                    f"| {label} | {row.get('latest_year', '') or '-'} | {latest_text} | {row.get('previous_year', '') or '-'} | {prev_text} |"
                )
        errors = macro_backdrop.get("errors", [])
        if isinstance(errors, list):
            for err in errors[:3]:
                lines.append(f"- Macro backdrop warning: {err}")
    else:
        lines.append("- No country-level macro backdrop rows in this run.")

    operational = (additional_context or {}).get("operational_risks", {})
    lines.append("")
    lines.append("### Operational risks (logistics / payments)")
    if isinstance(operational, dict) and operational.get("enabled"):
        logistics = operational.get("logistics", [])
        payments = operational.get("payments", [])
        lines.append(
            f"- Signals captured: logistics={_fmt_int(len(logistics) if isinstance(logistics, list) else 0)}, "
            f"payments={_fmt_int(len(payments) if isinstance(payments, list) else 0)}."
        )
        lines.append("| Type | Date | Headline |")
        lines.append("|---|---|---|")
        if isinstance(logistics, list):
            for row in logistics[:5]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"| Logistics | {row.get('date', '')} | {row.get('title', '')} |")
        if isinstance(payments, list):
            for row in payments[:5]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"| Payments | {row.get('date', '')} | {row.get('title', '')} |")
    else:
        lines.append("- No operational-risk signals in this run.")

    promo_radar = (additional_context or {}).get("competitor_promo_radar", {})
    lines.append("")
    lines.append("### Competitor promo radar")
    if isinstance(promo_radar, dict) and promo_radar.get("enabled"):
        rows = promo_radar.get("rows", [])
        lines.append(
            f"- Source: {promo_radar.get('source', 'Google News RSS query')} | Rows: {_fmt_int(len(rows) if isinstance(rows, list) else 0)}"
        )
        lines.append("| Date | Signal |")
        lines.append("|---|---|")
        if isinstance(rows, list):
            for row in rows[:10]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"| {row.get('date', '')} | {row.get('title', '')} |")
    else:
        lines.append("- No competitor promo-radar mentions in this run.")

    source_errors = (additional_context or {}).get("errors", [])
    if isinstance(source_errors, list):
        for err in source_errors[:5]:
            lines.append(f"- Source warning: {err}")

    seo_presentations = (additional_context or {}).get("seo_presentations", {})
    lines.append("")
    lines.append("### SEO team presentations (Drive)")
    if isinstance(seo_presentations, dict) and seo_presentations.get("enabled"):
        lines.append(f"- Source: {seo_presentations.get('source', 'Google Drive')}")
        year_rows = seo_presentations.get("years", [])
        if isinstance(year_rows, list) and year_rows:
            lines.append("| Year | Files in folder | Files with insights | Top insight sample |")
            lines.append("|---|---:|---:|---|")
            for year_row in year_rows[:2]:
                if not isinstance(year_row, dict):
                    continue
                year_label = str(year_row.get("year", ""))
                file_count = int(year_row.get("file_count", 0))
                file_rows = year_row.get("files", [])
                files_with_highlights = 0
                top_note = ""
                if isinstance(file_rows, list):
                    for file_row in file_rows:
                        if isinstance(file_row, dict):
                            notes = file_row.get("highlights", []) or []
                            note_count = len(notes)
                            if note_count:
                                files_with_highlights += 1
                                if not top_note:
                                    top_note = str(notes[0]).strip()
                if not top_note:
                    top_note = "No concrete slide highlights extracted."
                lines.append(
                    f"| {year_label} | {file_count} | {files_with_highlights} | {top_note} |"
                )
        else:
            lines.append("- No year folders found for current/previous year in the configured Drive root.")

        highlights = seo_presentations.get("highlights", [])
        if isinstance(highlights, list) and highlights:
            lines.append("")
            lines.append("| Date | File | Highlight |")
            lines.append("|---|---|---|")
            for row in highlights[:10]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('date', '')} | {row.get('file', '')} | {row.get('note', '')} |"
                )
        errors = seo_presentations.get("errors", [])
        if isinstance(errors, list):
            for err in errors[:5]:
                lines.append(f"- Presentation warning: {err}")
    else:
        lines.append("- No SEO presentation data in this run.")

    historical_reports = (additional_context or {}).get("historical_reports", {})
    lines.append("")
    lines.append("### Previous report continuity (Drive)")
    if isinstance(historical_reports, dict) and historical_reports.get("enabled"):
        lines.append(f"- Source: {historical_reports.get('source', 'Google Drive')}")
        lines.append(f"- Available reports in archive: {int(historical_reports.get('available_reports', 0))}")

        recent_reports = historical_reports.get("recent_reports", [])
        if isinstance(recent_reports, list) and recent_reports:
            lines.append("| Report date | File | Highlight sample |")
            lines.append("|---|---|---|")
            for row in recent_reports[:3]:
                if not isinstance(row, dict):
                    continue
                highlights = row.get("highlights", [])
                highlight = ""
                if isinstance(highlights, list) and highlights:
                    highlight = str(highlights[0]).strip()
                if not highlight:
                    highlight = str(row.get("excerpt", "")).strip()[:180]
                lines.append(
                    f"| {row.get('date', '')} | {row.get('name', '')} | {highlight} |"
                )
        else:
            lines.append("- No previous reports found before current run date.")

        yoy_report = historical_reports.get("yoy_report", {})
        if isinstance(yoy_report, dict) and yoy_report.get("id"):
            yoy_highlights = yoy_report.get("highlights", [])
            yoy_note = ""
            if isinstance(yoy_highlights, list) and yoy_highlights:
                yoy_note = str(yoy_highlights[0]).strip()
            if not yoy_note:
                yoy_note = str(yoy_report.get("excerpt", "")).strip()[:180]
            lines.append(
                f"- YoY reference report: {yoy_report.get('date', '')} | {yoy_report.get('name', '')} | {yoy_note}"
            )
        else:
            lines.append("- YoY reference report not found within configured tolerance.")

        history_errors = historical_reports.get("errors", [])
        if isinstance(history_errors, list):
            for err in history_errors[:5]:
                lines.append(f"- History warning: {err}")
    else:
        lines.append("- No historical report context in this run.")

    lines.append("")
    lines.append("### What changed vs prior-report hypotheses")
    continuity_rows = _hypothesis_continuity_rows(
        hypotheses=hypotheses,
        additional_context=additional_context,
        limit=6,
    )
    lines.append("| Current hypothesis category | Continuity status | Evidence |")
    lines.append("|---|---|---|")
    if continuity_rows:
        for row in continuity_rows:
            lines.append(
                f"| {row.get('category', '')} | {row.get('status', '')} | {row.get('evidence', '')} |"
            )
    else:
        lines.append("| (no continuity rows) | No prior-report context | Historical weekly reports were not available in this run. |")

    tracker = (additional_context or {}).get("hypothesis_tracker", {})
    lines.append("")
    lines.append("### Hypothesis tracker")
    if isinstance(tracker, dict):
        counts = tracker.get("counts", {})
        if isinstance(counts, dict):
            lines.append(
                "- Status counts this run: "
                f"new={_fmt_int(counts.get('new', 0))}, "
                f"confirmed={_fmt_int(counts.get('confirmed', 0))}, "
                f"rejected={_fmt_int(counts.get('rejected', 0))}."
            )
        active = tracker.get("active", [])
        rejected = tracker.get("rejected", [])
        lines.append("| Hypothesis ID | Status | Category | Confidence |")
        lines.append("|---|---|---|---:|")
        has_rows = False
        if isinstance(active, list):
            for row in active[:8]:
                if not isinstance(row, dict):
                    continue
                has_rows = True
                lines.append(
                    f"| {row.get('id', '')} | {row.get('status', '')} | {row.get('category', '')} | "
                    f"{int(row.get('confidence', 0) or 0)}/100 |"
                )
        if isinstance(rejected, list):
            for row in rejected[:4]:
                if not isinstance(row, dict):
                    continue
                has_rows = True
                lines.append(
                    f"| {row.get('id', '')} | {row.get('status', '')} | {row.get('category', '')} | "
                    f"{int(row.get('confidence', 0) or 0)}/100 |"
                )
        if not has_rows:
            lines.append("| (no tracked hypotheses) | - | - | 0/100 |")
    else:
        lines.append("- Hypothesis tracker not available in this run.")

    status_log = (additional_context or {}).get("status_log", {})
    lines.append("")
    lines.append("### SEO status log (Sheets)")
    if isinstance(status_log, dict) and status_log.get("enabled"):
        lines.append(f"- Source: {status_log.get('source', 'Google Sheets')}")
        file_name = str(status_log.get("file_name", "")).strip()
        if file_name:
            lines.append(f"- File: {file_name}")
        selected_sheets = status_log.get("selected_sheets", [])
        if isinstance(selected_sheets, list) and selected_sheets:
            lines.append(f"- Year sheets used: {', '.join(str(item) for item in selected_sheets[:4])}")

        entries = status_log.get("entries", [])
        if isinstance(entries, list) and entries:
            lines.append("| Date | Sheet | Topic | Summary |")
            lines.append("|---|---|---|---|")
            for row in entries[:10]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('date', '')} | {row.get('sheet', '')} | {row.get('topic', '')} | {row.get('summary', '')} |"
                )
        else:
            lines.append("- No dated status entries found in selected sheets.")

        status_errors = status_log.get("errors", [])
        if isinstance(status_errors, list):
            for err in status_errors[:5]:
                lines.append(f"- Status warning: {err}")
    else:
        lines.append("- No status-log context in this run.")

    lines.append("")
    lines.append("## Winter-break regional context (PL)")
    lines.append(
        "- Regional GMV proxy is estimated as: `population x average salary` (normalized to 100% total)."
    )
    source_name = str(ferie_context.get("source", "")).strip()
    source_url = str(ferie_context.get("source_url", "")).strip()
    if source_name:
        if source_url:
            lines.append(f"- Source: {source_name} ({source_url})")
        else:
            lines.append(f"- Source: {source_name}")
    ferie_errors = ferie_context.get("errors")
    if isinstance(ferie_errors, list):
        for error in ferie_errors[:3]:
            lines.append(f"- Data warning: {error}")

    missing_years = ferie_context.get("missing_years", [])
    if isinstance(missing_years, list) and missing_years:
        missing_years_label = ", ".join(str(year) for year in missing_years)
        lines.append(f"- Missing winter-break calendar for years: {missing_years_label}.")

    lines.append("| Window | Days with winter-break overlap | Avg daily GMV proxy exposure | Peak daily GMV proxy exposure |")
    lines.append("|---|---:|---:|---:|")
    ferie_window_stats = ferie_context.get("window_stats", {})
    for key in ["current_28d", "previous_28d", "yoy_52w"]:
        row = ferie_window_stats.get(key, {}) if isinstance(ferie_window_stats, dict) else {}
        lines.append(
            f"| {windows[key].name} | {int(row.get('days_with_ferie', 0))} | {float(row.get('avg_daily_gmv_share', 0.0)) * 100:.2f}% | {float(row.get('peak_daily_gmv_share', 0.0)) * 100:.2f}% |"
        )

    lines.append("")
    lines.append("### Strongest regions by GMV proxy")
    lines.append("| Voivodeship | Population (m) | Avg salary PLN | GMV proxy share |")
    lines.append("|---|---:|---:|---:|")
    profiles_ranked = ferie_context.get("profiles_ranked", [])
    if isinstance(profiles_ranked, list) and profiles_ranked:
        for profile in profiles_ranked[:8]:
            if not isinstance(profile, dict):
                continue
            lines.append(
                f"| {profile.get('name', '')} | {float(profile.get('population_m', 0.0)):.2f} | {_fmt_int(profile.get('avg_salary_pln', 0.0))} | {float(profile.get('gmv_share', 0.0)) * 100:.2f}% |"
            )
    else:
        lines.append("| (no data) | 0.00 | 0 | 0.00% |")

    lines.append("")
    lines.append("### Winter-break YoY differences by region (current window vs 52W)")
    lines.append("| Voivodeship | Current break days | YoY break days | Delta days | Contribution to YoY exposure delta (pp) |")
    lines.append("|---|---:|---:|---:|---:|")
    yoy_comparison = ferie_context.get("yoy_comparison", {})
    yoy_rows = yoy_comparison.get("rows", []) if isinstance(yoy_comparison, dict) else []
    if isinstance(yoy_rows, list) and yoy_rows:
        for row in yoy_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {row.get('name', '')} | {int(row.get('current_days', 0))} | {int(row.get('yoy_days', 0))} | {int(row.get('delta_days', 0)):+d} | {float(row.get('contribution_pp', 0.0)):+.2f} |"
            )
        lines.append(
            f"- Net YoY delta in avg daily winter-break exposure: {float(yoy_comparison.get('avg_daily_delta_pp', 0.0)):+.2f} pp."
        )
    else:
        lines.append("| (no YoY differences) | 0 | 0 | +0 | +0.00 |")

    lines.append("")
    lines.append("## Integrated reasoning")
    lines.append("Reasoning combines KPI, GSC segments, winter breaks/holidays, weather, and external sources.")
    for reasoning_line in _build_integrated_reasoning(
        totals=totals,
        scope_results=scope_results,
        external_signals=external_signals,
        weather_summary=weather_summary,
        ferie_context=ferie_context,
        segment_diagnostics=segment_diagnostics,
        additional_context=additional_context,
        senuto_summary=senuto_summary,
        senuto_error=senuto_error,
    ):
        lines.append(reasoning_line)

    lines.append("")
    lines.append("## Root cause matrix")
    lines.append("| Root cause category | Confidence | Hypothesis | Key evidence | Owner |")
    lines.append("|---|---:|---|---|---|")
    for row in hypotheses[:8]:
        evidence = row.get("evidence", [])
        evidence_text = ""
        if isinstance(evidence, list):
            evidence_text = "; ".join(str(item) for item in evidence[:2])
        lines.append(
            f"| {row.get('category', '')} | {int(row.get('confidence', 0))}/100 | {row.get('thesis', '')} | {evidence_text} | {row.get('owner', '')} |"
        )

    lines.append("")
    lines.append("## Upcoming trends (next 60 days)")
    trends = _build_upcoming_trends(
        run_date=run_date,
        external_signals=external_signals,
        ferie_trends=upcoming_ferie_trends,
        additional_context=additional_context,
    )
    if trends:
        lines.append("| Date | Trend | Business/SEO impact | Suggested action |")
        lines.append("|---|---|---|---|")
        for trend_date, title, impact, action in trends:
            lines.append(f"| {trend_date} | {title} | {impact} | {action} |")
    else:
        lines.append("- No major upcoming trend detected in the next 60 days.")

    lines.append("")
    lines.append("## Query filtering")
    if query_filter_stats:
        lines.append("| Scope | Dropped current | Dropped previous | Dropped YoY |")
        lines.append("|---|---:|---:|---:|")
        for scope_name, stats in query_filter_stats.items():
            lines.append(
                f"| {scope_name} | {stats.get('current', 0)} | {stats.get('previous', 0)} | {stats.get('yoy', 0)} |"
            )
    else:
        lines.append("- Query filtering disabled or no query-based scopes.")

    lines.append("")
    lines.append("## Segment diagnostics")

    def _segment_table(
        title: str,
        rows: list[dict[str, float | str]] | None,
    ) -> None:
        lines.append(f"### {title}")
        lines.append("| Segment | Current clicks | Previous clicks | Delta vs prev | Delta vs YoY | CTR current | Pos current |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        if not rows:
            lines.append("| (no data) | 0 | 0 | +0 (0.00%) | +0 (0.00%) | 0.00% | 0.00 |")
            return
        for row in rows:
            lines.append(
                f"| {row.get('segment', '')} | {_fmt_int(row.get('current_clicks', 0.0))} | {_fmt_int(row.get('previous_clicks', 0.0))} | {_fmt_signed_int(row.get('delta_vs_previous', 0.0))} ({_signed_pct(float(row.get('delta_pct_vs_previous', 0.0)))}) | {_fmt_signed_int(row.get('delta_vs_yoy', 0.0))} ({_signed_pct(float(row.get('delta_pct_vs_yoy', 0.0)))}) | {_pct(float(row.get('ctr_current', 0.0)))} | {float(row.get('position_current', 0.0)):.2f} |"
            )

    _segment_table(
        "Brand vs non-brand",
        (segment_diagnostics or {}).get("brand_non_brand"),
    )
    lines.append("")
    _segment_table(
        "By device",
        (segment_diagnostics or {}).get("device"),
    )
    lines.append("")
    _segment_table(
        "By page name",
        (segment_diagnostics or {}).get("page_template"),
    )

    lines.append("")
    lines.append("## Further analysis flags")
    flags = _follow_up_flags(
        totals=totals,
        segment_diagnostics=segment_diagnostics,
        external_signals=external_signals,
        weather_summary=weather_summary,
    )
    if flags:
        for flag in flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- No high-priority follow-up flags triggered by current thresholds.")

    lines.append("")
    lines.append("## Findings")
    finding_rows = _flatten_findings(scope_results)
    if finding_rows:
        for scope_name, finding in finding_rows:
            lines.append(
                f"- [{finding.severity.upper()}] ({scope_name}) {finding.title}: {finding.details} Action: {finding.recommendation}"
            )
    else:
        lines.append("- No critical findings from automated thresholds.")

    lines.append("")
    lines.append("## External signals (deduped, quality-scored)")
    quality_rows = _signal_quality_rows(external_signals, limit=12)
    lines.append("| Date | Signal theme | Source coverage | Severity | Source quality |")
    lines.append("|---|---|---:|---|---|")
    if quality_rows:
        for row in quality_rows:
            day = row.get("date")
            day_label = day.isoformat() if isinstance(day, date) else str(day or "")
            lines.append(
                f"| {day_label} | {row.get('title', '')} | {int(row.get('source_count', 0))} sources / {int(row.get('mentions', 0))} mentions | {str(row.get('severity', 'info')).upper()} | {str(row.get('quality', 'low')).upper()} |"
            )
    else:
        lines.append("| - | No external signals detected in this run | 0 | INFO | LOW |")

    lines.append("")
    lines.append("### News impact by GMV category")
    category_rows = _news_category_impact_rows(external_signals, limit=10)
    lines.append("| Category | Signals | Weighted impact | Latest date | Example headlines |")
    lines.append("|---|---:|---:|---|---|")
    if category_rows:
        for row in category_rows:
            latest_day = row.get("latest_day")
            latest_label = latest_day.isoformat() if isinstance(latest_day, date) else "-"
            examples = row.get("examples", [])
            if isinstance(examples, list):
                examples_text = "; ".join(str(item).strip() for item in examples[:2] if str(item).strip())
            else:
                examples_text = ""
            lines.append(
                f"| {row.get('category', '')} | {_fmt_int(row.get('signals', 0))} | {float(row.get('weighted_impact', 0.0)):.1f} | {latest_label} | {examples_text or '-'} |"
            )
    else:
        lines.append("| No categorized news signals | 0 | 0.0 | - | - |")

    lines.append("")
    lines.append("### Raw external signals (appendix log)")
    if external_signals:
        for signal in external_signals[:30]:
            url_suffix = f" ({signal.url})" if signal.url else ""
            lines.append(
                f"- [{signal.severity.upper()}] {signal.day.isoformat()} | {signal.source} | {signal.title}: {signal.details}{url_suffix}"
            )
    else:
        lines.append("- No raw external signals in this run.")

    lines.append("")
    lines.append("## Weather context")
    lines.append(
        f"- Avg temp current vs previous: {weather_summary.get('avg_temp_current_c', 0.0):.1f}C vs {weather_summary.get('avg_temp_previous_c', 0.0):.1f}C"
    )
    lines.append(
        f"- Total precipitation current vs previous: {weather_summary.get('precip_current_mm', 0.0):.1f}mm vs {weather_summary.get('precip_previous_mm', 0.0):.1f}mm"
    )
    forecast_start = str(weather_summary.get("forecast_start", "")).strip()
    forecast_end = str(weather_summary.get("forecast_end", "")).strip()
    if forecast_start and forecast_end:
        lines.append(
            "- Forecast next 7 days "
            f"({forecast_start} to {forecast_end}): avg temp {float(weather_summary.get('forecast_avg_temp_c', 0.0)):+.1f}C, "
            f"range {float(weather_summary.get('forecast_min_temp_c', 0.0)):+.1f}C to {float(weather_summary.get('forecast_max_temp_c', 0.0)):+.1f}C, "
            f"total precipitation {float(weather_summary.get('forecast_precip_mm', 0.0)):.1f}mm."
        )

    lines.append("")
    lines.append("## Senuto visibility")
    if senuto_summary:
        lines.append(
            f"- Average visibility current vs previous: {senuto_summary.get('avg_current', 0.0):.3f} vs {senuto_summary.get('avg_previous', 0.0):.3f} ({senuto_summary.get('avg_delta_pct', 0.0):+.2f}%)"
        )
        lines.append(
            f"- Latest visibility current vs previous: {senuto_summary.get('latest_current', 0.0):.3f} vs {senuto_summary.get('latest_previous', 0.0):.3f} ({senuto_summary.get('latest_delta_pct', 0.0):+.2f}%)"
        )
        if "avg_yoy" in senuto_summary:
            lines.append(
                f"- Average visibility YoY: {senuto_summary.get('avg_yoy', 0.0):.3f} ({senuto_summary.get('avg_delta_vs_yoy_pct', 0.0):+.2f}% vs current)"
            )
    elif senuto_error:
        lines.append(f"- Senuto unavailable in this run: {senuto_error}")
    else:
        lines.append("- Senuto not configured.")

    for scope_name, analysis in scope_results:
        lines.append("")
        lines.append(f"## Scope: {scope_name}")
        lines.extend(_scope_table("Top losers", analysis.top_losers))
        lines.append("")
        lines.extend(_scope_table("Top winners", analysis.top_winners))

    return "\n".join(lines)


def _scope_table(title: str, rows: list[KeyDelta]) -> list[str]:
    out = [f"### {title}"]
    out.append("| Key | Current clicks | Prev clicks | Delta vs prev | Delta vs YoY | CTR current | Pos current |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    if not rows:
        out.append("| (no data) | 0 | 0 | 0 | 0 | 0.00% | 0.00 |")
        return out

    for row in rows:
        key = row.key.replace("|", "/")
        out.append(
            f"| {key[:80]} | {_fmt_int(row.current_clicks)} | {_fmt_int(row.previous_clicks)} | {_fmt_signed_int(row.click_delta_vs_previous)} ({_signed_pct(row.click_delta_pct_vs_previous)}) | {_fmt_signed_int(row.click_delta_vs_yoy)} ({_signed_pct(row.click_delta_pct_vs_yoy)}) | {_pct(row.current_ctr)} | {row.current_position:.2f} |"
        )
    return out


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_markdown_separator(line: str) -> bool:
    compact = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return compact == ""


def _add_markdown_runs(paragraph, text: str) -> None:
    def _append_colored_run(value: str, *, bold: bool) -> None:
        last = 0
        for signed in SIGNED_VALUE_RE.finditer(value):
            start, end = signed.span()
            if start > last:
                base = paragraph.add_run(value[last:start])
                base.bold = bold
            token = signed.group(1)
            run = paragraph.add_run(token)
            run.bold = bold
            run.font.color.rgb = DARK_GREEN if token.startswith("+") else DARK_RED
            last = end
        if last < len(value):
            tail = paragraph.add_run(value[last:])
            tail.bold = bold

    last = 0
    for match in BOLD_MARKDOWN_RE.finditer(text):
        start, end = match.span()
        if start > last:
            _append_colored_run(text[last:start], bold=False)

        _append_colored_run(match.group(1), bold=True)
        last = end

    if last < len(text):
        _append_colored_run(text[last:], bold=False)


def _set_cell_markdown(cell, text: str) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    _add_markdown_runs(paragraph, text)


def _resolve_style_name(doc: Document, style_candidates: list[str], fallback: str = "Normal") -> str:
    for name in style_candidates:
        if not name:
            continue
        try:
            _ = doc.styles[name]
            return name
        except KeyError:
            continue
    return fallback


def _list_level(raw_line: str) -> int:
    leading = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
    spaces = leading.count(" ") + (leading.count("\t") * 4)
    return min(4, max(0, spaces // 2))


def _apply_paragraph_spacing(paragraph, *, compact: bool = False) -> None:
    pf = paragraph.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(3 if compact else 6)
    pf.line_spacing = 1.15


def write_docx(path: Path, title: str, content: str) -> None:
    doc = Document()
    doc.core_properties.title = title
    try:
        normal = doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(10.5)
    except KeyError:
        pass

    lines = content.splitlines()
    index = 0

    while index < len(lines):
        raw_line = lines[index].rstrip()
        stripped_line = raw_line.lstrip(" \t")

        if not raw_line:
            doc.add_paragraph("")
            index += 1
            continue

        if (
            raw_line.startswith("|")
            and index + 1 < len(lines)
            and lines[index + 1].startswith("|")
            and _is_markdown_separator(lines[index + 1])
        ):
            headers = _split_markdown_row(raw_line)
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = "Table Grid"
            for col, header in enumerate(headers):
                _set_cell_markdown(table.rows[0].cells[col], header)

            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                row_cells = _split_markdown_row(lines[index])
                row = table.add_row().cells
                for col in range(len(headers)):
                    value = row_cells[col] if col < len(row_cells) else ""
                    _set_cell_markdown(row[col], value)
                index += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", raw_line)
        if heading_match:
            level = min(len(heading_match.group(1)), 4)
            paragraph = doc.add_heading("", level=level)
            _add_markdown_runs(paragraph, heading_match.group(2))
            _apply_paragraph_spacing(paragraph, compact=True)
        elif re.match(r"^\d+\.\s+", stripped_line):
            level = _list_level(raw_line)
            style_name = _resolve_style_name(
                doc,
                [f"List Number {level + 1}", "List Number"],
                fallback="Normal",
            )
            paragraph = doc.add_paragraph("", style=style_name)
            _add_markdown_runs(paragraph, re.sub(r"^\d+\.\s+", "", stripped_line))
            _apply_paragraph_spacing(paragraph, compact=True)
        elif stripped_line.startswith("- "):
            level = _list_level(raw_line)
            style_name = _resolve_style_name(
                doc,
                [f"List Bullet {level + 1}", "List Bullet"],
                fallback="Normal",
            )
            paragraph = doc.add_paragraph("", style=style_name)
            _add_markdown_runs(paragraph, stripped_line[2:])
            _apply_paragraph_spacing(paragraph, compact=True)
        else:
            paragraph = doc.add_paragraph("")
            _add_markdown_runs(paragraph, raw_line)
            _apply_paragraph_spacing(paragraph, compact=False)

        index += 1

    doc.save(path)
