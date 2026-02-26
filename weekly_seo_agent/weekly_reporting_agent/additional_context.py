from __future__ import annotations

import html
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus, urlparse

import requests

from weekly_seo_agent.weekly_reporting_agent.clients.continuity_client import ContinuityClient
from weekly_seo_agent.weekly_reporting_agent.clients.seo_presentations_client import SEOPresentationsClient
from weekly_seo_agent.weekly_reporting_agent.models import DateWindow, ExternalSignal

COUNTRY_MARKET_META: dict[str, dict[str, str]] = {
    "PL": {
        "country_terms": "Poland OR Polska",
        "hl": "pl",
        "gl": "PL",
        "ceid": "PL:pl",
    },
    "CZ": {
        "country_terms": "Czechia OR Cesko",
        "hl": "cs",
        "gl": "CZ",
        "ceid": "CZ:cs",
    },
    "SK": {
        "country_terms": "Slovakia OR Slovensko",
        "hl": "sk",
        "gl": "SK",
        "ceid": "SK:sk",
    },
    "HU": {
        "country_terms": "Hungary OR Magyarorszag",
        "hl": "hu",
        "gl": "HU",
        "ceid": "HU:hu",
    },
}

COUNTRY_ISO3_MAP = {
    "PL": "POL",
    "CZ": "CZE",
    "SK": "SVK",
    "HU": "HUN",
}

DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
GOOGLE_TRENDS_BRAND_CACHE_TTL_SEC = 24 * 3600
GOOGLE_TRENDS_BRAND_THROTTLE_SEC = 0.9

FREE_PUBLIC_CITY_COORDS: dict[str, tuple[float, float]] = {
    "PL": (52.2297, 21.0122),  # Warsaw
    "CZ": (50.0755, 14.4378),  # Prague
    "SK": (48.1486, 17.1077),  # Bratislava
    "HU": (47.4979, 19.0402),  # Budapest
}

FREE_PUBLIC_RSS_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("Google Search Central Blog", "https://developers.google.com/search/blog/rss.xml", "high"),
    ("Search Engine Roundtable", "https://www.seroundtable.com/index.xml", "medium"),
    ("Search Engine Journal", "https://www.searchenginejournal.com/feed/", "medium"),
    ("Search Engine Land", "https://searchengineland.com/feed", "medium"),
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews", "medium"),
    ("Eurostat News", "https://ec.europa.eu/eurostat/web/main/news/rss", "info"),
    ("OECD Newsroom", "https://www.oecd.org/newsroom/rss.xml", "info"),
    ("ECB Press", "https://www.ecb.europa.eu/rss/press.html", "info"),
    ("Wikidata Status", "https://www.wikidata.org/w/index.php?title=Special:RecentChanges&feed=rss", "info"),
)

CAMPAIGN_KEYWORDS = (
    "allegro days",
    "allegro day",
    "dni allegro",
    "smart week",
    "smart! week",
    "megaraty",
    "black week",
    "black friday",
    "cyber monday",
    "sale",
    "promocj",
    "wyprzedaz",
    "prime day",
    "deal",
    "rabat",
    "kupon",
)

COMPETITOR_KEYWORDS = (
    "temu",
    "amazon",
    "aliexpress",
    "shein",
    "ceneo",
    "olx",
    "empik",
    "x-kom",
    "media expert",
    "rtv euro agd",
    "morele",
)

MARKET_EVENT_KEYWORDS = (
    "sale",
    "promotion",
    "discount",
    "black friday",
    "cyber monday",
    "smart week",
    "allegro days",
    "megaraty",
    "logistics",
    "delivery",
    "courier",
    "warehouse",
    "strike",
    "outage",
    "flood",
    "storm",
    "snow",
    "regulation",
    "law",
    "tax",
    "vat",
    "tariff",
    "inflation",
    "interest rate",
    "unemployment",
    "consumer confidence",
    "holiday",
    "easter",
    "valentine",
    "back to school",
    "christmas",
    "ecommerce",
    "e-commerce",
    "marketplace",
    "retail",
    "allegro",
    "temu",
    "amazon",
    "shein",
    "ceneo",
    "olx",
)

POSITIVE_EVENT_TOKENS = (
    "sale",
    "promotion",
    "discount",
    "campaign",
    "shopping festival",
    "holiday shopping",
    "consumer confidence rises",
    "wages rise",
    "pay increase",
    "bonus",
)

NEGATIVE_EVENT_TOKENS = (
    "strike",
    "outage",
    "flood",
    "storm",
    "snowstorm",
    "road closure",
    "delivery disruption",
    "warehouse fire",
    "cyber attack",
    "inflation rises",
    "tax increase",
    "vat increase",
    "tariff",
    "ban",
    "recession",
)

GOOGLE_UPDATE_TIMELINE_RSS_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("Google Search Central Blog", "https://developers.google.com/search/blog/rss.xml", "official"),
    ("Search Engine Roundtable", "https://www.seroundtable.com/index.xml", "industry"),
    ("Search Engine Land", "https://searchengineland.com/feed", "industry"),
    ("Search Engine Journal", "https://www.searchenginejournal.com/feed/", "industry"),
)

GOOGLE_UPDATE_TIMELINE_KEYWORDS = (
    "update",
    "core update",
    "spam update",
    "ranking volatility",
    "algorithm",
    "search console",
    "discover",
    "ai overview",
    "merchant listings",
    "product snippets",
)

SERP_CASE_STUDY_RSS_SOURCES: tuple[tuple[str, str], ...] = (
    ("Google Search Central Blog", "https://developers.google.com/search/blog/rss.xml"),
    ("Search Engine Roundtable", "https://www.seroundtable.com/index.xml"),
    ("Search Engine Land", "https://searchengineland.com/feed"),
    ("Search Engine Journal", "https://www.searchenginejournal.com/feed/"),
)

SERP_CASE_STUDY_INTENT_TOKENS = (
    "case study",
    "study",
    "analysis",
    "test",
    "experiment",
    "data",
)

SERP_CASE_STUDY_SIGNAL_TOKENS = (
    "ctr",
    "click-through",
    "click share",
    "search appearance",
    "serp feature",
    "merchant listings",
    "product snippets",
    "free listings",
    "shopping graph",
    "organic clicks",
    "ai overview",
    "ai mode",
    "rich result",
    "featured snippet",
)

SERP_CASE_TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "CTR and click distribution",
        ("ctr", "click-through", "click share", "traffic share", "organic clicks"),
    ),
    (
        "SERP layout and features",
        ("search appearance", "serp feature", "rich result", "featured snippet", "carousel", "ai overview", "ai mode"),
    ),
    (
        "Merchant and free listings",
        ("merchant listings", "product snippets", "free listings", "shopping graph", "merchant center"),
    ),
    (
        "Algorithm/update impact",
        ("core update", "spam update", "ranking volatility", "algorithm update", "search update"),
    ),
)


def _safe_pct(current: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if current > 0 else 0.0
    return (current - baseline) / baseline


def _normalize_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


def _detect_campaign_severity(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("black friday", "cyber monday", "black week")):
        return "high"
    if any(token in lowered for token in ("smart week", "allegro days", "megaraty", "prime day")):
        return "medium"
    return "info"


def _market_meta(country_code: str) -> dict[str, str]:
    code = country_code.strip().upper() or "PL"
    return COUNTRY_MARKET_META.get(code, COUNTRY_MARKET_META["PL"])


def _campaign_rss_trackers(country_code: str) -> tuple[tuple[str, str], ...]:
    market = _market_meta(country_code)
    hl = market["hl"]
    gl = market["gl"]
    ceid = market["ceid"]
    country_terms = market["country_terms"]

    allegro_query = (
        f"(({country_terms}) AND (\"Allegro Days\" OR \"Smart Week\" OR \"Megaraty\" OR "
        "\"Allegro Black Week\" OR \"Allegro Black Friday\" OR sale OR promotion OR promocja OR wyprzedaz))"
    )
    competitors_query = (
        f"(({country_terms}) AND (Temu OR Amazon OR AliExpress OR SHEIN OR Ceneo OR OLX) AND "
        "(\"Black Week\" OR \"Black Friday\" OR \"Cyber Monday\" OR sale OR promotion OR promocja OR wyprzedaz OR \"Prime Day\"))"
    )
    allegro_global_query = (
        "\"Allegro Days\" OR \"Allegro Day\" OR \"Smart Week\" OR Megaraty OR \"Dni Allegro\""
    )
    return (
        (
            f"Campaign tracker Allegro ({country_code.strip().upper() or 'PL'})",
            "https://news.google.com/rss/search?q="
            + quote_plus(allegro_query)
            + f"&hl={hl}&gl={gl}&ceid={ceid}",
        ),
        (
            f"Campaign tracker Competitors ({country_code.strip().upper() or 'PL'})",
            "https://news.google.com/rss/search?q="
            + quote_plus(competitors_query)
            + f"&hl={hl}&gl={gl}&ceid={ceid}",
        ),
        (
            f"Campaign tracker Allegro fallback ({country_code.strip().upper() or 'PL'})",
            "https://news.google.com/rss/search?q="
            + quote_plus(allegro_global_query)
            + f"&hl={hl}&gl={gl}&ceid={ceid}",
        ),
    )


def _fetch_campaign_tracker_signals(
    since: date,
    run_date: date,
    country_code: str = "PL",
) -> list[ExternalSignal]:
    signals: list[ExternalSignal] = []
    seen: set[tuple[str, str]] = set()

    for label, url in _campaign_rss_trackers(country_code):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except Exception:
            continue

        for item in root.findall(".//item"):
            title = html.unescape((item.findtext("title") or "").strip())
            link = (item.findtext("link") or "").strip()
            description = html.unescape((item.findtext("description") or "").strip())
            pub_raw = (item.findtext("pubDate") or item.findtext("published") or "").strip()
            pub_dt = _parse_datetime(pub_raw)
            day = pub_dt.date() if pub_dt else run_date
            if day < since:
                continue

            text = f"{title} {description}".lower()
            if not any(token in text for token in CAMPAIGN_KEYWORDS):
                continue
            if "competitor" in label.lower() and not any(
                token in text for token in COMPETITOR_KEYWORDS
            ):
                continue

            details = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", description)).strip()
            if len(details) > 260:
                details = details[:257] + "..."
            if not details:
                details = "Campaign-related marketplace signal from monitored external source."

            source_host = urlparse(url).netloc or "news"
            source = f"{label} ({source_host})"
            canonical_title = re.sub(r"\s*[-|:]\s*[^-|:]{1,80}$", "", title).strip().lower() or title.lower()
            dedupe_key = (canonical_title, day.isoformat())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            signals.append(
                ExternalSignal(
                    source=source,
                    day=day,
                    title=title or "Campaign event mention",
                    details=details,
                    severity=_detect_campaign_severity(f"{title} {details}"),
                    url=link or None,
                )
            )

    signals.sort(key=lambda row: (row.day, row.source), reverse=True)
    return signals[:24]


def _fetch_google_news_query_signals(
    *,
    country_code: str,
    query: str,
    source_label: str,
    since: date,
    max_rows: int = 12,
    severity: str = "medium",
) -> list[ExternalSignal]:
    market = _market_meta(country_code)
    hl = market["hl"]
    gl = market["gl"]
    ceid = market["ceid"]
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + f"&hl={hl}&gl={gl}&ceid={ceid}"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception:
        return []

    out: list[ExternalSignal] = []
    seen: set[str] = set()
    for item in root.findall(".//item"):
        title = html.unescape((item.findtext("title") or "").strip())
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        link = (item.findtext("link") or "").strip()
        description = html.unescape((item.findtext("description") or "").strip())
        pub_raw = (item.findtext("pubDate") or item.findtext("published") or "").strip()
        pub_dt = _parse_datetime(pub_raw)
        day = pub_dt.date() if pub_dt else date.today()
        if day < since:
            continue
        out.append(
            ExternalSignal(
                source=source_label,
                day=day,
                title=title,
                details=re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", description)).strip()[:240],
                severity=severity,
                url=link or None,
            )
        )
        if len(out) >= max_rows:
            break
    return out


def _fetch_generic_rss_signals(
    *,
    url: str,
    source_label: str,
    since: date,
    max_rows: int = 12,
    severity: str = "medium",
) -> list[ExternalSignal]:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception:
        return []

    out: list[ExternalSignal] = []
    seen: set[str] = set()
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//entry")
    for item in items:
        title = html.unescape((item.findtext("title") or "").strip())
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        link = (item.findtext("link") or "").strip()
        if not link:
            link_elem = item.find("link")
            if link_elem is not None:
                link = str(link_elem.attrib.get("href", "")).strip()
        description = html.unescape(
            (
                item.findtext("description")
                or item.findtext("summary")
                or item.findtext("{http://www.w3.org/2005/Atom}summary")
                or ""
            ).strip()
        )
        pub_raw = (
            item.findtext("pubDate")
            or item.findtext("published")
            or item.findtext("updated")
            or item.findtext("{http://www.w3.org/2005/Atom}updated")
            or ""
        ).strip()
        pub_dt = _parse_datetime(pub_raw)
        day = pub_dt.date() if pub_dt else date.today()
        if day < since:
            continue
        out.append(
            ExternalSignal(
                source=source_label,
                day=day,
                title=title[:180],
                details=re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", description)).strip()[:240],
                severity=severity,
                url=link or None,
            )
        )
        if len(out) >= max_rows:
            break
    return out


def _fetch_platform_regulatory_pulse(
    *,
    country_code: str,
    since: date,
    run_date: date,
    rss_urls: tuple[str, ...],
    top_rows: int,
) -> list[ExternalSignal]:
    market = _market_meta(country_code)
    country_terms = market["country_terms"]
    rows: list[ExternalSignal] = []

    queries = (
        (
            "Platform pulse (Allegro/competitors)",
            f"(({country_terms}) AND (Allegro OR Temu OR Amazon OR Shein OR Ceneo OR OLX) "
            "(platform change OR launch OR campaign OR promotion OR pricing OR delivery OR logistics))",
            "medium",
        ),
        (
            "Regulatory pulse (EU/local)",
            f"(({country_terms}) AND (EU regulation OR DSA OR DMA OR VAT OR tax OR e-commerce law "
            "OR consumer law OR customs OR tariff OR compliance))",
            "high",
        ),
        (
            "Marketplace pulse (new entrants)",
            f"(({country_terms}) AND (new marketplace OR new platform OR cross-border marketplace "
            "OR retail platform launch))",
            "medium",
        ),
    )
    for source_label, query, severity in queries:
        rows.extend(
            _fetch_google_news_query_signals(
                country_code=country_code,
                query=query,
                source_label=f"{source_label} ({country_code})",
                since=since,
                max_rows=max(4, top_rows // 2),
                severity=severity,
            )
        )

    for url in rss_urls:
        host = urlparse(url).netloc or "rss"
        rows.extend(
            _fetch_generic_rss_signals(
                url=url,
                source_label=f"Regulatory RSS ({host})",
                since=since,
                max_rows=max(2, top_rows // 3),
                severity="medium",
            )
        )

    deduped: list[ExternalSignal] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: (item.day, item.source), reverse=True):
        canonical = re.sub(r"\s*[-|:]\s*[^-|:]{1,80}$", "", row.title.lower()).strip() or row.title.lower()
        key = (canonical, row.day.isoformat())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= max(1, top_rows):
            break
    return deduped


def _fetch_macro_backdrop(country_code: str) -> dict[str, Any]:
    iso3 = COUNTRY_ISO3_MAP.get(country_code.strip().upper(), "")
    if not iso3:
        return {"country_code": country_code, "note": "No ISO3 mapping for macro backdrop.", "rows": {}, "errors": []}

    indicators = {
        "inflation_cpi_pct": "FP.CPI.TOTL.ZG",
        "unemployment_pct": "SL.UEM.TOTL.ZS",
        "gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    }
    out: dict[str, Any] = {
        "country_code": country_code,
        "source": "World Bank API",
        "rows": {},
        "errors": [],
    }
    for key, indicator in indicators.items():
        url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
        try:
            response = requests.get(url, params={"format": "json", "per_page": "8"}, timeout=30)
            response.raise_for_status()
            payload = response.json()
            rows = payload[1] if isinstance(payload, list) and len(payload) >= 2 else []
            if not isinstance(rows, list):
                rows = []
            usable = [
                row for row in rows
                if isinstance(row, dict) and row.get("value") is not None
            ]
            usable.sort(key=lambda row: str(row.get("date", "")), reverse=True)
            latest = usable[0] if usable else None
            previous = usable[1] if len(usable) > 1 else None
            out["rows"][key] = {
                "latest_year": str((latest or {}).get("date", "")),
                "latest_value": _parse_float((latest or {}).get("value")),
                "previous_year": str((previous or {}).get("date", "")),
                "previous_value": _parse_float((previous or {}).get("value")),
            }
        except Exception as exc:
            out.setdefault("errors", []).append(f"{key}: {exc}")
    return out


def _safe_get_json(
    *,
    url: str,
    params: dict[str, object] | None = None,
    timeout: int = 25,
    headers: dict[str, str] | None = None,
) -> Any:
    response = requests.get(url, params=params or {}, timeout=timeout, headers=headers or {})
    response.raise_for_status()
    return response.json()


def _collect_free_public_rss_signals(
    *,
    since: date,
    top_rows: int,
) -> tuple[list[dict[str, str]], list[ExternalSignal]]:
    rows: list[dict[str, str]] = []
    signals: list[ExternalSignal] = []
    for label, url, severity in FREE_PUBLIC_RSS_SOURCES:
        items = _fetch_generic_rss_signals(
            url=url,
            source_label=f"Free RSS: {label}",
            since=since,
            max_rows=max(1, top_rows),
            severity=severity,
        )
        rows.append(
            {
                "source": label,
                "type": "rss",
                "status": "ok" if items else "empty",
                "details": f"rows={len(items)}",
                "url": url,
            }
        )
        signals.extend(items[: max(1, min(3, top_rows))])
    return rows, signals


def _collect_free_public_api_rows(
    *,
    country_code: str,
    run_date: date,
    current_window: DateWindow,
    previous_window: DateWindow,
    target_domain: str,
    nager_country_code: str,
    eia_api_key: str,
) -> tuple[list[dict[str, str]], list[ExternalSignal]]:
    rows: list[dict[str, str]] = []
    signals: list[ExternalSignal] = []
    coords = FREE_PUBLIC_CITY_COORDS.get(country_code.strip().upper(), FREE_PUBLIC_CITY_COORDS["PL"])

    # 1) Nager.Date public holidays
    try:
        holidays_url = f"https://date.nager.at/api/v3/PublicHolidays/{run_date.year}/{nager_country_code}"
        payload = _safe_get_json(url=holidays_url, timeout=25)
        count = len(payload) if isinstance(payload, list) else 0
        rows.append(
            {
                "source": "Nager.Date Holidays API",
                "type": "api",
                "status": "ok",
                "details": f"year={run_date.year}; rows={count}",
                "url": holidays_url,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "Nager.Date Holidays API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://date.nager.at/Api",
            }
        )

    # 2) Frankfurter FX
    try:
        fx = _safe_get_json(
            url="https://api.frankfurter.app/latest",
            params={"from": "EUR", "to": "PLN,CZK,HUF"},
        )
        rates = fx.get("rates", {}) if isinstance(fx, dict) else {}
        rows.append(
            {
                "source": "Frankfurter FX API",
                "type": "api",
                "status": "ok",
                "details": f"rates={','.join(sorted(str(k) for k in rates.keys()))}" if isinstance(rates, dict) else "rates=n/a",
                "url": "https://www.frankfurter.app/docs/",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "Frankfurter FX API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://www.frankfurter.app/docs/",
            }
        )

    # 3) Open-Meteo forecast
    try:
        weather = _safe_get_json(
            url="https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords[0],
                "longitude": coords[1],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 7,
                "timezone": "UTC",
            },
        )
        daily = weather.get("daily", {}) if isinstance(weather, dict) else {}
        days = len(daily.get("time", [])) if isinstance(daily, dict) and isinstance(daily.get("time", []), list) else 0
        rows.append(
            {
                "source": "Open-Meteo API",
                "type": "api",
                "status": "ok",
                "details": f"forecast_days={days}",
                "url": "https://open-meteo.com/en/docs",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "Open-Meteo API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://open-meteo.com/en/docs",
            }
        )

    # 4) OpenAQ latest (public)
    try:
        openaq = _safe_get_json(
            url="https://api.openaq.org/v3/latest",
            params={"limit": 5, "coordinates": f"{coords[0]},{coords[1]}", "radius": 25000},
            headers={"X-API-Key": ""},
        )
        results = openaq.get("results", []) if isinstance(openaq, dict) else []
        rows.append(
            {
                "source": "OpenAQ API",
                "type": "api",
                "status": "ok",
                "details": f"results={len(results) if isinstance(results, list) else 0}",
                "url": "https://docs.openaq.org/",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "OpenAQ API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://docs.openaq.org/",
            }
        )

    # 5) Wikimedia pageviews
    try:
        project = "en.wikipedia.org"
        article = "Allegro"
        start = previous_window.start.strftime("%Y%m%d") + "00"
        end = current_window.end.strftime("%Y%m%d") + "00"
        pageviews_url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{project}/all-access/user/{article}/daily/{start}/{end}"
        )
        pageviews = _safe_get_json(url=pageviews_url)
        items = pageviews.get("items", []) if isinstance(pageviews, dict) else []
        total = 0
        if isinstance(items, list):
            for row in items:
                if isinstance(row, dict):
                    total += int(row.get("views", 0) or 0)
        rows.append(
            {
                "source": "Wikimedia Pageviews API",
                "type": "api",
                "status": "ok",
                "details": f"article={article}; total_views={total}",
                "url": "https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews",
            }
        )
        if total > 0:
            signals.append(
                ExternalSignal(
                    source=f"Wikimedia Pageviews ({country_code})",
                    day=run_date,
                    title="Public interest proxy: Allegro pageviews",
                    details=f"Wikipedia pageviews in analysis window: {total}.",
                    severity="info",
                    url="https://www.wikidata.org/wiki/Q78131",
                )
            )
    except Exception as exc:
        rows.append(
            {
                "source": "Wikimedia Pageviews API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews",
            }
        )

    # 6) Wikidata entity lookup
    try:
        wikidata_url = "https://www.wikidata.org/w/api.php"
        payload = _safe_get_json(
            url=wikidata_url,
            params={"action": "wbsearchentities", "search": target_domain.split(".")[0], "language": "en", "format": "json", "limit": 3},
        )
        search_rows = payload.get("search", []) if isinstance(payload, dict) else []
        rows.append(
            {
                "source": "Wikidata API",
                "type": "api",
                "status": "ok",
                "details": f"entities={len(search_rows) if isinstance(search_rows, list) else 0}",
                "url": "https://www.wikidata.org/w/api.php",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "Wikidata API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://www.wikidata.org/w/api.php",
            }
        )

    # 7) Eurostat SDMX (simple availability check)
    try:
        eurostat_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/"
        response = requests.get(eurostat_url, timeout=20)
        response.raise_for_status()
        rows.append(
            {
                "source": "Eurostat SDMX API",
                "type": "api",
                "status": "ok",
                "details": "endpoint reachable",
                "url": eurostat_url,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "Eurostat SDMX API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/",
            }
        )

    # 8) OECD SDMX API
    try:
        oecd_url = "https://sdmx.oecd.org/public/rest/dataflow"
        response = requests.get(oecd_url, timeout=20)
        response.raise_for_status()
        rows.append(
            {
                "source": "OECD SDMX API",
                "type": "api",
                "status": "ok",
                "details": "endpoint reachable",
                "url": oecd_url,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "OECD SDMX API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://sdmx.oecd.org/public/rest/",
            }
        )

    # 9) OpenSky API
    try:
        states = _safe_get_json(url="https://opensky-network.org/api/states/all", timeout=20)
        total_states = len(states.get("states", [])) if isinstance(states, dict) and isinstance(states.get("states", []), list) else 0
        rows.append(
            {
                "source": "OpenSky Network API",
                "type": "api",
                "status": "ok",
                "details": f"states={total_states}",
                "url": "https://openskynetwork.github.io/opensky-api/rest.html",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "source": "OpenSky Network API",
                "type": "api",
                "status": "error",
                "details": str(exc)[:180],
                "url": "https://openskynetwork.github.io/opensky-api/rest.html",
            }
        )

    # 10) EIA Open Data (optional API key)
    if eia_api_key.strip():
        try:
            eia_url = "https://api.eia.gov/v2/total-energy/data/"
            eia_payload = _safe_get_json(
                url=eia_url,
                params={"api_key": eia_api_key, "frequency": "monthly", "data[0]": "value", "length": 1},
                timeout=25,
            )
            data_rows = (((eia_payload or {}).get("response", {}) if isinstance(eia_payload, dict) else {}).get("data", []))
            rows.append(
                {
                    "source": "EIA Open Data API",
                    "type": "api",
                    "status": "ok",
                    "details": f"rows={len(data_rows) if isinstance(data_rows, list) else 0}",
                    "url": "https://www.eia.gov/opendata/",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source": "EIA Open Data API",
                    "type": "api",
                    "status": "error",
                    "details": str(exc)[:180],
                    "url": "https://www.eia.gov/opendata/",
                }
            )
    else:
        rows.append(
            {
                "source": "EIA Open Data API",
                "type": "api",
                "status": "skipped",
                "details": "missing EIA_API_KEY",
                "url": "https://www.eia.gov/opendata/",
            }
        )

    return rows, signals


def _collect_free_public_source_hub(
    *,
    country_code: str,
    run_date: date,
    current_window: DateWindow,
    previous_window: DateWindow,
    target_domain: str,
    enabled: bool,
    top_rows_per_source: int,
    nager_country_code: str,
    eia_api_key: str,
) -> tuple[dict[str, Any], list[ExternalSignal]]:
    if not enabled:
        return {
            "enabled": False,
            "source": "Free public source hub",
            "rows": [],
            "errors": [],
            "note": "Disabled by configuration.",
        }, []

    rows: list[dict[str, str]] = []
    signals: list[ExternalSignal] = []
    errors: list[str] = []

    try:
        rss_rows, rss_signals = _collect_free_public_rss_signals(
            since=previous_window.start,
            top_rows=max(1, top_rows_per_source),
        )
        rows.extend(rss_rows)
        signals.extend(rss_signals)
    except Exception as exc:
        errors.append(f"RSS group failed: {exc}")

    try:
        api_rows, api_signals = _collect_free_public_api_rows(
            country_code=country_code,
            run_date=run_date,
            current_window=current_window,
            previous_window=previous_window,
            target_domain=target_domain,
            nager_country_code=nager_country_code,
            eia_api_key=eia_api_key,
        )
        rows.extend(api_rows)
        signals.extend(api_signals)
    except Exception as exc:
        errors.append(f"API group failed: {exc}")

    # Explicitly track sources already integrated elsewhere in pipeline,
    # so this hub represents full 20-source inventory in one place.
    rows.extend(
        [
            {"source": "Google News RSS queries", "type": "native", "status": "integrated", "details": "campaign + competitor + operational", "url": "https://news.google.com/rss"},
            {"source": "GDELT DOC API", "type": "native", "status": "integrated", "details": "market_event_calendar", "url": "https://api.gdeltproject.org/api/v2/doc/doc"},
            {"source": "OpenHolidays API", "type": "native", "status": "integrated", "details": "public holidays + school breaks", "url": "https://openholidaysapi.org"},
            {"source": "NBP API", "type": "native", "status": "integrated", "details": "fx in macro context", "url": "https://api.nbp.pl/api"},
            {"source": "World Bank API", "type": "native", "status": "integrated", "details": "macro_backdrop", "url": "https://api.worldbank.org"},
        ]
    )

    return {
        "enabled": True,
        "source": "Free public source hub",
        "country_code": country_code,
        "rows": rows[:80],
        "errors": errors,
        "top_rows_per_source": max(1, top_rows_per_source),
    }, signals[:40]


def _market_event_type(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("black friday", "cyber monday", "sale", "promotion", "discount", "smart week", "allegro days", "megaraty")):
        return "Campaign/Promotions"
    if any(token in lowered for token in ("logistics", "delivery", "courier", "warehouse", "parcel", "strike", "port", "rail")):
        return "Logistics/Delivery"
    if any(token in lowered for token in ("regulation", "law", "tax", "vat", "tariff", "customs", "compliance")):
        return "Regulation/Tax"
    if any(token in lowered for token in ("inflation", "interest rate", "unemployment", "consumer confidence", "wages", "salary", "gdp")):
        return "Macro/Consumer"
    if any(token in lowered for token in ("flood", "storm", "snow", "heatwave", "cold wave", "weather warning")):
        return "Weather disruption"
    if any(token in lowered for token in ("holiday", "easter", "valentine", "christmas", "back to school")):
        return "Calendar/Shopping intent"
    if any(token in lowered for token in ("allegro", "temu", "amazon", "shein", "ceneo", "olx")):
        return "Marketplace competition"
    return "Market signal"


def _gmv_impact_from_text(text: str) -> tuple[str, str, int, str]:
    lowered = text.lower()
    positive_hits = sum(1 for token in POSITIVE_EVENT_TOKENS if token in lowered)
    negative_hits = sum(1 for token in NEGATIVE_EVENT_TOKENS if token in lowered)
    score = abs(positive_hits - negative_hits)
    confidence = min(95, 50 + score * 9)

    if negative_hits > positive_hits:
        level = "HIGH" if score >= 2 else "MEDIUM"
        direction = "Downside risk"
        reason = "Potential demand/logistics friction or purchasing-power pressure."
    elif positive_hits > negative_hits:
        level = "HIGH" if score >= 2 else "MEDIUM"
        direction = "Upside potential"
        reason = "Likely demand uplift from campaigns or shopping-intent catalysts."
    else:
        level = "MEDIUM" if (positive_hits or negative_hits) else "LOW"
        direction = "Mixed"
        reason = "Signal may influence GMV timing, but direction is not clear yet."
    return level, direction, confidence, reason


def _fetch_market_event_calendar(
    country_code: str,
    since: date,
    until: date,
    top_rows: int,
    api_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc",
) -> list[dict[str, Any]]:
    market = _market_meta(country_code)
    country_terms = market["country_terms"]
    country_tag = country_code.strip().upper() or "PL"

    query_variants = [
        (
            f"({country_terms}) AND "
            "("
            "ecommerce OR retail OR marketplace OR logistics OR strike OR inflation OR tax OR weather OR holiday OR "
            "\"black friday\" OR \"cyber monday\" OR \"allegro days\" OR \"smart week\" OR megaraty OR allegro OR temu OR amazon OR shein"
            ")"
        ),
        (
            f"({country_terms}) AND "
            "(ecommerce OR retail OR marketplace OR logistics OR tax OR weather OR allegro OR temu OR amazon OR shein)"
        ),
        (
            f"({country_terms}) AND "
            "(ecommerce OR retail OR marketplace OR allegro OR temu OR amazon)"
        ),
    ]
    articles: list[dict[str, Any]] = []
    last_error: str = ""
    for query in query_variants:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "DateDesc",
            "maxrecords": str(max(top_rows * 6, 30)),
        }
        response = requests.get(api_base_url, params=params, timeout=35)
        response.raise_for_status()
        try:
            payload = response.json()
        except Exception:
            snippet = response.text[:180].strip()
            last_error = snippet or "Non-JSON response from market-event API."
            continue
        if isinstance(payload, dict):
            rows_raw = payload.get("articles", [])
            if isinstance(rows_raw, list):
                articles = [row for row in rows_raw if isinstance(row, dict)]
                if articles:
                    break
    if not articles and last_error:
        raise RuntimeError(last_error)
    if not isinstance(articles, list):
        return []

    rows: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title", "")).strip()
        url = str(article.get("url", "")).strip()
        domain = str(article.get("domain", "")).strip() or (urlparse(url).netloc or "")
        seen_raw = str(article.get("seendate", "")).strip()
        if len(seen_raw) >= 8:
            day = _parse_date(f"{seen_raw[0:4]}-{seen_raw[4:6]}-{seen_raw[6:8]}")
        else:
            day = _parse_date(article.get("seendate"))
        if day is None:
            day = since
        if day < since or day > until:
            continue
        if not title:
            continue
        title_key = title.lower()
        if title_key in seen_titles:
            continue

        text_blob = f"{title} {domain}".lower()
        if not any(token in text_blob for token in MARKET_EVENT_KEYWORDS):
            continue

        event_type = _market_event_type(text_blob)
        impact_level, impact_direction, confidence, reason = _gmv_impact_from_text(text_blob)
        rows.append(
            {
                "date": day.isoformat(),
                "country_code": country_tag,
                "title": title,
                "source": domain or "news",
                "url": url,
                "event_type": event_type,
                "impact_level": impact_level,
                "impact_direction": impact_direction,
                "confidence": confidence,
                "gmv_reason": reason,
            }
        )
        seen_titles.add(title_key)
        if len(rows) >= top_rows:
            break

    rows.sort(
        key=lambda row: (
            str(row.get("date", "")),
            {"HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(str(row.get("impact_level", "LOW")), 0),
        ),
        reverse=True,
    )
    return rows[:top_rows]


def _extract_approx_traffic(raw: str) -> int:
    text = raw.strip().replace(",", "").replace("+", "")
    if not text:
        return 0
    match = re.match(r"^(\d+(?:\.\d+)?)([KkMm]?)$", text)
    if not match:
        return 0
    value = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = 1
    if suffix == "k":
        multiplier = 1000
    if suffix == "m":
        multiplier = 1000000
    return int(value * multiplier)


def _avg_fx_for_window(
    rates_by_day: dict[date, float],
    window: DateWindow,
) -> float:
    values: list[float] = []
    for offset in range(window.days):
        day = window.start + timedelta(days=offset)
        if day in rates_by_day:
            values.append(rates_by_day[day])
    if not values:
        return 0.0
    return sum(values) / len(values)


def _origin_from_url(target_site_url: str) -> str:
    parsed = urlparse(target_site_url.strip())
    scheme = parsed.scheme or "https"
    host = (parsed.netloc or parsed.path).strip().rstrip("/")
    if not host:
        raise RuntimeError(f"Invalid target site URL for CrUX: {target_site_url}")
    return f"{scheme}://{host}"


def _request_json_with_retry(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: int = 40,
    max_attempts: int = 3,
    retry_sleep_sec: float = 1.2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=timeout,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(retry_sleep_sec * attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            raise RuntimeError(f"Unexpected JSON payload from {url}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_attempts:
                time.sleep(retry_sleep_sec * attempt)
                continue
            break
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def _request_text_with_retry(
    method: str,
    url: str,
    *,
    session: requests.Session | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 40,
    max_attempts: int = 4,
    retry_sleep_sec: float = 1.4,
) -> str:
    last_error: Exception | None = None
    requester = session.request if session is not None else requests.request
    for attempt in range(1, max_attempts + 1):
        try:
            response = requester(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(retry_sleep_sec * attempt)
                continue
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_attempts:
                time.sleep(retry_sleep_sec * attempt)
                continue
            break
    raise RuntimeError(f"Text request failed for {url}: {last_error}")


def _local_cache_dir() -> Path:
    path = Path(".cache/weekly_seo_agent")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_local_cache(prefix: str, key: str, max_age_sec: int) -> dict[str, Any] | None:
    digest = re.sub(r"[^a-zA-Z0-9_-]", "_", key)[:180]
    path = _local_cache_dir() / f"{prefix}_{digest}.json"
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > max_age_sec:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _save_local_cache(prefix: str, key: str, payload: dict[str, Any]) -> None:
    digest = re.sub(r"[^a-zA-Z0-9_-]", "_", key)[:180]
    path = _local_cache_dir() / f"{prefix}_{digest}.json"
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _parse_crux_p75(metrics: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        row = metrics.get(key)
        if not isinstance(row, dict):
            continue
        percentiles = row.get("percentiles")
        if not isinstance(percentiles, dict):
            continue
        value = _parse_float(percentiles.get("p75"))
        if value is None:
            continue
        if key == "cumulative_layout_shift" and value > 1.0:
            return value / 100.0
        return value
    return 0.0


def _cwv_bucket_lcp(value_ms: float) -> str:
    if value_ms <= 2500:
        return "good"
    if value_ms <= 4000:
        return "needs_improvement"
    return "poor"


def _cwv_bucket_inp(value_ms: float) -> str:
    if value_ms <= 200:
        return "good"
    if value_ms <= 500:
        return "needs_improvement"
    return "poor"


def _cwv_bucket_cls(value: float) -> str:
    if value <= 0.1:
        return "good"
    if value <= 0.25:
        return "needs_improvement"
    return "poor"


def _overall_cwv_category(lcp_ms: float, inp_ms: float, cls: float) -> str:
    bucket_rank = {"good": 0, "needs_improvement": 1, "poor": 2}
    buckets = (
        _cwv_bucket_lcp(lcp_ms),
        _cwv_bucket_inp(inp_ms),
        _cwv_bucket_cls(cls),
    )
    return max(buckets, key=lambda item: bucket_rank[item])


def _normalize_title_token(value: str) -> str:
    text = _normalize_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _classify_google_update_kind(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("core update", "spam update", "algorithm update")):
        return "Algorithm update"
    if any(token in lowered for token in ("ai overview", "ai mode", "serp feature", "search appearance")):
        return "SERP layout change"
    if any(token in lowered for token in ("merchant listings", "product snippets", "free listings", "shopping graph")):
        return "Commerce SERP/feed"
    if any(token in lowered for token in ("outage", "incident", "error")):
        return "Operational incident"
    return "Search guidance"


def _classify_serp_case_topic(text: str) -> str:
    lowered = text.lower()
    for label, tokens in SERP_CASE_TOPIC_RULES:
        if any(token in lowered for token in tokens):
            return label
    return "General SERP observations"


def _is_serp_case_candidate(text: str) -> bool:
    lowered = text.lower()
    intent_hits = sum(1 for token in SERP_CASE_STUDY_INTENT_TOKENS if token in lowered)
    signal_hits = sum(1 for token in SERP_CASE_STUDY_SIGNAL_TOKENS if token in lowered)
    if signal_hits <= 0:
        return False
    if intent_hits > 0:
        return True
    # Keep data-rich posts even when explicit "case study" wording is missing.
    return signal_hits >= 2


def _collect_google_updates_timeline(
    *,
    run_date: date,
    current_window: DateWindow,
    previous_window: DateWindow,
    yoy_window: DateWindow,
    status_endpoint: str,
    blog_rss_url: str,
    scan_months: int = 13,
) -> dict[str, Any]:
    scan_days = max(390, int(scan_months) * 30)
    since_day = run_date - timedelta(days=scan_days)
    current_month_end = current_window.end
    current_month_start = current_month_end - timedelta(days=29)
    previous_month_end = current_month_start - timedelta(days=1)
    previous_month_start = previous_month_end - timedelta(days=29)
    yoy_month_start = current_month_start - timedelta(weeks=52)
    yoy_month_end = current_month_end - timedelta(weeks=52)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        payload = _safe_get_json(url=status_endpoint, timeout=30)
        if isinstance(payload, list):
            for incident in payload:
                if not isinstance(incident, dict):
                    continue
                begin = _parse_datetime(incident.get("begin"))
                day = begin.date() if begin else None
                if not isinstance(day, date) or day < since_day:
                    continue
                title = str(incident.get("external_desc") or incident.get("id") or "Google Search incident").strip()
                if not title:
                    continue
                updates = incident.get("updates") or []
                latest_update = updates[-1] if isinstance(updates, list) and updates and isinstance(updates[-1], dict) else {}
                details = html.unescape(str(latest_update.get("text", ""))).strip()
                details = re.sub(r"\s+", " ", details)
                if len(details) > 240:
                    details = details[:237] + "..."
                text_blob = f"{title} {details}".lower()
                if "search" not in text_blob and "google" not in text_blob:
                    continue
                rows.append(
                    {
                        "date": day.isoformat(),
                        "title": title[:180],
                        "source": "Google Search Status",
                        "source_type": "official",
                        "kind": _classify_google_update_kind(text_blob),
                        "url": f"https://status.search.google.com/incidents/{incident.get('id', '')}",
                        "details": details,
                    }
                )
    except Exception as exc:
        errors.append(f"status_endpoint: {exc}")

    rss_sources = list(GOOGLE_UPDATE_TIMELINE_RSS_SOURCES)
    if blog_rss_url.strip():
        rss_sources[0] = ("Google Search Central Blog", blog_rss_url.strip(), "official")
    for label, url, source_type in rss_sources:
        try:
            items = _fetch_generic_rss_signals(
                url=url,
                source_label=label,
                since=since_day,
                max_rows=220,
                severity="info",
            )
            for item in items:
                text_blob = f"{item.title} {item.details}".lower()
                if not any(token in text_blob for token in GOOGLE_UPDATE_TIMELINE_KEYWORDS):
                    continue
                rows.append(
                    {
                        "date": item.day.isoformat(),
                        "title": str(item.title).strip()[:180],
                        "source": label,
                        "source_type": source_type,
                        "kind": _classify_google_update_kind(text_blob),
                        "url": item.url or "",
                        "details": str(item.details).strip()[:240],
                    }
                )
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: str(item.get("date", "")), reverse=True):
        title_key = _normalize_title_token(str(row.get("title", "")))
        date_key = str(row.get("date", "")).strip()[:10]
        if not title_key or not date_key:
            continue
        key = (date_key, title_key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    def _count_in_range(start_day: date, end_day: date) -> int:
        count = 0
        for row in deduped:
            day = _parse_date(row.get("date"))
            if not isinstance(day, date):
                continue
            if start_day <= day <= end_day:
                count += 1
        return count

    count_current_week = _count_in_range(current_window.start, current_window.end)
    count_previous_week = _count_in_range(previous_window.start, previous_window.end)
    count_current_month = _count_in_range(current_month_start, current_month_end)
    count_previous_month = _count_in_range(previous_month_start, previous_month_end)
    count_yoy_month = _count_in_range(yoy_month_start, yoy_month_end)

    topic_counts: dict[str, int] = {}
    for row in deduped:
        kind = str(row.get("kind", "")).strip() or "Search guidance"
        topic_counts[kind] = int(topic_counts.get(kind, 0) or 0) + 1

    latest = deduped[0] if deduped else {}
    return {
        "enabled": bool(deduped),
        "source": "Google update timeline (status + blog + trusted industry RSS)",
        "scan_months": max(13, int(scan_months)),
        "scan_start": since_day.isoformat(),
        "scan_end": run_date.isoformat(),
        "windows": {
            "current_week": {"start": current_window.start.isoformat(), "end": current_window.end.isoformat()},
            "previous_week": {"start": previous_window.start.isoformat(), "end": previous_window.end.isoformat()},
            "current_30d": {"start": current_month_start.isoformat(), "end": current_month_end.isoformat()},
            "previous_30d": {"start": previous_month_start.isoformat(), "end": previous_month_end.isoformat()},
            "yoy_30d": {"start": yoy_month_start.isoformat(), "end": yoy_month_end.isoformat()},
        },
        "summary": {
            "count_current_week": count_current_week,
            "count_previous_week": count_previous_week,
            "delta_week_vs_previous": count_current_week - count_previous_week,
            "delta_week_pct_vs_previous": (_safe_pct(count_current_week, count_previous_week) * 100.0) if count_previous_week else 0.0,
            "count_current_30d": count_current_month,
            "count_previous_30d": count_previous_month,
            "count_yoy_30d": count_yoy_month,
            "delta_30d_vs_previous": count_current_month - count_previous_month,
            "delta_30d_pct_vs_previous": (_safe_pct(count_current_month, count_previous_month) * 100.0) if count_previous_month else 0.0,
            "delta_30d_vs_yoy": count_current_month - count_yoy_month,
            "delta_30d_pct_vs_yoy": (_safe_pct(count_current_month, count_yoy_month) * 100.0) if count_yoy_month else 0.0,
            "total_count_13m": len(deduped),
            "topic_counts_13m": topic_counts,
            "latest_update_date": str(latest.get("date", "")).strip(),
            "latest_update_title": str(latest.get("title", "")).strip(),
        },
        "rows": deduped[:120],
        "errors": errors,
    }


def _collect_serp_case_studies_context(
    *,
    run_date: date,
    current_window: DateWindow,
    previous_window: DateWindow,
    yoy_window: DateWindow,
    scan_months: int = 13,
) -> tuple[dict[str, Any], list[ExternalSignal]]:
    scan_days = max(390, int(scan_months) * 30)
    since_day = run_date - timedelta(days=scan_days)
    current_month_end = current_window.end
    current_month_start = current_month_end - timedelta(days=29)
    previous_month_end = current_month_start - timedelta(days=1)
    previous_month_start = previous_month_end - timedelta(days=29)
    yoy_month_start = current_month_start - timedelta(weeks=52)
    yoy_month_end = current_month_end - timedelta(weeks=52)

    rows: list[dict[str, Any]] = []
    signals: list[ExternalSignal] = []
    errors: list[str] = []

    for label, url in SERP_CASE_STUDY_RSS_SOURCES:
        try:
            items = _fetch_generic_rss_signals(
                url=url,
                source_label=label,
                since=since_day,
                max_rows=240,
                severity="info",
            )
            for item in items:
                text_blob = f"{item.title} {item.details}".lower()
                if not _is_serp_case_candidate(text_blob):
                    continue
                intent_hits = sum(1 for token in SERP_CASE_STUDY_INTENT_TOKENS if token in text_blob)
                signal_hits = sum(1 for token in SERP_CASE_STUDY_SIGNAL_TOKENS if token in text_blob)
                score = min(100, 35 + (intent_hits * 15) + (signal_hits * 10))
                topic = _classify_serp_case_topic(text_blob)
                rows.append(
                    {
                        "date": item.day.isoformat(),
                        "title": str(item.title).strip()[:180],
                        "source": label,
                        "topic": topic,
                        "relevance_score": score,
                        "url": item.url or "",
                        "summary": str(item.details).strip()[:240],
                    }
                )
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("relevance_score", 0) or 0),
            str(item.get("date", "")),
        ),
        reverse=True,
    ):
        key = (
            str(row.get("date", "")).strip()[:10],
            _normalize_title_token(str(row.get("title", ""))),
        )
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    def _count_in_range(start_day: date, end_day: date) -> int:
        count = 0
        for row in deduped:
            day = _parse_date(row.get("date"))
            if not isinstance(day, date):
                continue
            if start_day <= day <= end_day:
                count += 1
        return count

    topic_counts_13m: dict[str, int] = {}
    for row in deduped:
        topic = str(row.get("topic", "")).strip() or "General SERP observations"
        topic_counts_13m[topic] = int(topic_counts_13m.get(topic, 0) or 0) + 1

    count_current_week = _count_in_range(current_window.start, current_window.end)
    count_previous_week = _count_in_range(previous_window.start, previous_window.end)
    count_current_month = _count_in_range(current_month_start, current_month_end)
    count_previous_month = _count_in_range(previous_month_start, previous_month_end)
    count_yoy_month = _count_in_range(yoy_month_start, yoy_month_end)

    latest = deduped[0] if deduped else {}
    for row in deduped[:3]:
        day = _parse_date(row.get("date")) or run_date
        signals.append(
            ExternalSignal(
                source="SERP case studies (13M scan)",
                day=day,
                title=str(row.get("title", "")).strip()[:180] or "SERP case study signal",
                details=(
                    f"Topic: {str(row.get('topic', '')).strip() or 'General SERP observations'}; "
                    f"relevance {int(row.get('relevance_score', 0) or 0)}/100. "
                    f"{str(row.get('summary', '')).strip()[:180]}"
                ).strip(),
                severity="info",
                url=str(row.get("url", "")).strip() or None,
            )
        )

    return {
        "enabled": bool(deduped),
        "source": "SERP case-study scanner (trusted SEO RSS)",
        "scan_months": max(13, int(scan_months)),
        "scan_start": since_day.isoformat(),
        "scan_end": run_date.isoformat(),
        "windows": {
            "current_week": {"start": current_window.start.isoformat(), "end": current_window.end.isoformat()},
            "previous_week": {"start": previous_window.start.isoformat(), "end": previous_window.end.isoformat()},
            "current_30d": {"start": current_month_start.isoformat(), "end": current_month_end.isoformat()},
            "previous_30d": {"start": previous_month_start.isoformat(), "end": previous_month_end.isoformat()},
            "yoy_30d": {"start": yoy_month_start.isoformat(), "end": yoy_month_end.isoformat()},
        },
        "summary": {
            "count_current_week": count_current_week,
            "count_previous_week": count_previous_week,
            "delta_week_vs_previous": count_current_week - count_previous_week,
            "delta_week_pct_vs_previous": (_safe_pct(count_current_week, count_previous_week) * 100.0) if count_previous_week else 0.0,
            "count_current_30d": count_current_month,
            "count_previous_30d": count_previous_month,
            "count_yoy_30d": count_yoy_month,
            "delta_30d_vs_previous": count_current_month - count_previous_month,
            "delta_30d_pct_vs_previous": (_safe_pct(count_current_month, count_previous_month) * 100.0) if count_previous_month else 0.0,
            "delta_30d_vs_yoy": count_current_month - count_yoy_month,
            "delta_30d_pct_vs_yoy": (_safe_pct(count_current_month, count_yoy_month) * 100.0) if count_yoy_month else 0.0,
            "total_count_13m": len(deduped),
            "topic_counts_13m": topic_counts_13m,
            "latest_case_date": str(latest.get("date", "")).strip(),
            "latest_case_title": str(latest.get("title", "")).strip(),
        },
        "rows": deduped[:120],
        "errors": errors,
    }, signals


def collect_additional_context(
    target_site_url: str,
    target_domain: str,
    report_country_code: str,
    run_date: date,
    current_window: DateWindow,
    previous_window: DateWindow,
    google_drive_client_secret_path: str = "",
    google_drive_token_path: str = ".google_drive_token.json",
    google_drive_folder_name: str = "SEO Weekly Reports",
    google_drive_folder_id: str = "",
    seo_presentations_enabled: bool = False,
    seo_presentations_folder_reference: str = "",
    seo_presentations_max_files_per_year: int = 20,
    seo_presentations_max_text_files_per_year: int = 8,
    historical_reports_enabled: bool = True,
    historical_reports_count: int = 3,
    historical_reports_yoy_tolerance_days: int = 28,
    status_log_enabled: bool = True,
    status_file_reference: str = "",
    status_max_rows: int = 12,
    product_trends_enabled: bool = True,
    product_trends_comparison_sheet_reference: str = "",
    product_trends_upcoming_sheet_reference: str = "",
    product_trends_current_sheet_reference: str = "",
    product_trends_top_rows: int = 12,
    product_trends_horizon_days: int = 31,
    trade_plan_enabled: bool = True,
    trade_plan_sheet_reference: str = "",
    trade_plan_yoy_sheet_reference: str = "",
    trade_plan_tab_map: dict[str, str] | None = None,
    trade_plan_yoy_tab_map: dict[str, str] | None = None,
    trade_plan_top_rows: int = 12,
    platform_pulse_enabled: bool = True,
    platform_pulse_rss_urls: tuple[str, ...] = (),
    platform_pulse_top_rows: int = 12,
    pagespeed_api_key: str = "",
    google_trends_rss_url: str = "https://trends.google.com/trending/rss?geo=PL",
    nbp_api_base_url: str = "https://api.nbp.pl/api",
    imgw_warnings_url: str = "https://danepubliczne.imgw.pl/api/data/warningsmeteo",
    market_events_enabled: bool = True,
    market_events_api_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc",
    market_events_top_rows: int = 12,
    google_status_endpoint: str = "https://status.search.google.com/incidents.json",
    google_blog_rss: str = "https://developers.google.com/search/blog/rss.xml",
    google_updates_scan_months: int = 13,
    serp_case_study_scan_months: int = 13,
    free_public_sources_enabled: bool = True,
    free_public_sources_top_rows: int = 3,
    nager_holidays_country_code: str = "PL",
    eia_api_key: str = "",
) -> tuple[dict[str, Any], list[ExternalSignal]]:
    country_code = report_country_code.strip().upper() or "PL"
    yoy_window = DateWindow(
        name="YoY aligned (52 weeks ago)",
        start=current_window.start - timedelta(weeks=52),
        end=current_window.end - timedelta(weeks=52),
    )
    context: dict[str, Any] = {
        "country_code": country_code,
        "pagespeed": {},
        "pagespeed_source": "",
        "google_trends_brand": {},
        "duckduckgo_context": {},
        "campaign_events": [],
        "macro": {},
        "macro_backdrop": {},
        "operational_risks": {},
        "competitor_promo_radar": {},
        "market_event_calendar": {},
        "seo_presentations": {},
        "historical_reports": {},
        "status_log": {},
        "product_trends": {},
        "trade_plan": {},
        "platform_regulatory_pulse": {},
        "google_updates_timeline": {},
        "serp_case_studies": {},
        "free_public_source_hub": {},
        "errors": [],
    }
    signals: list[ExternalSignal] = []

    try:
        pagespeed: dict[str, Any] = {}
        pagespeed_source = ""

        if pagespeed_api_key.strip():
            try:
                pagespeed = _fetch_crux_snapshot(
                    target_site_url=target_site_url,
                    api_key=pagespeed_api_key,
                )
                pagespeed_source = "crux_api"
            except Exception as crux_exc:
                pagespeed = _fetch_pagespeed_snapshot(
                    target_site_url=target_site_url,
                    api_key=pagespeed_api_key,
                )
                pagespeed_source = "pagespeed_fallback"
                context["errors"].append(
                    f"CrUX fetch failed; using PageSpeed fallback: {crux_exc}"
                )
        else:
            pagespeed = _fetch_pagespeed_snapshot(
                target_site_url=target_site_url,
                api_key="",
            )
            pagespeed_source = "pagespeed_no_key"

        context["pagespeed"] = pagespeed
        context["pagespeed_source"] = pagespeed_source
        mobile = pagespeed.get("mobile", {})
        lcp = float(mobile.get("lcp_ms", 0.0))
        inp = float(mobile.get("inp_ms", 0.0))
        cls = float(mobile.get("cls", 0.0))
        if lcp >= 2500 or inp >= 200 or cls >= 0.1:
            signals.append(
                ExternalSignal(
                    source="PageSpeed Insights",
                    day=run_date,
                    title="Core Web Vitals risk on origin",
                    details=(
                        f"Mobile field data: LCP {lcp:.0f}ms, INP {inp:.0f}ms, CLS {cls:.2f}. "
                        "Slower CWV can reduce SEO performance and conversion."
                    ),
                    severity="medium",
                    url="https://pagespeed.web.dev/",
                )
            )
    except Exception as exc:
        message = str(exc)
        if "429" in message and not pagespeed_api_key.strip():
            context["errors"].append(
                "CrUX/PageSpeed unavailable due public rate limit. "
                "Set `PAGESPEED_API_KEY` (Google API key with Chrome UX Report API and PageSpeed Insights API enabled)."
            )
        else:
            context["errors"].append(f"CrUX/PageSpeed fetch failed: {exc}")

    try:
        brand_context = _fetch_google_trends_brand_context(
            country_code=country_code,
            current_window=current_window,
            previous_window=previous_window,
        )
        context["google_trends_brand"] = brand_context
        summary = brand_context.get("summary", {})
        if isinstance(summary, dict):
            wow = float(summary.get("delta_pct_vs_previous", 0.0))
            yoy = float(summary.get("delta_pct_vs_yoy", 0.0))
            if abs(wow) >= 12.0 or abs(yoy) >= 12.0:
                direction = "up" if wow >= 0 else "down"
                signals.append(
                    ExternalSignal(
                        source=f"Google Trends brand ({country_code})",
                        day=run_date,
                        title="Branded-search demand shift",
                        details=(
                            f"Average interest is {direction} vs previous window ({wow:+.1f}%) "
                            f"and vs YoY ({yoy:+.1f}%)."
                        ),
                        severity="medium" if abs(wow) >= 20.0 or abs(yoy) >= 20.0 else "info",
                        url="https://trends.google.com/",
                    )
                )
    except Exception as exc:
        context["errors"].append(f"Google Trends brand fetch failed: {exc}")

    try:
        campaign_signals = _fetch_campaign_tracker_signals(
            since=previous_window.start,
            run_date=run_date,
            country_code=country_code,
        )
        context["campaign_events"] = [
            {
                "date": row.day.isoformat(),
                "source": row.source,
                "title": row.title,
                "severity": row.severity,
                "url": row.url or "",
            }
            for row in campaign_signals[:24]
        ]
        signals.extend(campaign_signals)
    except Exception as exc:
        context["errors"].append(f"Campaign tracker fetch failed: {exc}")

    # Dedicated competitor promo radar (separate from generic campaign tracker).
    try:
        market = _market_meta(country_code)
        country_terms = market["country_terms"]
        competitors_query = (
            f"(({country_terms}) AND (Temu OR Amazon OR AliExpress OR SHEIN OR Ceneo OR OLX OR MediaExpert) "
            "AND (sale OR promotion OR promocja OR rabat OR coupon OR voucher OR \"black friday\" OR \"cyber monday\" OR \"prime day\"))"
        )
        competitor_promo_signals = _fetch_google_news_query_signals(
            country_code=country_code,
            query=competitors_query,
            source_label=f"Competitor promo radar ({country_code})",
            since=previous_window.start,
            max_rows=16,
            severity="medium",
        )
        context["competitor_promo_radar"] = {
            "enabled": True,
            "source": "Google News RSS query",
            "rows": [
                {
                    "date": row.day.isoformat(),
                    "source": row.source,
                    "title": row.title,
                    "url": row.url or "",
                }
                for row in competitor_promo_signals
            ],
        }
        signals.extend(competitor_promo_signals)
    except Exception as exc:
        context["errors"].append(f"Competitor promo radar fetch failed: {exc}")

    # Operational risk feed: logistics and payment disruptions.
    try:
        market = _market_meta(country_code)
        country_terms = market["country_terms"]
        logistics_query = (
            f"(({country_terms}) AND (courier OR parcel OR delivery OR logistics OR warehouse OR strike OR outage "
            "OR InPost OR DPD OR DHL OR UPS OR Poczta) AND (delay OR disruption OR awaria OR outage OR strike))"
        )
        payments_query = (
            f"(({country_terms}) AND (payments OR payment gateway OR BLIK OR Przelewy24 OR PayU OR Visa OR Mastercard OR bank) "
            "AND (outage OR awaria OR disruption OR downtime))"
        )
        logistics_signals = _fetch_google_news_query_signals(
            country_code=country_code,
            query=logistics_query,
            source_label=f"Operational risk logistics ({country_code})",
            since=previous_window.start,
            max_rows=10,
            severity="medium",
        )
        payment_signals = _fetch_google_news_query_signals(
            country_code=country_code,
            query=payments_query,
            source_label=f"Operational risk payments ({country_code})",
            since=previous_window.start,
            max_rows=10,
            severity="medium",
        )
        context["operational_risks"] = {
            "enabled": True,
            "source": "Google News RSS query",
            "logistics": [
                {
                    "date": row.day.isoformat(),
                    "title": row.title,
                    "url": row.url or "",
                }
                for row in logistics_signals
            ],
            "payments": [
                {
                    "date": row.day.isoformat(),
                    "title": row.title,
                    "url": row.url or "",
                }
                for row in payment_signals
            ],
        }
        signals.extend(logistics_signals)
        signals.extend(payment_signals)
    except Exception as exc:
        context["errors"].append(f"Operational risk fetch failed: {exc}")

    # Country-level macro backdrop (annual, source of longer-term demand pressure).
    try:
        macro_backdrop = _fetch_macro_backdrop(country_code=country_code)
        context["macro_backdrop"] = macro_backdrop
        rows = macro_backdrop.get("rows", {}) if isinstance(macro_backdrop, dict) else {}
        if isinstance(rows, dict):
            inflation = rows.get("inflation_cpi_pct", {})
            unemployment = rows.get("unemployment_pct", {})
            if isinstance(inflation, dict) and inflation.get("latest_value") is not None:
                inf_latest = float(inflation.get("latest_value") or 0.0)
                inf_prev = float(inflation.get("previous_value") or 0.0)
                signals.append(
                    ExternalSignal(
                        source=f"Macro backdrop {country_code}",
                        day=run_date,
                        title=f"Inflation backdrop: {inf_latest:.2f}% ({inflation.get('latest_year', '')})",
                        details=f"Previous available year: {inflation.get('previous_year', '')} ({inf_prev:.2f}%).",
                        severity="medium" if inf_latest >= 5.0 else "info",
                        url="https://api.worldbank.org/",
                    )
                )
            if isinstance(unemployment, dict) and unemployment.get("latest_value") is not None:
                un_latest = float(unemployment.get("latest_value") or 0.0)
                signals.append(
                    ExternalSignal(
                        source=f"Macro backdrop {country_code}",
                        day=run_date,
                        title=f"Unemployment backdrop: {un_latest:.2f}% ({unemployment.get('latest_year', '')})",
                        details="Use as context for market-wide demand pressure.",
                        severity="info",
                        url="https://api.worldbank.org/",
                    )
                )
    except Exception as exc:
        context["errors"].append(f"Macro backdrop fetch failed: {exc}")

    if country_code == "PL":
        try:
            macro, macro_signals = _fetch_macro_context(
                current_window=current_window,
                previous_window=previous_window,
                run_date=run_date,
                nbp_api_base_url=nbp_api_base_url,
                imgw_warnings_url=imgw_warnings_url,
            )
            context["macro"] = macro
            signals.extend(macro_signals)
        except Exception as exc:
            context["errors"].append(f"Macro fetch failed: {exc}")
    else:
        context["macro"] = {
            "country_code": country_code,
            "note": "NBP/IMGW sources are PL-only and are skipped for this market.",
        }

    if market_events_enabled:
        market_context: dict[str, Any] = {
            "enabled": True,
            "source": "GDELT DOC API",
            "country_code": country_code,
            "events": [],
            "events_yoy": [],
            "top_rows": max(1, market_events_top_rows),
            "window_current": {
                "start": previous_window.start.isoformat(),
                "end": (current_window.end + timedelta(days=31)).isoformat(),
            },
            "window_yoy": {
                "start": (yoy_window.start - timedelta(days=28)).isoformat(),
                "end": (yoy_window.end + timedelta(days=31)).isoformat(),
            },
            "errors": [],
        }
        try:
            market_rows = _fetch_market_event_calendar(
                country_code=country_code,
                since=previous_window.start,
                until=current_window.end + timedelta(days=31),
                top_rows=max(1, market_events_top_rows),
                api_base_url=market_events_api_base_url,
            )
            market_rows_yoy = _fetch_market_event_calendar(
                country_code=country_code,
                since=yoy_window.start - timedelta(days=28),
                until=yoy_window.end + timedelta(days=31),
                top_rows=max(1, market_events_top_rows),
                api_base_url=market_events_api_base_url,
            )
            market_context["events"] = market_rows
            market_context["events_yoy"] = market_rows_yoy
            market_context["counts"] = {
                "current": len(market_rows),
                "yoy": len(market_rows_yoy),
                "delta_vs_yoy": len(market_rows) - len(market_rows_yoy),
                "delta_pct_vs_yoy": (
                    ((len(market_rows) - len(market_rows_yoy)) / len(market_rows_yoy)) * 100.0
                    if len(market_rows_yoy)
                    else 0.0
                ),
            }
            for row in market_rows[: max(1, market_events_top_rows)]:
                if not isinstance(row, dict):
                    continue
                day = _parse_date(row.get("date")) or run_date
                severity = str(row.get("impact_level", "LOW")).strip().lower()
                if severity not in {"high", "medium"}:
                    severity = "info"
                signals.append(
                    ExternalSignal(
                        source=f"Market Events API {country_code} (GDELT)",
                        day=day,
                        title=str(row.get("title", "")).strip()[:180] or "Market event",
                        details=(
                            f"{row.get('event_type', 'Market signal')} | "
                            f"{row.get('impact_direction', 'Mixed')} "
                            f"({row.get('impact_level', 'LOW')}, conf {int(row.get('confidence', 0) or 0)}/100). "
                            f"{str(row.get('gmv_reason', '')).strip()}"
                        ),
                        severity=severity,
                        url=str(row.get("url", "")).strip() or None,
                    )
                )
        except Exception as exc:
            market_context.setdefault("errors", []).append(str(exc))
            context["errors"].append(f"Market event calendar fetch failed: {exc}")
        context["market_event_calendar"] = market_context

    if platform_pulse_enabled:
        try:
            pulse_rows = _fetch_platform_regulatory_pulse(
                country_code=country_code,
                since=previous_window.start,
                run_date=run_date,
                rss_urls=platform_pulse_rss_urls,
                top_rows=max(1, int(platform_pulse_top_rows)),
            )
            context["platform_regulatory_pulse"] = {
                "enabled": bool(pulse_rows),
                "country_code": country_code,
                "source": "Platform+Regulatory pulse (Google News + RSS)",
                "rows": [
                    {
                        "date": row.day.isoformat(),
                        "source": row.source,
                        "title": row.title,
                        "details": row.details,
                        "severity": row.severity,
                        "url": row.url or "",
                    }
                    for row in pulse_rows
                ],
                "top_rows": max(1, int(platform_pulse_top_rows)),
                "errors": [],
            }
            signals.extend(pulse_rows)
        except Exception as exc:
            context["platform_regulatory_pulse"] = {
                "enabled": False,
                "country_code": country_code,
                "source": "Platform+Regulatory pulse (Google News + RSS)",
                "rows": [],
                "top_rows": max(1, int(platform_pulse_top_rows)),
                "errors": [str(exc)],
            }
            context["errors"].append(f"Platform/regulatory pulse fetch failed: {exc}")

    try:
        updates_context = _collect_google_updates_timeline(
            run_date=run_date,
            current_window=current_window,
            previous_window=previous_window,
            yoy_window=yoy_window,
            status_endpoint=google_status_endpoint,
            blog_rss_url=google_blog_rss,
            scan_months=max(13, int(google_updates_scan_months)),
        )
        context["google_updates_timeline"] = updates_context
    except Exception as exc:
        context["errors"].append(f"Google updates timeline fetch failed: {exc}")

    try:
        case_studies_context, case_study_signals = _collect_serp_case_studies_context(
            run_date=run_date,
            current_window=current_window,
            previous_window=previous_window,
            yoy_window=yoy_window,
            scan_months=max(13, int(serp_case_study_scan_months)),
        )
        context["serp_case_studies"] = case_studies_context
        signals.extend(case_study_signals)
    except Exception as exc:
        context["errors"].append(f"SERP case-study scanner failed: {exc}")

    try:
        free_public_ctx, free_public_signals = _collect_free_public_source_hub(
            country_code=country_code,
            run_date=run_date,
            current_window=current_window,
            previous_window=previous_window,
            target_domain=target_domain,
            enabled=free_public_sources_enabled,
            top_rows_per_source=max(1, int(free_public_sources_top_rows)),
            nager_country_code=(nager_holidays_country_code or country_code).strip().upper() or country_code,
            eia_api_key=eia_api_key,
        )
        context["free_public_source_hub"] = free_public_ctx
        signals.extend(free_public_signals)
    except Exception as exc:
        context["errors"].append(f"Free public source hub fetch failed: {exc}")

    # DuckDuckGo as fallback-only context source: run only when stronger sources are sparse.
    stronger_signal_count = sum(
        1
        for row in signals
        if isinstance(row, ExternalSignal)
        and row.severity.lower() in {"high", "medium"}
        and (
            "campaign" in row.source.lower()
            or "market events" in row.source.lower()
            or "weekly seo digest" in row.source.lower()
            or "google search" in row.source.lower()
            or "seo update analysis" in row.source.lower()
        )
    )
    if stronger_signal_count < 3:
        try:
            ddg_context = _fetch_duckduckgo_context(
                target_domain=target_domain,
                country_code=country_code,
                current_window=current_window,
            )
            context["duckduckgo_context"] = ddg_context
            rows = ddg_context.get("rows", [])
            if isinstance(rows, list):
                for row in rows[:3]:
                    if not isinstance(row, dict):
                        continue
                    heading = str(row.get("heading", "")).strip() or str(row.get("query", "")).strip()
                    details_parts: list[str] = []
                    abstract = str(row.get("abstract", "")).strip()
                    if abstract:
                        details_parts.append(abstract[:220] + ("..." if len(abstract) > 220 else ""))
                    related = row.get("related_topics", [])
                    if isinstance(related, list) and related:
                        details_parts.append("Related: " + "; ".join(str(item) for item in related[:2]))
                    signals.append(
                        ExternalSignal(
                            source=f"DuckDuckGo fallback ({country_code})",
                            day=run_date,
                            title=heading or "External context hint",
                            details=" ".join(details_parts).strip() or "Context hint from DuckDuckGo Instant Answer API.",
                            severity="info",
                            url=str(row.get("abstract_url", "")).strip() or "https://duckduckgo.com/",
                        )
                    )
        except Exception as exc:
            context["errors"].append(f"DuckDuckGo fallback fetch failed: {exc}")
    else:
        context["duckduckgo_context"] = {
            "enabled": False,
            "source": "DuckDuckGo Instant Answer API",
            "country_code": country_code,
            "rows": [],
            "errors": [],
            "note": "Skipped by policy because stronger sources were available.",
        }

    if seo_presentations_enabled and seo_presentations_folder_reference:
        try:
            presentations_client = SEOPresentationsClient(
                client_secret_path=google_drive_client_secret_path,
                token_path=google_drive_token_path,
                folder_reference=seo_presentations_folder_reference,
                max_files_per_year=seo_presentations_max_files_per_year,
                max_text_files_per_year=seo_presentations_max_text_files_per_year,
            )
            seo_context = presentations_client.collect_context(run_date=run_date)
            context["seo_presentations"] = seo_context
        except Exception as exc:
            context["errors"].append(f"SEO presentations fetch failed: {exc}")

    trend_context_requested = bool(
        product_trends_enabled
        and (
            product_trends_comparison_sheet_reference.strip()
            or product_trends_upcoming_sheet_reference.strip()
            or product_trends_current_sheet_reference.strip()
        )
    )
    trade_plan_requested = bool(
        trade_plan_enabled and trade_plan_sheet_reference.strip()
    )

    if (
        historical_reports_enabled
        or status_log_enabled
        or trend_context_requested
        or trade_plan_requested
    ) and google_drive_client_secret_path:
        continuity_client = ContinuityClient(
            client_secret_path=google_drive_client_secret_path,
            token_path=google_drive_token_path,
            reports_folder_name=google_drive_folder_name,
            reports_folder_id=google_drive_folder_id,
            status_file_reference=status_file_reference,
            status_search_folder_reference=seo_presentations_folder_reference,
            max_recent_reports=historical_reports_count,
            yoy_tolerance_days=historical_reports_yoy_tolerance_days,
            max_status_rows=status_max_rows,
        )

        if historical_reports_enabled:
            try:
                historical_context = continuity_client.collect_historical_reports(run_date=run_date)
                context["historical_reports"] = historical_context
            except Exception as exc:
                context["errors"].append(f"Historical report fetch failed: {exc}")

        if status_log_enabled:
            try:
                status_context = continuity_client.collect_status_updates(run_date=run_date)
                context["status_log"] = status_context
            except Exception as exc:
                context["errors"].append(f"Status log fetch failed: {exc}")

        if trend_context_requested:
            try:
                trend_context = continuity_client.collect_product_trends(
                    run_date=run_date,
                    comparison_sheet_reference=product_trends_comparison_sheet_reference,
                    upcoming_sheet_reference=product_trends_upcoming_sheet_reference,
                    current_sheet_reference=product_trends_current_sheet_reference,
                    target_domain=target_domain,
                    top_rows=product_trends_top_rows,
                    horizon_days=product_trends_horizon_days,
                )
                context["product_trends"] = trend_context
            except Exception as exc:
                context["errors"].append(f"Product trends fetch failed: {exc}")

        if trade_plan_requested:
            try:
                tab_map = trade_plan_tab_map or {}
                tab_name = str(tab_map.get(country_code, "")).strip()
                if not tab_name:
                    tab_name = f"TP_{run_date.year}_{country_code}"
                yoy_tab_map = trade_plan_yoy_tab_map or {}
                yoy_tab_name = str(yoy_tab_map.get(country_code, "")).strip()
                if not yoy_tab_name and str(run_date.year) in tab_name:
                    yoy_tab_name = tab_name.replace(str(run_date.year), str(run_date.year - 1))
                trade_plan_context = continuity_client.collect_trade_plan_context(
                    run_date=run_date,
                    sheet_reference=trade_plan_sheet_reference,
                    tab_name=tab_name,
                    current_window_start=current_window.start,
                    current_window_end=current_window.end,
                    previous_window_start=previous_window.start,
                    previous_window_end=previous_window.end,
                    yoy_window_start=yoy_window.start,
                    yoy_window_end=yoy_window.end,
                    yoy_sheet_reference=trade_plan_yoy_sheet_reference,
                    yoy_tab_name=yoy_tab_name,
                    top_rows=max(1, int(trade_plan_top_rows)),
                )
                context["trade_plan"] = trade_plan_context
                if trade_plan_context.get("enabled"):
                    channel_rows = trade_plan_context.get("channel_split", [])
                    campaign_rows = trade_plan_context.get("campaign_rows", [])
                    if isinstance(channel_rows, list):
                        for row in channel_rows[:3]:
                            if not isinstance(row, dict):
                                continue
                            channel = str(row.get("channel", "")).strip() or "channel"
                            spend_delta = float(row.get("delta_spend", 0.0) or 0.0)
                            spend_pct = row.get("delta_spend_pct")
                            spend_pct_text = f"{float(spend_pct):+.2f}%" if isinstance(spend_pct, (int, float)) else "n/a"
                            signals.append(
                                ExternalSignal(
                                    source=f"Trade Plan ({country_code})",
                                    day=current_window.end,
                                    title=f"Channel pressure: {channel}",
                                    details=(
                                        f"Planned spend delta vs previous week: {spend_delta:+.0f} "
                                        f"({spend_pct_text})."
                                    ),
                                    severity="medium" if abs(spend_delta) >= 10000 else "info",
                                    url=str(trade_plan_context.get("sheet", {}).get("url", "")).strip() or None,
                                )
                            )
                    if isinstance(campaign_rows, list):
                        for row in campaign_rows[:3]:
                            if not isinstance(row, dict):
                                continue
                            campaign = str(row.get("campaign", "")).strip() or "Campaign"
                            category = str(row.get("category", "")).strip() or "unspecified"
                            current_spend = float(row.get("current_spend", 0.0) or 0.0)
                            first_date = str(row.get("first_date", "")).strip()
                            last_date = str(row.get("last_date", "")).strip()
                            signals.append(
                                ExternalSignal(
                                    source=f"Trade Plan ({country_code})",
                                    day=current_window.end,
                                    title=f"Planned campaign: {campaign}",
                                    details=(
                                        f"Category: {category}; spend in current window: {current_spend:.0f}; "
                                        f"schedule: {first_date} to {last_date}."
                                    ),
                                    severity="info",
                                    url=str(trade_plan_context.get("sheet", {}).get("url", "")).strip() or None,
                                )
                            )
                else:
                    errors = trade_plan_context.get("errors", [])
                    if isinstance(errors, list) and errors:
                        context["errors"].append(f"Trade plan fetch failed: {errors[0]}")
            except Exception as exc:
                context["errors"].append(f"Trade plan fetch failed: {exc}")

    signals.sort(key=lambda row: (row.day, row.source), reverse=True)
    return context, signals


def _fetch_crux_snapshot(target_site_url: str, api_key: str) -> dict[str, Any]:
    key = api_key.strip()
    if not key:
        raise RuntimeError("PAGESPEED_API_KEY is required for CrUX API.")

    origin = _origin_from_url(target_site_url)
    endpoint = "https://chromeuxreport.googleapis.com/v1/records:queryRecord"
    out: dict[str, Any] = {}

    for form_factor, strategy in (("PHONE", "mobile"), ("DESKTOP", "desktop")):
        payload = _request_json_with_retry(
            method="POST",
            url=endpoint,
            params={"key": key},
            json_body={"origin": origin, "formFactor": form_factor},
            timeout=40,
            max_attempts=4,
            retry_sleep_sec=1.5,
        )
        record = payload.get("record") or {}
        metrics = record.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}

        lcp_ms = _parse_crux_p75(metrics, ("largest_contentful_paint",))
        inp_ms = _parse_crux_p75(
            metrics,
            (
                "interaction_to_next_paint",
                "experimental_interaction_to_next_paint",
                "first_input_delay",
            ),
        )
        cls = _parse_crux_p75(metrics, ("cumulative_layout_shift",))
        out[strategy] = {
            "lcp_ms": lcp_ms,
            "inp_ms": inp_ms,
            "cls": cls,
            "overall_category": _overall_cwv_category(lcp_ms, inp_ms, cls),
        }
    return out


def _fetch_pagespeed_snapshot(target_site_url: str, api_key: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for strategy in ("mobile", "desktop"):
        params = {
            "url": target_site_url,
            "strategy": strategy,
            "category": "performance",
        }
        if api_key:
            params["key"] = api_key
        payload = _request_json_with_retry(
            method="GET",
            url="https://www.googleapis.com/pagespeedonline/v5/runPagespeed",
            params=params,
            timeout=40,
            max_attempts=4,
            retry_sleep_sec=1.3,
        )
        loading = payload.get("loadingExperience") or {}
        metrics = loading.get("metrics") or {}
        lcp = _parse_float(
            (metrics.get("LARGEST_CONTENTFUL_PAINT_MS") or {}).get("percentile")
        ) or 0.0
        inp = _parse_float(
            (metrics.get("INTERACTION_TO_NEXT_PAINT") or {}).get("percentile")
        ) or 0.0
        cls_raw = _parse_float(
            (metrics.get("CUMULATIVE_LAYOUT_SHIFT_SCORE") or {}).get("percentile")
        ) or 0.0
        cls = cls_raw / 100 if cls_raw > 1.0 else cls_raw
        out[strategy] = {
            "lcp_ms": lcp,
            "inp_ms": inp,
            "cls": cls,
            "overall_category": str(loading.get("overall_category", "")).strip()
            or _overall_cwv_category(lcp, inp, cls),
        }
    return out


def _fetch_google_trends(rss_url: str, since: date) -> list[dict[str, Any]]:
    response = requests.get(rss_url, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    ns = {"ht": "https://trends.google.com/trending/rss"}
    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = (item.findtext("pubDate") or "").strip()
        pub_dt = _parse_datetime(pub_raw)
        day = pub_dt.date() if pub_dt else None
        if day and day < since:
            continue

        approx_raw = (item.findtext("ht:approx_traffic", namespaces=ns) or "").strip()
        approx_traffic = _extract_approx_traffic(approx_raw)

        if not title:
            continue
        rows.append(
            {
                "topic": html.unescape(title),
                "url": link,
                "day": day or date.today(),
                "approx_traffic": approx_traffic,
                "approx_traffic_label": approx_raw or str(approx_traffic),
            }
        )

    rows.sort(key=lambda row: (row["day"], int(row.get("approx_traffic", 0))), reverse=True)
    return rows[:30]


def _parse_google_trends_json(response_text: str) -> dict[str, Any]:
    raw = response_text.strip()
    if raw.startswith(")]}'"):
        parts = raw.split("\n", 1)
        if len(parts) == 2:
            raw = parts[1].strip()
        else:
            raw = raw[4:].lstrip(", \n\r\t")
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _fetch_google_trends_keyword_interest(
    *,
    keyword: str,
    country_code: str,
    start_day: date,
    end_day: date,
) -> dict[str, Any]:
    cleaned_keyword = keyword.strip()
    if not cleaned_keyword:
        return {}
    market = _market_meta(country_code)
    hl = market["hl"]
    geo = country_code.strip().upper() or "PL"
    timeframe = f"{start_day.isoformat()} {end_day.isoformat()}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; weekly-seo-agent/1.0)",
        "Accept-Language": f"{hl},en;q=0.8",
        "Referer": "https://trends.google.com/trends/",
    }
    session = requests.Session()
    try:
        # Warmup call is required for stable cookie/session handling; without it,
        # Trends frequently responds with 429 to API endpoints.
        session.get(
            "https://trends.google.com/trends/",
            headers=headers,
            timeout=25,
        )
    except Exception:
        pass

    req_payload = {
        "comparisonItem": [{"keyword": cleaned_keyword, "geo": geo, "time": timeframe}],
        "category": 0,
        "property": "",
    }
    explore_text = _request_text_with_retry(
        method="GET",
        url="https://trends.google.com/trends/api/explore",
        session=session,
        params={
            "hl": hl,
            "tz": "0",
            "req": json.dumps(req_payload, separators=(",", ":")),
        },
        headers=headers,
        timeout=30,
        max_attempts=5,
        retry_sleep_sec=1.6,
    )
    explore_data = _parse_google_trends_json(explore_text)
    widgets = explore_data.get("widgets", [])
    if not isinstance(widgets, list):
        return {}

    widget = next(
        (
            row
            for row in widgets
            if isinstance(row, dict) and str(row.get("id", "")).strip().upper() == "TIMESERIES"
        ),
        None,
    )
    if not isinstance(widget, dict):
        return {}
    token = str(widget.get("token", "")).strip()
    request_payload = widget.get("request", {})
    if not token or not isinstance(request_payload, dict):
        return {}

    time.sleep(GOOGLE_TRENDS_BRAND_THROTTLE_SEC)
    multiline_text = _request_text_with_retry(
        method="GET",
        url="https://trends.google.com/trends/api/widgetdata/multiline",
        session=session,
        params={
            "hl": hl,
            "tz": "0",
            "token": token,
            "req": json.dumps(request_payload, separators=(",", ":")),
        },
        headers=headers,
        timeout=30,
        max_attempts=5,
        retry_sleep_sec=1.6,
    )
    multiline_data = _parse_google_trends_json(multiline_text)
    default = multiline_data.get("default", {})
    timeline = default.get("timelineData", []) if isinstance(default, dict) else []
    if not isinstance(timeline, list):
        return {}

    values: list[float] = []
    for row in timeline:
        if not isinstance(row, dict):
            continue
        value = row.get("value", [])
        if not isinstance(value, list) or not value:
            continue
        try:
            values.append(float(value[0]))
        except (TypeError, ValueError):
            continue
    if not values:
        return {}

    return {
        "keyword": cleaned_keyword,
        "window_start": start_day.isoformat(),
        "window_end": end_day.isoformat(),
        "points": len(values),
        "avg_interest": sum(values) / len(values),
        "max_interest": max(values),
        "latest_interest": values[-1],
    }


def _fetch_google_trends_brand_context(
    *,
    country_code: str,
    current_window: DateWindow,
    previous_window: DateWindow,
) -> dict[str, Any]:
    cache_key = (
        f"{country_code}_{current_window.start.isoformat()}_{current_window.end.isoformat()}_"
        f"{previous_window.start.isoformat()}_{previous_window.end.isoformat()}"
    )
    cached = _load_local_cache(
        "google_trends_brand",
        cache_key,
        max_age_sec=GOOGLE_TRENDS_BRAND_CACHE_TTL_SEC,
    )
    if isinstance(cached, dict) and cached.get("enabled"):
        return cached

    yoy_window = DateWindow(
        name="YoY aligned (52 weeks ago)",
        start=current_window.start - timedelta(weeks=52),
        end=current_window.end - timedelta(weeks=52),
    )
    # Keep request volume low to avoid Trends rate limits.
    keywords = ("allegro",)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for keyword in keywords:
        try:
            current_row = _fetch_google_trends_keyword_interest(
                keyword=keyword,
                country_code=country_code,
                start_day=current_window.start,
                end_day=current_window.end,
            )
            previous_row = _fetch_google_trends_keyword_interest(
                keyword=keyword,
                country_code=country_code,
                start_day=previous_window.start,
                end_day=previous_window.end,
            )
            yoy_row = _fetch_google_trends_keyword_interest(
                keyword=keyword,
                country_code=country_code,
                start_day=yoy_window.start,
                end_day=yoy_window.end,
            )
            if not current_row:
                continue
            current_avg = float(current_row.get("avg_interest", 0.0))
            previous_avg = float(previous_row.get("avg_interest", 0.0)) if previous_row else 0.0
            yoy_avg = float(yoy_row.get("avg_interest", 0.0)) if yoy_row else 0.0
            rows.append(
                {
                    "keyword": keyword,
                    "current_avg": current_avg,
                    "previous_avg": previous_avg,
                    "yoy_avg": yoy_avg,
                    "delta_pct_vs_previous": (_safe_pct(current_avg, previous_avg) * 100.0) if previous_avg else 0.0,
                    "delta_pct_vs_yoy": (_safe_pct(current_avg, yoy_avg) * 100.0) if yoy_avg else 0.0,
                    "current_window": f"{current_window.start.isoformat()} to {current_window.end.isoformat()}",
                    "previous_window": f"{previous_window.start.isoformat()} to {previous_window.end.isoformat()}",
                    "yoy_window": f"{yoy_window.start.isoformat()} to {yoy_window.end.isoformat()}",
                }
            )
        except Exception as exc:
            errors.append(f"{keyword}: {exc}")

    summary: dict[str, Any] = {}
    if rows:
        current_avg = sum(float(row.get("current_avg", 0.0)) for row in rows) / len(rows)
        previous_avg = sum(float(row.get("previous_avg", 0.0)) for row in rows) / len(rows)
        yoy_avg = sum(float(row.get("yoy_avg", 0.0)) for row in rows) / len(rows)
        summary = {
            "avg_current": current_avg,
            "avg_previous": previous_avg,
            "avg_yoy": yoy_avg,
            "delta_pct_vs_previous": (_safe_pct(current_avg, previous_avg) * 100.0) if previous_avg else 0.0,
            "delta_pct_vs_yoy": (_safe_pct(current_avg, yoy_avg) * 100.0) if yoy_avg else 0.0,
        }

    result = {
        "enabled": bool(rows),
        "source": "Google Trends API (interest over time)",
        "country_code": country_code,
        "rows": rows,
        "summary": summary,
        "errors": errors,
    }
    if result.get("enabled"):
        _save_local_cache("google_trends_brand", cache_key, result)
    return result


def _ddg_query_once(query: str, *, kl: str) -> dict[str, Any]:
    response = requests.get(
        DUCKDUCKGO_API_URL,
        params={
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "kl": kl,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _fetch_duckduckgo_context(
    *,
    target_domain: str,
    country_code: str,
    current_window: DateWindow,
) -> dict[str, Any]:
    market = _market_meta(country_code)
    kl = f"{market['hl']}-{market['gl'].lower()}"
    year = current_window.end.year
    queries = [
        f"{target_domain} SEO visibility drop causes",
        f"{target_domain} ecommerce traffic decline reasons {year}",
        f"Google Discover update {year} ecommerce impact",
        f"{target_domain} campaign promotion impact organic traffic",
        f"{market['country_terms']} ecommerce demand shift weather impact",
    ]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for query in queries:
        try:
            payload = _ddg_query_once(query, kl=kl)
            abstract = str(payload.get("AbstractText", "")).strip()
            heading = str(payload.get("Heading", "")).strip()
            abstract_url = str(payload.get("AbstractURL", "")).strip()

            related_topics = payload.get("RelatedTopics", [])
            related_snippets: list[str] = []
            if isinstance(related_topics, list):
                for item in related_topics:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("Text", "")).strip()
                    if text:
                        related_snippets.append(text)
                    nested = item.get("Topics", [])
                    if isinstance(nested, list):
                        for nitem in nested:
                            if not isinstance(nitem, dict):
                                continue
                            ntext = str(nitem.get("Text", "")).strip()
                            if ntext:
                                related_snippets.append(ntext)
                    if len(related_snippets) >= 5:
                        break

            if abstract or heading or related_snippets:
                rows.append(
                    {
                        "query": query,
                        "heading": heading,
                        "abstract": abstract,
                        "abstract_url": abstract_url,
                        "related_topics": related_snippets[:5],
                    }
                )
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    return {
        "enabled": bool(rows),
        "source": "DuckDuckGo Instant Answer API",
        "country_code": country_code,
        "rows": rows,
        "errors": errors,
    }


def _fetch_nbp_rates(
    code: str,
    current_window: DateWindow,
    previous_window: DateWindow,
    nbp_api_base_url: str,
) -> dict[str, Any]:
    start = previous_window.start.isoformat()
    end = current_window.end.isoformat()
    url = f"{nbp_api_base_url.rstrip('/')}/exchangerates/rates/a/{code}/{start}/{end}/?format=json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rates = payload.get("rates") or []

    by_day: dict[date, float] = {}
    last_day = previous_window.start
    for row in rates:
        if not isinstance(row, dict):
            continue
        day = _parse_date(row.get("effectiveDate"))
        value = _parse_float(row.get("mid"))
        if day is None or value is None:
            continue
        by_day[day] = value
        if day > last_day:
            last_day = day

    avg_current = _avg_fx_for_window(by_day, current_window)
    avg_previous = _avg_fx_for_window(by_day, previous_window)
    latest = by_day.get(last_day, 0.0)
    return {
        "code": code,
        "avg_current": avg_current,
        "avg_previous": avg_previous,
        "latest": latest,
        "delta_pct_vs_previous": _safe_pct(avg_current, avg_previous) * 100.0,
    }


def _fetch_imgw_warnings(run_date: date, imgw_warnings_url: str) -> list[dict[str, Any]]:
    response = requests.get(imgw_warnings_url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return []

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        to_raw = row.get("obowiazuje_do") or row.get("to")
        from_raw = row.get("obowiazuje_od") or row.get("from")
        to_dt = _parse_datetime(to_raw)
        from_dt = _parse_datetime(from_raw)
        if to_dt and to_dt.date() < run_date:
            continue
        if from_dt and (run_date - from_dt.date()).days > 7:
            continue
        severity_raw = str(row.get("stopien") or row.get("severity") or "").strip()
        severity_match = re.search(r"[1-3]", severity_raw)
        severity = severity_match.group(0) if severity_match else severity_raw
        event = str(row.get("nazwa_zdarzenia") or row.get("event") or "Warning").strip()
        areas = str(
            row.get("wojewodztwo") or row.get("obszar") or row.get("powiaty") or ""
        ).strip()
        dedupe_key = (
            event.lower(),
            severity.lower(),
            (to_dt.date().isoformat() if to_dt else ""),
            areas.lower(),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(
            {
                "event": event,
                "severity": severity,
                "from": from_dt.date().isoformat() if from_dt else "",
                "to": to_dt.date().isoformat() if to_dt else "",
                "areas": areas,
            }
        )
    out.sort(
        key=lambda row: (
            int(re.search(r"[1-3]", str(row.get("severity", ""))).group(0))
            if re.search(r"[1-3]", str(row.get("severity", "")))
            else 0,
            str(row.get("to", "")),
            str(row.get("event", "")),
        ),
        reverse=True,
    )
    return out[:50]


def _fetch_macro_context(
    current_window: DateWindow,
    previous_window: DateWindow,
    run_date: date,
    nbp_api_base_url: str,
    imgw_warnings_url: str,
) -> tuple[dict[str, Any], list[ExternalSignal]]:
    macro: dict[str, Any] = {}
    signals: list[ExternalSignal] = []

    eur = _fetch_nbp_rates(
        code="eur",
        current_window=current_window,
        previous_window=previous_window,
        nbp_api_base_url=nbp_api_base_url,
    )
    usd = _fetch_nbp_rates(
        code="usd",
        current_window=current_window,
        previous_window=previous_window,
        nbp_api_base_url=nbp_api_base_url,
    )
    macro["nbp_fx"] = {"eur_pln": eur, "usd_pln": usd}

    for pair_label, row in (("EUR/PLN", eur), ("USD/PLN", usd)):
        delta = float(row.get("delta_pct_vs_previous", 0.0))
        if abs(delta) >= 2.0:
            direction = "wzrosl" if delta > 0 else "spadl"
            signals.append(
                ExternalSignal(
                    source="NBP FX",
                    day=run_date,
                    title=f"{pair_label} {direction} o {abs(delta):.2f}% vs poprzednie 28 dni",
                    details=(
                        f"Srednia current={float(row.get('avg_current', 0.0)):.4f}, "
                        f"previous={float(row.get('avg_previous', 0.0)):.4f}."
                    ),
                    severity="medium",
                    url="https://api.nbp.pl/",
                )
            )

    warnings = _fetch_imgw_warnings(run_date=run_date, imgw_warnings_url=imgw_warnings_url)
    macro["imgw_warnings"] = warnings[:10]
    macro["imgw_warnings_total"] = len(warnings)
    if warnings:
        high_count = 0
        for row in warnings:
            severity_text = str(row.get("severity", "")).strip()
            match = re.search(r"[1-3]", severity_text)
            if match and match.group(0) in {"2", "3"}:
                high_count += 1
        macro["imgw_high_severity_count"] = high_count
        severity = "high" if high_count >= 3 else "medium"
        signals.append(
            ExternalSignal(
                source="IMGW warnings",
                day=run_date,
                title=f"Aktywne ostrzezenia pogodowe: {len(warnings)}",
                details=(
                    f"Ostrzezenia stopnia 2-3: {high_count}. "
                    "Pogoda moze przesuwac popyt i terminy dostaw."
                ),
                severity=severity,
                url=imgw_warnings_url,
            )
        )

    return macro, signals
