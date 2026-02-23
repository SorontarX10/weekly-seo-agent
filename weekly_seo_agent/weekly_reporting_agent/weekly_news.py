from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import html
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests

from weekly_seo_agent.weekly_reporting_agent.models import DateWindow


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published: date | None
    domain: str
    summary: str
    topic: str


def _normalize_title_for_dedup(title: str) -> str:
    text = (title or "").strip().lower()
    # Remove social tags and noisy suffixes that create near-duplicates.
    text = re.sub(r"\s+via\s+@[\w, @.-]+$", "", text)
    text = re.sub(r"\s*[–-]\s*search engine journal$", "", text)
    text = re.sub(r"\s*[–-]\s*seo pulse$", "", text)
    text = re.sub(r"\s*[–-]\s*ppc pulse$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_title(title: str) -> str:
    text = (title or "").strip()
    text = re.sub(r"\s+via\s+@[\w, @.-]+$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[–-]\s*Search Engine Journal$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_summary(summary: str, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", (summary or "")).strip()
    text = re.sub(r"\bThe post .+? appeared first on .+?\.\s*$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_datetime(raw_value: object) -> datetime | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
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


def _domain_matches(domain: str, allowlist: tuple[str, ...]) -> bool:
    if not allowlist:
        return True
    for allowed in allowlist:
        allowed = allowed.strip().lower()
        if not allowed:
            continue
        if domain == allowed:
            return True
        if domain.endswith("." + allowed):
            return True
    return False


def _rss_items(url: str) -> list[dict[str, object]]:
    try:
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception:
        return []

    items = root.findall(".//item")
    if not items:
        items = root.findall(".//entry")

    rows: list[dict[str, object]] = []
    for item in items[:80]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not link:
            link_elem = item.find("link")
            if link_elem is not None:
                link = str(link_elem.attrib.get("href", "")).strip()
        description = (
            item.findtext("description")
            or item.findtext("summary")
            or item.findtext("{http://www.w3.org/2005/Atom}summary")
            or ""
        ).strip()
        source_elem = item.find("source")
        source_text = ""
        source_url = ""
        if source_elem is not None:
            source_text = (source_elem.text or "").strip()
            source_url = str(source_elem.attrib.get("url", "") or "").strip()
        pub_raw = (
            item.findtext("pubDate")
            or item.findtext("published")
            or item.findtext("updated")
            or item.findtext("{http://www.w3.org/2005/Atom}updated")
            or ""
        ).strip()
        rows.append(
            {
                "title": title,
                "link": link,
                "description": description,
                "published_raw": pub_raw,
                "source_text": source_text,
                "source_url": source_url,
            }
        )
    return rows


def _collect_from_rss(
    urls: tuple[str, ...],
    allowlist: tuple[str, ...],
    keywords: tuple[str, ...],
    window: DateWindow,
    topic: str,
) -> list[NewsItem]:
    items: list[NewsItem] = []
    keywords_lower = tuple(keyword.lower() for keyword in keywords if keyword.strip())

    for url in urls:
        for row in _rss_items(url):
            title = _strip_html(str(row.get("title") or ""))
            link = str(row.get("link") or "").strip()
            description = _strip_html(str(row.get("description") or ""))
            published = _parse_datetime(row.get("published_raw"))

            if published is not None:
                published_date = published.date()
                if published_date < window.start or published_date > window.end:
                    continue
            else:
                published_date = None

            text_blob = f"{title} {description}".lower()
            if keywords_lower and not any(keyword in text_blob for keyword in keywords_lower):
                continue

            domain = (urlparse(link).netloc or urlparse(link).path or "").lower()
            source_text = str(row.get("source_text") or "").strip()
            source_url = str(row.get("source_url") or "").strip()
            source_domain = (urlparse(source_url).netloc or urlparse(source_url).path or "").lower()
            domain_check = source_domain or domain
            if domain_check and not _domain_matches(domain_check, allowlist):
                continue

            source_host = urlparse(url).netloc or "rss"
            preferred_url = link
            if domain == "news.google.com" and source_url:
                preferred_url = source_url
                domain = source_domain or domain
            source_label = source_text or domain or source_host
            items.append(
                NewsItem(
                    title=_clean_title(title or "Untitled"),
                    url=preferred_url,
                    source=source_label,
                    published=published_date,
                    domain=domain or source_host,
                    summary=_clean_summary(description or "No description provided."),
                    topic=topic,
                )
            )
    return items


def _unique_items(items: list[NewsItem], limit: int) -> list[NewsItem]:
    seen: set[tuple[str, str]] = set()
    out: list[NewsItem] = []
    for item in sorted(items, key=lambda row: row.published or date.min, reverse=True):
        key = (_normalize_title_for_dedup(item.title), item.domain.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def collect_weekly_news(
    window: DateWindow,
    seo_urls: tuple[str, ...],
    geo_urls: tuple[str, ...],
    seo_allowlist: tuple[str, ...],
    geo_allowlist: tuple[str, ...],
    seo_keywords: tuple[str, ...],
    geo_keywords: tuple[str, ...],
    max_items: int,
) -> tuple[list[NewsItem], list[NewsItem]]:
    seo_items = _collect_from_rss(
        urls=seo_urls,
        allowlist=seo_allowlist,
        keywords=seo_keywords,
        window=window,
        topic="SEO",
    )
    geo_items = _collect_from_rss(
        urls=geo_urls,
        allowlist=geo_allowlist,
        keywords=geo_keywords,
        window=window,
        topic="GEO",
    )
    return (
        _unique_items(seo_items, max_items),
        _unique_items(geo_items, max_items),
    )
