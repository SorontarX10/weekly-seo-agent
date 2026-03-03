from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse

import requests


DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass(slots=True)
class WebResearchItem:
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"
    page_excerpt: str = ""
    page_error: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "page_excerpt": self.page_excerpt,
            "page_error": self.page_error,
        }


def run_web_research(
    *,
    query: str,
    region: str = "us-en",
    max_results: int = 6,
    fetch_pages: bool = True,
    max_pages: int = 3,
    page_char_limit: int = 1800,
    timeout_sec: int = 20,
    page_fetcher_factory: Callable[[], "PlaywrightStealthFetcher"] | None = None,
) -> dict:
    normalized_query = _normalize_query(query)
    rows = duckduckgo_search(
        query=normalized_query,
        region=region,
        max_results=max_results,
        timeout_sec=timeout_sec,
    )
    if len(rows) < max_results:
        fallback_rows = duckduckgo_html_search(
            query=normalized_query,
            region=region,
            max_results=max_results,
            timeout_sec=timeout_sec,
        )
        rows = _merge_research_rows(rows, fallback_rows, max_results=max_results)

    items = [
        WebResearchItem(
            title=str(row.get("title", "")).strip(),
            url=str(row.get("url", "")).strip(),
            snippet=str(row.get("snippet", "")).strip(),
            source=str(row.get("source", "duckduckgo")).strip() or "duckduckgo",
        )
        for row in rows
    ]

    warning = ""
    fallback_used = any(str(item.source).strip() == "duckduckgo_html" for item in items)
    playwright_attempted = 0
    playwright_success = 0
    playwright_errors = 0
    if fetch_pages and items:
        fetcher = (
            page_fetcher_factory()
            if page_fetcher_factory is not None
            else PlaywrightStealthFetcher(timeout_ms=max(5000, timeout_sec * 1000))
        )
        for item in items:
            if playwright_attempted >= max_pages:
                break
            if not item.url:
                continue
            playwright_attempted += 1
            try:
                item.page_excerpt = fetcher.fetch_page_text(
                    item.url,
                    max_chars=page_char_limit,
                )
                if item.page_excerpt:
                    playwright_success += 1
            except Exception as exc:
                message = str(exc).strip() or exc.__class__.__name__
                item.page_error = message
                playwright_errors += 1
                if not warning:
                    warning = f"Playwright fetch partially failed: {message}"
    elif fetch_pages and not items:
        warning = (
            "Search returned 0 URLs. Playwright page checks were skipped."
        )

    summary_text = _build_research_summary(
        query=normalized_query,
        region=region,
        items=items,
        fetch_pages=fetch_pages,
        playwright_attempted=playwright_attempted,
        playwright_success=playwright_success,
        playwright_errors=playwright_errors,
        fallback_used=fallback_used,
    )
    return {
        "query": normalized_query,
        "region": region,
        "provider": "duckduckgo_instant_answer",
        "items": [item.to_dict() for item in items],
        "summary_text": summary_text,
        "warning": warning,
    }


def duckduckgo_search(
    *,
    query: str,
    region: str,
    max_results: int,
    timeout_sec: int,
) -> list[dict]:
    response = requests.get(
        DUCKDUCKGO_API_URL,
        params={
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "kl": region,
        },
        timeout=max(5, timeout_sec),
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return []

    out: list[dict] = []
    seen_urls: set[str] = set()

    heading = _clean_text(str(payload.get("Heading", "")))
    abstract = _clean_text(str(payload.get("AbstractText", "")))
    abstract_url = str(payload.get("AbstractURL", "")).strip()
    if abstract and abstract_url:
        _append_unique_result(
            out=out,
            seen_urls=seen_urls,
            title=heading or _host_label(abstract_url),
            url=abstract_url,
            snippet=abstract,
            source="duckduckgo_abstract",
        )

    direct_results = payload.get("Results", [])
    if isinstance(direct_results, list):
        for row in direct_results:
            if not isinstance(row, dict):
                continue
            _append_unique_result(
                out=out,
                seen_urls=seen_urls,
                title=_clean_text(str(row.get("Text", "")))[:140] or "Result",
                url=str(row.get("FirstURL", "")).strip(),
                snippet=_clean_text(str(row.get("Text", ""))),
                source="duckduckgo_result",
            )
            if len(out) >= max_results:
                break

    related = payload.get("RelatedTopics", [])
    if isinstance(related, list) and len(out) < max_results:
        for row in related:
            if len(out) >= max_results:
                break
            _consume_related_topic(
                row=row,
                out=out,
                seen_urls=seen_urls,
                max_results=max_results,
            )

    return out[:max_results]


def duckduckgo_html_search(
    *,
    query: str,
    region: str,
    max_results: int,
    timeout_sec: int,
) -> list[dict]:
    response = requests.get(
        DUCKDUCKGO_HTML_URL,
        params={
            "q": query,
            "kl": region,
        },
        timeout=max(5, timeout_sec),
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    response.raise_for_status()
    html_text = str(response.text or "")
    if not html_text.strip():
        return []

    out: list[dict] = []
    seen_urls: set[str] = set()
    anchor_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in anchor_pattern.finditer(html_text):
        if len(out) >= max_results:
            break
        href = _resolve_duckduckgo_result_url(match.group(1))
        if not href:
            continue
        title = _strip_html_tags(match.group(2))
        _append_unique_result(
            out=out,
            seen_urls=seen_urls,
            title=title or _host_label(href),
            url=href,
            snippet=title,
            source="duckduckgo_html",
        )
    return out[:max_results]


def _consume_related_topic(
    *,
    row: dict | object,
    out: list[dict],
    seen_urls: set[str],
    max_results: int,
) -> None:
    if not isinstance(row, dict):
        return
    text = _clean_text(str(row.get("Text", "")))
    first_url = str(row.get("FirstURL", "")).strip()
    if text and first_url:
        _append_unique_result(
            out=out,
            seen_urls=seen_urls,
            title=text[:140] or _host_label(first_url),
            url=first_url,
            snippet=text,
            source="duckduckgo_related",
        )
    nested = row.get("Topics", [])
    if isinstance(nested, list):
        for nrow in nested:
            if len(out) >= max_results:
                break
            _consume_related_topic(
                row=nrow,
                out=out,
                seen_urls=seen_urls,
                max_results=max_results,
            )


def _append_unique_result(
    *,
    out: list[dict],
    seen_urls: set[str],
    title: str,
    url: str,
    snippet: str,
    source: str,
) -> None:
    normalized_url = url.strip()
    if not normalized_url:
        return
    if normalized_url in seen_urls:
        return
    seen_urls.add(normalized_url)
    out.append(
        {
            "title": _clean_text(title)[:200] or _host_label(normalized_url),
            "url": normalized_url,
            "snippet": _clean_text(snippet)[:500],
            "source": source,
        }
    )


class PlaywrightStealthFetcher:
    def __init__(self, *, timeout_ms: int = 20000):
        self.timeout_ms = max(2000, timeout_ms)

    def fetch_page_text(self, url: str, *, max_chars: int = 1800) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(
                "Playwright is not installed. Install with 'pip install playwright' "
                "and then run 'playwright install chromium'."
            ) from exc

        stealth_apply = None
        try:
            from playwright_stealth import stealth_sync  # type: ignore

            stealth_apply = stealth_sync
        except Exception:
            stealth_apply = None

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = browser.new_context(
                user_agent=DEFAULT_USER_AGENT,
                locale="en-US",
                viewport={"width": 1366, "height": 768},
            )
            page = context.new_page()
            if stealth_apply is not None:
                stealth_apply(page)
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 5000))
            except Exception:
                pass
            title = _clean_text(page.title())
            body_text = _clean_text(
                str(page.evaluate("() => document.body ? document.body.innerText : ''"))
            )
            browser.close()

        if not body_text:
            raise RuntimeError("Page body text is empty")

        excerpt = body_text[:max(200, max_chars)].strip()
        if title:
            return f"{title}\n{excerpt}".strip()
        return excerpt


def _build_research_summary(
    *,
    query: str,
    region: str,
    items: list[WebResearchItem],
    fetch_pages: bool,
    playwright_attempted: int,
    playwright_success: int,
    playwright_errors: int,
    fallback_used: bool,
) -> str:
    lines = [
        f"# Web Research: {query}",
        "",
        f"- Provider: DuckDuckGo Instant Answer API",
        f"- Fallback: {'DuckDuckGo HTML SERP' if fallback_used else 'none'}",
        f"- Region: {region}",
        f"- Retrieved results: {len(items)}",
        f"- Playwright requested: {'yes' if fetch_pages else 'no'}",
        f"- Playwright pages attempted: {playwright_attempted}",
        f"- Playwright excerpts captured: {playwright_success}",
        f"- Playwright errors: {playwright_errors}",
        "",
        "## Results",
    ]
    for idx, item in enumerate(items, start=1):
        lines.extend(
            [
                f"### {idx}. {item.title or 'Result'}",
                f"- URL: {item.url}",
                f"- Snippet: {item.snippet or '-'}",
            ]
        )
        if item.page_excerpt:
            lines.append(f"- Playwright excerpt: {item.page_excerpt}")
        if item.page_error:
            lines.append(f"- Playwright error: {item.page_error}")
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_query(value: str) -> str:
    normalized = _clean_text(value)
    if len(normalized) < 3:
        raise ValueError("Query must have at least 3 characters")
    return normalized


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _host_label(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.strip() or parsed.path.strip() or "source"
    return host.lower()


def _merge_research_rows(primary: list[dict], fallback: list[dict], *, max_results: int) -> list[dict]:
    merged: list[dict] = []
    seen_urls: set[str] = set()
    for source_rows in (primary, fallback):
        for row in source_rows:
            if len(merged) >= max_results:
                return merged
            url = str(row.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(row)
    return merged


def _resolve_duckduckgo_result_url(href: str) -> str:
    value = html.unescape(str(href or "").strip())
    if not value:
        return ""
    if value.startswith("//"):
        value = f"https:{value}"

    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    uddg_values = query.get("uddg", [])
    if uddg_values:
        return unquote(uddg_values[0]).strip()
    if value.startswith("/l/?"):
        local_qs = parse_qs(value.split("?", 1)[1] if "?" in value else "")
        local_uddg = local_qs.get("uddg", [])
        if local_uddg:
            return unquote(local_uddg[0]).strip()
    return value


def _strip_html_tags(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", str(text or ""))
    return _clean_text(html.unescape(without_tags))
