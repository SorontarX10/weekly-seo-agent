from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

from weekly_seo_agent.models import DateWindow, ExternalSignal

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

COMPETITOR_EVENT_TOKENS = (
    "temu",
    "amazon",
    "aliexpress",
    "shein",
    "ebay",
    "ceneo",
    "olx",
    "empik",
    "media expert",
    "rtv euro agd",
    "x-kom",
    "morele",
)

SEO_UPDATE_ANALYSIS_RSS_URLS = (
    "https://www.searchenginejournal.com/feed/",
    "https://searchengineland.com/feed",
    "https://www.seroundtable.com/index.xml",
)

SEO_UPDATE_KEYWORDS = (
    "core update",
    "discover update",
    "google update",
    "ranking volatility",
    "search update",
    "algorithm update",
)

MAJOR_GLOBAL_EVENT_TOKENS = (
    "war",
    "energy crisis",
    "recession",
    "inflation",
    "interest rate",
    "tariff",
    "sanction",
    "earthquake",
    "flood",
    "pandemic",
    "black friday",
    "cyber monday",
)

CATEGORY_RELEVANCE_TOKENS = (
    "ecommerce",
    "e-commerce",
    "retail",
    "marketplace",
    "shopping",
    "delivery",
    "logistics",
    "vat",
    "tax",
    "allegro",
    "temu",
    "amazon",
    "shein",
    "ceneo",
    "olx",
)


class ExternalSignalsClient:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        weather_label: str,
        market_country_code: str,
        status_endpoint: str,
        blog_rss_url: str,
        holidays_country_code: str,
        holidays_api_base_url: str,
        holidays_language_code: str,
        news_scraping_enabled: bool,
        news_rss_urls_pl: tuple[str, ...],
        news_rss_urls_global: tuple[str, ...],
        news_html_urls_pl: tuple[str, ...],
        news_html_urls_global: tuple[str, ...],
        news_keywords: tuple[str, ...],
        news_max_signals: int,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.weather_label = weather_label
        self.market_country_code = market_country_code.strip().upper() or "PL"
        self.status_endpoint = status_endpoint
        self.blog_rss_url = blog_rss_url
        self.holidays_country_code = holidays_country_code
        self.holidays_api_base_url = holidays_api_base_url.rstrip("/")
        self.holidays_language_code = holidays_language_code

        self.news_scraping_enabled = news_scraping_enabled
        self.news_rss_urls_pl = news_rss_urls_pl
        self.news_rss_urls_global = news_rss_urls_global
        self.news_html_urls_pl = news_html_urls_pl
        self.news_html_urls_global = news_html_urls_global
        self.news_keywords = tuple(keyword.lower() for keyword in news_keywords)
        self.news_max_signals = news_max_signals
        self._subdivision_name_cache: dict[str, str] | None = None
        self._http_retry_attempts = 3
        self._http_retry_backoff_sec = 1.2

    def _http_get(
        self,
        url: str,
        *,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self._http_retry_attempts + 1):
            try:
                request_kwargs: dict[str, object] = {"timeout": timeout}
                if params is not None:
                    request_kwargs["params"] = params
                if headers is not None:
                    request_kwargs["headers"] = headers
                response = requests.get(url, **request_kwargs)
                status_code = int(getattr(response, "status_code", 200) or 200)
                if status_code in {429, 500, 502, 503, 504} and attempt < self._http_retry_attempts:
                    time.sleep(self._http_retry_backoff_sec * attempt)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self._http_retry_attempts:
                    break
                time.sleep(self._http_retry_backoff_sec * attempt)
        raise RuntimeError(f"GET failed for {url}: {last_error}")

    def _country_news_rss_url(self) -> str:
        country_map = {
            "PL": ("pl", "PL", "PL:pl", "allegro OR ecommerce OR retail OR marketplace OR logistyka"),
            "CZ": ("cs", "CZ", "CZ:cs", "allegro OR ecommerce OR retail OR marketplace OR logistika"),
            "SK": ("sk", "SK", "SK:sk", "allegro OR ecommerce OR retail OR marketplace OR logistika"),
            "HU": ("hu", "HU", "HU:hu", "allegro OR ecommerce OR retail OR marketplace OR logisztika"),
        }
        hl, gl, ceid, query = country_map.get(
            self.market_country_code,
            ("en-US", "US", "US:en", "ecommerce OR retail OR marketplace OR logistics"),
        )
        return (
            "https://news.google.com/rss/search?q="
            + quote_plus(query)
            + f"&hl={hl}&gl={gl}&ceid={ceid}"
        )

    def collect(
        self,
        current_window: DateWindow,
        previous_window: DateWindow,
    ) -> tuple[list[ExternalSignal], dict[str, float]]:
        signals: list[ExternalSignal] = []
        weather_summary: dict[str, float] = {}
        try:
            weather_summary = self._weather_summary(current_window, previous_window)
            weather_summary.update(
                self._weather_forecast_summary(reference_day=current_window.end, days=7)
            )
        except Exception as exc:
            signals.append(
                ExternalSignal(
                    source="Weather",
                    day=current_window.end,
                    title="Weather source degraded",
                    details=f"Weather source failed, using partial context only: {exc}",
                    severity="medium",
                )
            )

        for fetch_name, fetch_fn in (
            ("Google Search Status", lambda: self._google_status_signals(previous_window.start)),
            ("Google Search Central Blog", lambda: self._google_blog_signals(previous_window.start)),
            ("SEO update analysis", lambda: self._seo_update_analysis_signals(previous_window.start)),
            (
                "Public Holidays",
                lambda: self._holiday_signals(
                    start=current_window.start,
                    end=current_window.end + timedelta(days=60),
                ),
            ),
        ):
            try:
                signals.extend(fetch_fn())
            except Exception as exc:
                signals.append(
                    ExternalSignal(
                        source=fetch_name,
                        day=current_window.end,
                        title=f"{fetch_name} source degraded",
                        details=f"Source call failed and was skipped: {exc}",
                        severity="medium",
                    )
                )

        if self.news_scraping_enabled:
            try:
                signals.extend(self._news_signals(previous_window.start))
            except Exception as exc:
                signals.append(
                    ExternalSignal(
                        source="News",
                        day=current_window.end,
                        title="News sources degraded",
                        details=f"News scraping failed and was skipped: {exc}",
                        severity="medium",
                    )
                )

        temp_diff = weather_summary.get("avg_temp_diff_c", 0.0)
        precip_change_pct = weather_summary.get("precip_change_pct", 0.0)
        if abs(temp_diff) >= 3.0:
            direction = "higher" if temp_diff > 0 else "lower"
            signals.append(
                ExternalSignal(
                    source="Weather",
                    day=current_window.end,
                    title=f"Temperature anomaly in {self.weather_label}",
                    details=(
                        f"Average temperature was {abs(temp_diff):.1f}C {direction} vs previous 28 days."
                    ),
                    severity="medium",
                )
            )

        if abs(precip_change_pct) >= 40.0:
            direction = "up" if precip_change_pct > 0 else "down"
            signals.append(
                ExternalSignal(
                    source="Weather",
                    day=current_window.end,
                    title=f"Precipitation change in {self.weather_label}",
                    details=(
                        f"Total precipitation is {direction} {abs(precip_change_pct):.1f}% vs previous 28 days."
                    ),
                    severity="medium",
                )
            )

        forecast_avg = float(weather_summary.get("forecast_avg_temp_c", 0.0))
        forecast_precip = float(weather_summary.get("forecast_precip_mm", 0.0))
        forecast_start = str(weather_summary.get("forecast_start", "")).strip()
        forecast_end = str(weather_summary.get("forecast_end", "")).strip()
        if forecast_start and forecast_end:
            if abs(forecast_avg - weather_summary.get("avg_temp_current_c", 0.0)) >= 2.5:
                signals.append(
                    ExternalSignal(
                        source="Weather forecast",
                        day=current_window.end + timedelta(days=1),
                        title=f"7-day weather outlook ({forecast_start} to {forecast_end})",
                        details=(
                            f"Forecast avg temp {forecast_avg:+.1f}C, total precipitation {forecast_precip:.1f}mm."
                        ),
                        severity="medium",
                    )
                )

        signals.sort(key=lambda item: (item.day, item.source), reverse=True)
        return signals, weather_summary

    def _news_signals(self, since: date) -> list[ExternalSignal]:
        signals: list[ExternalSignal] = []

        country_url = self._country_news_rss_url()
        signals.extend(
            self._rss_news_signals(
                url=country_url,
                since=since,
                geo_label=self.market_country_code,
            )
        )

        for url in self.news_rss_urls_pl:
            signals.extend(self._rss_news_signals(url=url, since=since, geo_label="PL"))
        for url in self.news_rss_urls_global:
            signals.extend(self._rss_news_signals(url=url, since=since, geo_label="GLOBAL"))

        for url in self.news_html_urls_pl:
            signals.extend(self._html_news_signals(url=url, geo_label="PL"))
        for url in self.news_html_urls_global:
            signals.extend(self._html_news_signals(url=url, geo_label="GLOBAL"))

        deduped: list[ExternalSignal] = []
        seen: set[tuple[str, str]] = set()
        for signal in sorted(signals, key=lambda item: item.day, reverse=True):
            canonical_title = re.sub(
                r"\s*[-|:]\s*[^-|:]{1,80}$",
                "",
                signal.title.lower().strip(),
            ).strip() or signal.title.lower().strip()
            key = (canonical_title, signal.day.isoformat())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(signal)
            if len(deduped) >= self.news_max_signals:
                break

        return deduped

    def _rss_news_signals(self, url: str, since: date, geo_label: str) -> list[ExternalSignal]:
        try:
            response = self._http_get(url, timeout=20)
            root = ET.fromstring(response.text)
        except Exception:
            return []

        source_host = urlparse(url).netloc or "rss"
        signals: list[ExternalSignal] = []

        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            description = (item.findtext("description") or "").strip()
            pub_raw = (item.findtext("pubDate") or item.findtext("published") or "").strip()

            pub_dt = self._parse_datetime(pub_raw)
            if pub_dt is not None and pub_dt.date() < since:
                continue

            clean_description = html.unescape(re.sub(r"<[^>]+>", " ", description))
            clean_description = re.sub(r"\s+", " ", clean_description).strip()
            if len(clean_description) > 250:
                clean_description = clean_description[:247] + "..."

            if not self._is_news_relevant(title, clean_description):
                continue

            day = pub_dt.date() if pub_dt else date.today()
            severity = self._news_severity(f"{title} {clean_description}")

            signals.append(
                ExternalSignal(
                    source=f"News RSS {geo_label} ({source_host})",
                    day=day,
                    title=title or "News signal",
                    details=clean_description or "Potential market-impacting news signal.",
                    severity=severity,
                    url=link or None,
                )
            )

        return signals

    def _html_news_signals(self, url: str, geo_label: str) -> list[ExternalSignal]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            )
        }

        try:
            response = self._http_get(url, headers=headers, timeout=20)
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception:
            return []

        source_host = urlparse(url).netloc or "news-site"
        signals: list[ExternalSignal] = []
        seen_titles: set[str] = set()

        candidates = soup.select("h1, h2, h3, article a, main a")
        for element in candidates[:200]:
            title = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if len(title) < 18 or len(title) > 180:
                continue
            lower_title = title.lower()
            if lower_title in seen_titles:
                continue
            if not self._is_news_relevant(title, ""):
                continue

            seen_titles.add(lower_title)
            severity = self._news_severity(title)
            signals.append(
                ExternalSignal(
                    source=f"News HTML {geo_label} ({source_host})",
                    day=date.today(),
                    title=title,
                    details="Headline captured from monitored news source.",
                    severity=severity,
                    url=url,
                )
            )
            if len(signals) >= 5:
                break

        return signals

    def _is_news_relevant(self, title: str, details: str) -> bool:
        text = f"{title} {details}".lower()
        if any(token in text for token in CAMPAIGN_EVENT_TOKENS):
            return True
        if any(token in text for token in COMPETITOR_EVENT_TOKENS) and any(
            campaign in text for campaign in CAMPAIGN_EVENT_TOKENS
        ):
            return True
        if any(token in text for token in MAJOR_GLOBAL_EVENT_TOKENS):
            return True
        if any(token in text for token in CATEGORY_RELEVANCE_TOKENS):
            return True
        if not self.news_keywords:
            return True
        return any(keyword in text for keyword in self.news_keywords)

    @staticmethod
    def _news_severity(text: str) -> str:
        lowered = text.lower()
        high_impact = (
            "outage",
            "awaria",
            "strajk",
            "powodz",
            "flood",
            "war",
            "wojna",
            "recession",
            "clo",
            "vat",
            "tax",
            "regulation",
            "ban",
            "core update",
            "algorithm update",
        )
        medium_impact = (
            "google",
            "search",
            "e-commerce",
            "ecommerce",
            "allegro",
            "logistyka",
            "kurier",
            "inflacja",
            "stopy procentowe",
        )

        if any(token in lowered for token in high_impact):
            return "high"
        if any(token in lowered for token in medium_impact):
            return "medium"
        return "info"

    def _weather_summary(
        self,
        current_window: DateWindow,
        previous_window: DateWindow,
    ) -> dict[str, float]:
        start = previous_window.start
        end = current_window.end
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Europe/Warsaw",
        }

        response = self._http_get(url, params=params, timeout=30)
        payload = response.json()
        daily = payload.get("daily", {})

        days = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])

        stats: dict[date, tuple[float, float]] = {}
        for day_raw, temp_raw, precip_raw in zip(days, temps, precip, strict=False):
            try:
                day = date.fromisoformat(day_raw)
                stats[day] = (float(temp_raw), float(precip_raw))
            except (TypeError, ValueError):
                continue

        def summarize(window: DateWindow) -> tuple[float, float]:
            window_days = [
                stats[window.start + timedelta(days=offset)]
                for offset in range(window.days)
                if (window.start + timedelta(days=offset)) in stats
            ]
            if not window_days:
                return 0.0, 0.0
            avg_temp = sum(item[0] for item in window_days) / len(window_days)
            total_precip = sum(item[1] for item in window_days)
            return avg_temp, total_precip

        current_avg_temp, current_precip = summarize(current_window)
        previous_avg_temp, previous_precip = summarize(previous_window)

        precip_change_pct = 0.0
        if previous_precip:
            precip_change_pct = ((current_precip - previous_precip) / previous_precip) * 100

        return {
            "avg_temp_current_c": current_avg_temp,
            "avg_temp_previous_c": previous_avg_temp,
            "avg_temp_diff_c": current_avg_temp - previous_avg_temp,
            "precip_current_mm": current_precip,
            "precip_previous_mm": previous_precip,
            "precip_change_pct": precip_change_pct,
        }

    def _weather_forecast_summary(self, reference_day: date, days: int = 7) -> dict[str, float | str]:
        start_day = reference_day + timedelta(days=1)
        end_day = start_day + timedelta(days=max(1, days) - 1)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Europe/Warsaw",
        }
        try:
            response = self._http_get(url, params=params, timeout=30)
            payload = response.json()
            daily = payload.get("daily", {})
            t_max = daily.get("temperature_2m_max", []) or []
            t_min = daily.get("temperature_2m_min", []) or []
            precip = daily.get("precipitation_sum", []) or []
            if not (isinstance(t_max, list) and isinstance(t_min, list) and isinstance(precip, list)):
                return {}
            days_count = min(len(t_max), len(t_min), len(precip))
            if days_count <= 0:
                return {}

            avg_daily: list[float] = []
            min_temp: float | None = None
            max_temp: float | None = None
            total_precip = 0.0
            for index in range(days_count):
                try:
                    max_value = float(t_max[index])
                    min_value = float(t_min[index])
                    precip_value = float(precip[index])
                except (TypeError, ValueError):
                    continue
                avg_daily.append((max_value + min_value) / 2.0)
                total_precip += precip_value
                min_temp = min(min_temp, min_value) if min_temp is not None else min_value
                max_temp = max(max_temp, max_value) if max_temp is not None else max_value

            if not avg_daily:
                return {}

            return {
                "forecast_start": start_day.isoformat(),
                "forecast_end": end_day.isoformat(),
                "forecast_days": float(days_count),
                "forecast_avg_temp_c": sum(avg_daily) / len(avg_daily),
                "forecast_min_temp_c": float(min_temp if min_temp is not None else 0.0),
                "forecast_max_temp_c": float(max_temp if max_temp is not None else 0.0),
                "forecast_precip_mm": total_precip,
            }
        except Exception:
            return {}

    def _seo_update_analysis_signals(self, since: date) -> list[ExternalSignal]:
        signals: list[ExternalSignal] = []
        seen: set[tuple[str, str]] = set()

        for url in SEO_UPDATE_ANALYSIS_RSS_URLS:
            try:
                response = self._http_get(url, timeout=25)
                root = ET.fromstring(response.text)
            except Exception:
                continue

            source_host = urlparse(url).netloc or "seo-news"
            items = root.findall(".//item")
            if not items:
                items = root.findall(".//entry")

            for item in items[:60]:
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
                pub_raw = (
                    item.findtext("pubDate")
                    or item.findtext("published")
                    or item.findtext("updated")
                    or item.findtext("{http://www.w3.org/2005/Atom}updated")
                    or ""
                ).strip()
                pub_dt = self._parse_datetime(pub_raw)
                if pub_dt is None or pub_dt.date() < since:
                    continue

                text_blob = html.unescape(f"{title} {description}").lower()
                if not any(keyword in text_blob for keyword in SEO_UPDATE_KEYWORDS):
                    continue
                if "google" not in text_blob and "discover" not in text_blob:
                    continue

                positives: list[str] = []
                negatives: list[str] = []
                affected: list[str] = []

                if any(token in text_blob for token in ("quality", "useful", "improv", "better", "recovery")):
                    positives.append("quality/relevance improvements")
                if any(token in text_blob for token in ("volatility", "drop", "decline", "loss", "hit", "down")):
                    negatives.append("visibility volatility or declines")
                if "discover" in text_blob:
                    affected.append("Discover surfaces")
                if "news" in text_blob:
                    affected.append("news-oriented content")
                if "affiliate" in text_blob:
                    affected.append("affiliate pages")
                if any(token in text_blob for token in ("forum", "ugc")):
                    affected.append("UGC/forum pages")
                if any(token in text_blob for token in ("e-commerce", "ecommerce", "category page")):
                    affected.append("e-commerce/category pages")

                snippet = re.sub(r"<[^>]+>", " ", description)
                snippet = re.sub(r"\s+", " ", html.unescape(snippet)).strip()
                if len(snippet) > 180:
                    snippet = snippet[:177] + "..."

                parts: list[str] = []
                if positives:
                    parts.append("Positives: " + ", ".join(sorted(set(positives))))
                if negatives:
                    parts.append("Risks: " + ", ".join(sorted(set(negatives))))
                if affected:
                    parts.append("Likely affected: " + ", ".join(sorted(set(affected))))
                if snippet:
                    parts.append(f"Excerpt: {snippet}")
                details = " | ".join(parts) if parts else "Early SEO analysis related to Google update."

                dedupe_key = (source_host.lower(), title.lower())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                signals.append(
                    ExternalSignal(
                        source=f"SEO Update Analysis ({source_host})",
                        day=pub_dt.date(),
                        title=title or "SEO update analysis",
                        details=details,
                        severity="medium",
                        url=link or None,
                    )
                )

        signals.sort(key=lambda row: (row.day, row.source), reverse=True)
        return signals[:12]

    def _google_status_signals(self, since: date) -> list[ExternalSignal]:
        response = self._http_get(self.status_endpoint, timeout=30)
        payload = response.json()

        signals: list[ExternalSignal] = []
        if not isinstance(payload, list):
            return signals

        for incident in payload:
            if not isinstance(incident, dict):
                continue
            begin = self._parse_datetime(incident.get("begin"))
            if begin is None or begin.date() < since:
                continue

            updates = incident.get("updates") or []
            latest_update = updates[-1] if updates and isinstance(updates[-1], dict) else {}
            update_text = html.unescape(str(latest_update.get("text", ""))).strip()
            update_text = re.sub(r"\s+", " ", update_text)
            if len(update_text) > 400:
                update_text = update_text[:397] + "..."

            title = str(incident.get("external_desc") or incident.get("id") or "Google Search incident")
            severity = "info"
            lowered = title.lower()
            if "issue" in lowered or "outage" in lowered or "error" in lowered:
                severity = "high"
            elif "update" in lowered:
                severity = "medium"

            signals.append(
                ExternalSignal(
                    source="Google Search Status",
                    day=begin.date(),
                    title=title,
                    details=update_text or "Status update without additional details.",
                    severity=severity,
                    url=f"https://status.search.google.com/incidents/{incident.get('id', '')}",
                )
            )
        return signals

    def _google_blog_signals(self, since: date) -> list[ExternalSignal]:
        response = self._http_get(self.blog_rss_url, timeout=30)
        root = ET.fromstring(response.text)

        signals: list[ExternalSignal] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            description = (item.findtext("description") or "").strip()
            pub_dt = self._parse_datetime(pub_raw)
            if pub_dt is None or pub_dt.date() < since:
                continue

            clean_description = html.unescape(re.sub(r"<[^>]+>", " ", description))
            clean_description = re.sub(r"\s+", " ", clean_description).strip()
            if len(clean_description) > 300:
                clean_description = clean_description[:297] + "..."

            severity = "info"
            lowered = title.lower()
            if "core update" in lowered or "update" in lowered:
                severity = "medium"

            signals.append(
                ExternalSignal(
                    source="Google Search Central Blog",
                    day=pub_dt.date(),
                    title=title or "Google Search Central post",
                    details=clean_description or "New publication on Search Central Blog.",
                    severity=severity,
                    url=link or None,
                )
            )
        return signals

    def _holiday_signals(self, start: date, end: date) -> list[ExternalSignal]:
        signals: list[ExternalSignal] = []
        url = f"{self.holidays_api_base_url}/PublicHolidays"

        response = self._http_get(
            url,
            params={
                "countryIsoCode": self.holidays_country_code,
                "languageIsoCode": self.holidays_language_code,
                "validFrom": start.isoformat(),
                "validTo": end.isoformat(),
            },
            timeout=30,
        )
        holidays = response.json()
        if not isinstance(holidays, list):
            return signals

        subdivision_names = self._load_subdivision_names()
        for holiday in holidays:
            if not isinstance(holiday, dict):
                continue

            day_raw = holiday.get("startDate") or holiday.get("date")
            if not day_raw:
                continue
            try:
                holiday_day = date.fromisoformat(str(day_raw))
            except ValueError:
                continue
            if holiday_day < start or holiday_day > end:
                continue

            local_name = self._pick_localized_name(holiday.get("name"))
            if not local_name:
                local_name = str(holiday.get("id") or "Holiday")

            scope = str(holiday.get("regionalScope") or "").strip().lower()
            is_nationwide = bool(holiday.get("nationwide")) or scope == "national"

            details = "National holiday can affect demand and search behavior."
            if not is_nationwide:
                subdivision_codes = holiday.get("subdivisions")
                region_labels: list[str] = []
                if isinstance(subdivision_codes, list):
                    for row in subdivision_codes:
                        if not isinstance(row, dict):
                            continue
                        code = str(row.get("code", "")).strip()
                        if not code:
                            continue
                        region_labels.append(subdivision_names.get(code, code))
                if region_labels:
                    details = (
                        "Regional holiday can affect demand and traffic by area "
                        f"({', '.join(region_labels[:8])})."
                    )
                else:
                    details = "Regional holiday can affect demand and traffic by area."

            signals.append(
                ExternalSignal(
                    source="Public Holidays",
                    day=holiday_day,
                    title=f"Holiday: {local_name}",
                    details=details,
                    severity="info",
                )
            )

        return signals

    def _load_subdivision_names(self) -> dict[str, str]:
        if self._subdivision_name_cache is not None:
            return self._subdivision_name_cache

        mapping: dict[str, str] = {}
        url = f"{self.holidays_api_base_url}/Subdivisions"
        response = self._http_get(
            url,
            params={"countryIsoCode": self.holidays_country_code},
            timeout=30,
        )
        payload = response.json()
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("code", "")).strip()
                if not code:
                    continue
                label = self._pick_localized_name(row.get("name")) or code
                mapping[code] = label

        self._subdivision_name_cache = mapping
        return mapping

    def _pick_localized_name(self, payload: object) -> str:
        if isinstance(payload, list):
            target_lang = self.holidays_language_code.lower()
            for row in payload:
                if not isinstance(row, dict):
                    continue
                row_lang = str(row.get("language", "")).strip().lower()
                if row_lang == target_lang:
                    text = str(row.get("text", "")).strip()
                    if text:
                        return text
            for row in payload:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("text", "")).strip()
                if text:
                    return text
        return ""

    @staticmethod
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
