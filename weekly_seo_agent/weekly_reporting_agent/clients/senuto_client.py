from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import urljoin

import requests
from requests import Response

from weekly_seo_agent.weekly_reporting_agent.models import VisibilityPoint


class SenutoClient:
    """Configurable Senuto API client.

    The endpoint path is configurable because account setups may use different
    endpoint versions.
    """

    AUTH_ERROR_CODES = {401, 403, 418}

    def __init__(
        self,
        token: str,
        email: str,
        password: str,
        token_endpoint: str,
        base_url: str,
        domain: str,
        visibility_endpoint: str,
        fetch_mode: str = "topLevelDomain",
        country_id: int = 50,
        date_interval: str = "weekly",
        visibility_metric: str = "top10",
    ) -> None:
        self.token = token
        self.email = email
        self.password = password
        self.token_endpoint = token_endpoint
        self.base_url = base_url
        self.domain = domain
        self.visibility_endpoint = visibility_endpoint
        self.fetch_mode = fetch_mode
        self.country_id = country_id
        self.date_interval = date_interval
        self.visibility_metric = visibility_metric
        self._session_token = self._normalize_token(token)

    @staticmethod
    def _normalize_token(raw: str) -> str:
        token = raw.strip().strip("'\"")
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        return token

    @staticmethod
    def _json_headers() -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "weekly-seo-agent/0.1",
        }

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "weekly-seo-agent/0.1",
        }

    def _auth_url_candidates(self) -> list[str]:
        endpoint = (self.token_endpoint or "/api/users/token").strip()
        if endpoint.startswith(("http://", "https://")):
            return [endpoint]

        endpoint_normalized = "/" + endpoint.lstrip("/")
        endpoint_variants = [endpoint_normalized]
        if endpoint_normalized.startswith("/api/"):
            endpoint_variants.append("/" + endpoint_normalized[len("/api/"):])
        else:
            endpoint_variants.append("/api" + endpoint_normalized)
        endpoint_variants.extend(("/api/users/token", "/users/token"))

        base = self.base_url.strip().rstrip("/") or "https://api.senuto.com"
        base_variants = [base]
        if base.endswith("/api"):
            base_variants.append(base[:-4])
        else:
            base_variants.append(f"{base}/api")

        urls: list[str] = []
        seen: set[str] = set()
        for base_variant in base_variants:
            for endpoint_variant in endpoint_variants:
                effective_endpoint = endpoint_variant
                if base_variant.endswith("/api") and endpoint_variant.startswith("/api/"):
                    effective_endpoint = endpoint_variant[len("/api") :]
                candidate = urljoin(
                    f"{base_variant.rstrip('/')}/", effective_endpoint.lstrip("/")
                )
                if candidate in seen:
                    continue
                seen.add(candidate)
                urls.append(candidate)
        return urls

    @staticmethod
    def _extract_token(payload: object) -> str:
        if isinstance(payload, dict):
            for key in ("token", "access_token", "bearer_token"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            data = payload.get("data")
            if isinstance(data, dict):
                for key in ("token", "access_token", "bearer_token"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return ""

    @staticmethod
    def _response_payload(response: Response) -> object:
        try:
            return response.json()
        except ValueError:
            return None

    @classmethod
    def _response_message(cls, response: Response) -> str:
        payload = cls._response_payload(response)
        if isinstance(payload, dict):
            direct_message = payload.get("message")
            if isinstance(direct_message, str) and direct_message.strip():
                return direct_message.strip()

            data = payload.get("data")
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    nested_message = error.get("message")
                    if isinstance(nested_message, str) and nested_message.strip():
                        return nested_message.strip()
                nested_message = data.get("message")
                if isinstance(nested_message, str) and nested_message.strip():
                    return nested_message.strip()

            error = payload.get("error")
            if isinstance(error, dict):
                nested_message = error.get("message")
                if isinstance(nested_message, str) and nested_message.strip():
                    return nested_message.strip()

        text = response.text.strip()
        if not text:
            return "No response body."
        if len(text) > 240:
            return text[:237] + "..."
        return text

    def _authenticate(self, force_refresh: bool = False) -> str:
        if self._session_token and not force_refresh:
            return self._session_token

        if self._session_token and force_refresh and not (self.email and self.password):
            raise RuntimeError(
                "Senuto token rejected and cannot refresh automatically. "
                "Provide a valid SENUTO_TOKEN or SENUTO_EMAIL + SENUTO_PASSWORD."
            )

        if not (self.email and self.password):
            raise RuntimeError(
                "Senuto token missing. Provide SENUTO_TOKEN or SENUTO_EMAIL + SENUTO_PASSWORD."
            )

        payload = {"email": self.email, "password": self.password}
        attempts: list[str] = []

        for token_url in self._auth_url_candidates():
            try:
                response = requests.post(
                    token_url,
                    json=payload,
                    headers=self._json_headers(),
                    timeout=30,
                )
            except requests.RequestException as exc:
                attempts.append(f"{token_url} -> request error: {exc}")
                continue

            response_message = self._response_message(response)
            if response.ok:
                token = self._normalize_token(
                    self._extract_token(self._response_payload(response))
                )
                if token:
                    self._session_token = token
                    return token
                attempts.append(
                    f"{token_url} -> {response.status_code}: token missing in response ({response_message})"
                )
                continue

            attempts.append(
                f"{token_url} -> {response.status_code}: {response_message}"
            )

        details = " | ".join(attempts[:3])
        if len(attempts) > 3:
            details += " | ..."
        raise RuntimeError(f"Senuto authentication failed. {details}")

    def _get_with_auth_retry(self, url: str, params: dict[str, str | int]) -> requests.Response:
        return self._request_with_auth_retry(
            method="GET",
            url=url,
            params=params,
        )

    def _request_with_auth_retry(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        form_data: dict[str, Any] | None = None,
    ) -> requests.Response:
        token = self._authenticate(force_refresh=False)
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=self._headers(token),
                params=params,
                json=json_body,
                data=form_data,
                timeout=45,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Senuto API request failed: {exc}") from exc

        if response.status_code in self.AUTH_ERROR_CODES and self.email and self.password:
            token = self._authenticate(force_refresh=True)
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    headers=self._headers(token),
                    params=params,
                    json=json_body,
                    data=form_data,
                    timeout=45,
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Senuto API request failed after token refresh: {exc}") from exc

        if response.status_code in self.AUTH_ERROR_CODES:
            message = self._response_message(response)
            raise RuntimeError(
                "Senuto authorization failed for data endpoint. "
                f"Status: {response.status_code}. Details: {message}. "
                "Set valid SENUTO_TOKEN or SENUTO_EMAIL + SENUTO_PASSWORD."
            )
        if not response.ok:
            message = self._response_message(response)
            raise RuntimeError(
                f"Senuto API request failed ({response.status_code}): {message}"
            )
        return response

    def _post_json_with_auth_retry(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        return self._request_with_auth_retry(
            method="POST",
            url=url,
            json_body=payload,
        )

    def _post_form_with_auth_retry(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> requests.Response:
        return self._request_with_auth_retry(
            method="POST",
            url=url,
            form_data=payload,
        )

    def fetch_token(self, force_refresh: bool = False) -> str:
        return self._authenticate(force_refresh=force_refresh)

    def fetch_visibility(
        self,
        start_date: date,
        end_date: date,
    ) -> list[VisibilityPoint]:
        endpoint = self.visibility_endpoint.strip()
        if not endpoint:
            return []

        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        params = {
            "domain": self.domain,
            "fetch_mode": self.fetch_mode,
            "country_id": self.country_id,
            "date_min": start_date.isoformat(),
            "date_max": end_date.isoformat(),
            "date_interval": self.date_interval,
            "isDataReadyToLoad": "true",
            "date_from": start_date.isoformat(),
            "date_to": end_date.isoformat(),
        }

        response = self._get_with_auth_retry(url=url, params=params)
        return self._parse_visibility_payload(
            response.json(),
            fallback_day=end_date,
        )

    def fetch_countries_list(self) -> list[dict[str, Any]]:
        endpoint = "/api/visibility_analysis/app/getCountriesList"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        response = self._request_with_auth_retry(method="GET", url=url)
        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "id": int(float(row.get("id", 0) or 0)),
                    "name": str(row.get("name", "")).strip(),
                    "code": str(row.get("code", "")).strip().lower(),
                }
            )
        return out

    def resolve_country_id(
        self,
        country_code: str,
        fallback_country_id: int | None = None,
    ) -> int:
        target = country_code.strip().lower()
        if not target:
            return int(fallback_country_id or self.country_id)
        try:
            rows = self.fetch_countries_list()
        except Exception:
            return int(fallback_country_id or self.country_id)

        # Prefer PL new-base when both exist.
        if target == "pl":
            for code in ("pl_new", "pl"):
                for row in rows:
                    if str(row.get("code", "")).strip().lower() == code:
                        value = int(float(row.get("id", 0) or 0))
                        if value > 0:
                            return value

        for row in rows:
            if str(row.get("code", "")).strip().lower() == target:
                value = int(float(row.get("id", 0) or 0))
                if value > 0:
                    return value
        return int(fallback_country_id or self.country_id)

    def fetch_competitors_overview(self, top_n: int = 10) -> list[dict[str, Any]]:
        endpoint = "/api/visibility_analysis/reports/competitors/getData"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        payload = {
            "domain": self.domain,
            "fetch_mode": self.fetch_mode,
            "country_id": self.country_id,
        }
        response = self._post_json_with_auth_retry(url=url, payload=payload)
        data = response.json().get("data", [])
        if not isinstance(data, list):
            return []

        rows: list[dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            if bool(row.get("is_main_domain")):
                continue
            statistics = row.get("statistics", {})
            if not isinstance(statistics, dict):
                statistics = {}
            visibility = statistics.get("visibility", {})
            top10 = statistics.get("top10", {})
            domain_rank = statistics.get("domain_rank", {})
            rows.append(
                {
                    "domain": str(row.get("domain", "")).strip(),
                    "common_keywords": float(row.get("common_keywords", 0.0) or 0.0),
                    "visibility_current": float(visibility.get("current", 0.0) or 0.0),
                    "visibility_previous": float(visibility.get("previous", 0.0) or 0.0),
                    "visibility_diff": float(visibility.get("diff", 0.0) or 0.0),
                    "visibility_diff_pct": float(visibility.get("percent", 0.0) or 0.0) * 100.0,
                    "top10_current": float(top10.get("current", 0.0) or 0.0),
                    "domain_rank_current": float(domain_rank.get("current", 0.0) or 0.0),
                }
            )
        rows.sort(
            key=lambda item: (item.get("common_keywords", 0.0), item.get("visibility_current", 0.0)),
            reverse=True,
        )
        return rows[: max(1, top_n)]

    @staticmethod
    def _domain_matches(domain: str, allowed_domains: list[str]) -> bool:
        candidate = domain.strip().lower()
        if not candidate:
            return False
        for raw in allowed_domains:
            allowed = raw.strip().lower()
            if not allowed:
                continue
            if candidate == allowed or candidate.endswith(f".{allowed}") or allowed.endswith(f".{candidate}"):
                return True
        return False

    def fetch_competitors_overview_for_domains(
        self,
        competitors_domains: list[str],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        rows = self.fetch_competitors_overview(top_n=max(50, top_n))
        if not competitors_domains:
            return rows[: max(1, top_n)]
        filtered = [
            row
            for row in rows
            if isinstance(row, dict)
            and self._domain_matches(str(row.get("domain", "")), competitors_domains)
        ]
        return filtered[: max(1, top_n)]

    def fetch_competitor_keyword_gap(
        self,
        competitors_domains: list[str],
        top_n: int = 10,
        gte: int = 1,
        lte: int = 50,
    ) -> list[dict[str, Any]]:
        endpoint = "/api/visibility_analysis/tools/competitors_analysis/getData"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))

        cleaned_domains = [domain.strip().lower() for domain in competitors_domains if domain.strip()]
        if not cleaned_domains:
            return []

        payload = {
            "main_domain": {"domain": self.domain, "gte": gte, "lte": lte},
            "country_id": self.country_id,
            "mode": "common_keywords",
            "competitors_domains": [
                {"domain": domain, "gte": gte, "lte": lte}
                for domain in cleaned_domains
            ],
        }
        response = self._post_json_with_auth_retry(url=url, payload=payload)
        data = response.json().get("data", [])
        if not isinstance(data, list):
            return []

        out: list[dict[str, Any]] = []
        target_pos_key = f"{self.domain}_pos"
        for row in data:
            if not isinstance(row, dict):
                continue
            keyword = str(row.get("keyword", "")).strip()
            if not keyword:
                continue
            if "allegro" in keyword.lower():
                continue

            try:
                target_pos = float(row.get(target_pos_key, 0.0) or 0.0)
            except (TypeError, ValueError):
                target_pos = 0.0

            best_competitor = ""
            best_competitor_pos = 0.0
            for competitor in cleaned_domains:
                key = f"{competitor}_pos"
                try:
                    competitor_pos = float(row.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if competitor_pos <= 0:
                    continue
                if best_competitor_pos <= 0 or competitor_pos < best_competitor_pos:
                    best_competitor = competitor
                    best_competitor_pos = competitor_pos

            if not best_competitor or best_competitor_pos <= 0:
                continue

            searches = float(row.get("searches", 0.0) or 0.0)
            difficulty = float(row.get("difficulty", 0.0) or 0.0)
            gap_positions = (
                (target_pos - best_competitor_pos)
                if target_pos > 0
                else (50.0 - best_competitor_pos + 1.0)
            )
            gap_score = searches * max(1.0, gap_positions) / max(1.0, difficulty)

            # Strict opportunity only: skip rows where Allegro is already equal/better.
            if target_pos > 0 and target_pos <= best_competitor_pos:
                continue

            out.append(
                {
                    "keyword": keyword,
                    "searches": searches,
                    "difficulty": difficulty,
                    "target_pos": target_pos,
                    "best_competitor": best_competitor,
                    "best_competitor_pos": best_competitor_pos,
                    "gap_positions": gap_positions,
                    "gap_score": gap_score,
                    "gap_type": "opportunity",
                }
            )

        out.sort(key=lambda item: item.get("gap_score", 0.0), reverse=True)
        return out[: max(1, top_n)]

    def fetch_positions_wins_losses(self, top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
        base_payload = {
            "domain": self.domain,
            "fetch_mode": self.fetch_mode,
            "country_id": str(self.country_id),
            "page": "1",
            "limit": str(max(1, top_n)),
        }
        out: dict[str, list[dict[str, Any]]] = {"wins": [], "losses": []}
        for label, endpoint in (
            ("wins", "/api/visibility_analysis/reports/positions/getWins"),
            ("losses", "/api/visibility_analysis/reports/positions/getLosses"),
        ):
            url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
            response = self._post_form_with_auth_retry(url=url, payload=base_payload)
            rows = response.json().get("data", [])
            if not isinstance(rows, list):
                continue
            parsed: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                keyword = str(row.get("keyword", "")).strip()
                stats = row.get("statistics", {})
                if not keyword or not isinstance(stats, dict):
                    continue
                visibility = stats.get("visibility", {})
                if not isinstance(visibility, dict):
                    visibility = {}
                parsed.append(
                    {
                        "keyword": keyword,
                        "visibility_current": float(visibility.get("current", 0.0) or 0.0),
                        "visibility_previous": float(visibility.get("previous", 0.0) or 0.0),
                        "visibility_diff": float(visibility.get("diff", 0.0) or 0.0),
                        "visibility_diff_pct": float(visibility.get("percent", 0.0) or 0.0) * 100.0,
                    }
                )
            out[label] = parsed[: max(1, top_n)]
        return out

    def fetch_history_acquired_lost(
        self,
        start_date: date,
        end_date: date,
        top_n: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {
            "keywords_acquired": [],
            "keywords_lost": [],
            "urls_acquired": [],
            "urls_lost": [],
        }

        keyword_payload = {
            "country_id": str(self.country_id),
            "date_max": end_date.isoformat(),
            "date_min": start_date.isoformat(),
            "days_compare_mode": "yesterday",
            "domain": self.domain,
            "fetch_mode": self.fetch_mode,
            "isDataReadyToLoad": True,
            "limit": max(1, top_n),
            "order": {"dir": "desc", "prop": "statistics.visibility.diff"},
            "page": 1,
        }
        for label, endpoint in (
            ("keywords_acquired", "/api/visibility_analysis/reports/history/keywords/getAcquired"),
            ("keywords_lost", "/api/visibility_analysis/reports/history/keywords/getLost"),
        ):
            url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
            response = self._post_json_with_auth_retry(url=url, payload=keyword_payload)
            rows = response.json().get("data", [])
            if not isinstance(rows, list):
                continue
            parsed: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                keyword = str(row.get("keyword", "")).strip()
                if not keyword:
                    continue
                stats = row.get("statistics", {})
                visibility = stats.get("visibility", {}) if isinstance(stats, dict) else {}
                parsed.append(
                    {
                        "keyword": keyword,
                        "visibility_current": float(visibility.get("current", 0.0) or 0.0),
                        "visibility_diff": float(visibility.get("diff", 0.0) or 0.0),
                    }
                )
            out[label] = parsed[: max(1, top_n)]

        url_payload = {
            "domain": self.domain,
            "date_min": start_date.isoformat(),
            "date_max": end_date.isoformat(),
            "country_id": str(self.country_id),
            "order[prop]": "statistics.visibility.current",
            "order[dir]": "desc",
            "fetch_mode": self.fetch_mode,
            "page": "1",
            "limit": str(max(1, top_n)),
        }
        for label, endpoint in (
            ("urls_acquired", "/api/visibility_analysis/reports/history/urls/getAcquired"),
            ("urls_lost", "/api/visibility_analysis/reports/history/urls/getLost"),
        ):
            url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
            response = self._post_form_with_auth_retry(url=url, payload=url_payload)
            rows = response.json().get("data", [])
            if not isinstance(rows, list):
                continue
            parsed: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                page_url = str(row.get("url", "") or row.get("domain_url", "")).strip()
                stats = row.get("statistics", {})
                visibility = stats.get("visibility", {}) if isinstance(stats, dict) else {}
                if not page_url:
                    continue
                parsed.append(
                    {
                        "url": page_url,
                        "visibility_current": float(visibility.get("current", 0.0) or 0.0),
                        "visibility_diff": float(visibility.get("diff", 0.0) or 0.0),
                    }
                )
            out[label] = parsed[: max(1, top_n)]
        return out

    def fetch_direct_answers_overview(self, top_n: int = 10) -> list[dict[str, Any]]:
        endpoint = "/api/visibility_analysis/tools/direct_answers/getData"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        payload = {
            "domain": self.domain,
            "country_id": str(self.country_id),
            "page": "1",
            "limit": str(max(1, top_n)),
        }
        response = self._post_form_with_auth_retry(url=url, payload=payload)
        rows = response.json().get("data", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            keyword = str(row.get("keyword", "")).strip()
            if not keyword:
                continue
            out.append(
                {
                    "keyword": keyword,
                    "feature": str(row.get("feature", "")).strip(),
                    "searches": float(row.get("searches", 0.0) or 0.0),
                    "position": float(row.get("position", 0.0) or 0.0),
                }
            )
        return out[: max(1, top_n)]

    def fetch_domain_seasonality(self) -> dict[str, Any]:
        endpoint = "/api/visibility_analysis/reports/domain_seasonality/getSeasonalityChartData"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        params = {
            "domain": self.domain,
            "fetch_mode": self.fetch_mode,
            "country_id": self.country_id,
        }
        response = self._request_with_auth_retry(method="GET", url=url, params=params)
        payload = response.json()
        raw = payload.get("data", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw, dict):
            return {"trend_values": []}

        trend_values: list[float] = []
        for index in range(1, 13):
            key = f"trend_{index}"
            try:
                trend_values.append(float(raw.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                trend_values.append(0.0)

        if not trend_values:
            return {"trend_values": []}
        if all(abs(value) < 1e-9 for value in trend_values):
            return {"trend_values": []}
        peak_idx = max(range(len(trend_values)), key=lambda idx: trend_values[idx])
        low_idx = min(range(len(trend_values)), key=lambda idx: trend_values[idx])
        return {
            "trend_values": trend_values,
            "peak_month": int(peak_idx + 1),
            "peak_value": float(trend_values[peak_idx]),
            "low_month": int(low_idx + 1),
            "low_value": float(trend_values[low_idx]),
        }

    def fetch_market_ranking(self, top_n: int = 10) -> list[dict[str, Any]]:
        endpoint = "/api/visibility_analysis/tools/domains_ranking/getRankingData"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        payload = {
            "country_id": str(self.country_id),
            "match_mode": "domain",
            "page": "1",
            "limit": str(max(1, top_n)),
        }
        response = self._post_form_with_auth_retry(url=url, payload=payload)
        rows = response.json().get("data", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("domain", "")).strip()
            statistics = row.get("statistics", {})
            if not domain or not isinstance(statistics, dict):
                continue
            rank = statistics.get("rank", {})
            visibility = statistics.get("visibility", {})
            out.append(
                {
                    "domain": domain,
                    "category": str(row.get("category", "")).strip(),
                    "share": float(row.get("share", 0.0) or 0.0),
                    "rank_current": float(rank.get("recent_value", 0.0) or rank.get("current", 0.0) or 0.0),
                    "rank_previous": float(rank.get("older_value", 0.0) or rank.get("previous", 0.0) or 0.0),
                    "visibility_current": float(visibility.get("recent_value", 0.0) or visibility.get("current", 0.0) or 0.0),
                    "visibility_previous": float(visibility.get("older_value", 0.0) or visibility.get("previous", 0.0) or 0.0),
                }
            )
        return out[: max(1, top_n)]

    def fetch_keyword_trending(self, top_n: int = 10) -> list[dict[str, Any]]:
        endpoint = "/api/keywords_analysis/reports/keywords/getTrending"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        payload = {
            "parameters": [
                {
                    "data_fetch_mode": "domain",
                    "value": [self.domain],
                }
            ],
            "country_id": str(self.country_id),
            "match_mode": "wide",
            "filtering": [{"filters": []}],
            "page": 1,
            "limit": max(1, top_n),
        }
        response = self._post_json_with_auth_retry(url=url, payload=payload)
        rows = response.json().get("data", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            keyword = str(row.get("keyword", "")).strip()
            if not keyword:
                continue
            out.append(
                {
                    "keyword": keyword,
                    "searches": float(row.get("searches", 0.0) or 0.0),
                    "growth": float(row.get("growth", 0.0) or row.get("trend", 0.0) or 0.0),
                }
            )
        out.sort(key=lambda item: item.get("searches", 0.0), reverse=True)
        return out[: max(1, top_n)]

    def fetch_serp_volatility(
        self,
        keyword: str,
        start_date: date,
        end_date: date,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        candidate = keyword.strip()
        if not candidate:
            return []
        endpoint = "/api/visibility_analysis/tools/serp_history/getData"
        url = urljoin(f"{self.base_url.rstrip('/')}/", endpoint.lstrip("/"))
        payload = {
            "country_id": str(self.country_id),
            "keyword": candidate,
            "top10_only": "false",
            "date_min": start_date.isoformat(),
            "date_max": end_date.isoformat(),
            "page": "1",
            "limit": str(max(1, top_n)),
        }
        response = self._post_form_with_auth_retry(url=url, payload=payload)
        rows = response.json().get("data", [])
        if not isinstance(rows, list):
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("domain", "")).strip()
            positions = row.get("positions", [])
            monthly_positions = row.get("monthly_positions", [])
            if not domain:
                continue
            pos_values: list[float] = []
            if isinstance(positions, list):
                for value in positions:
                    try:
                        pos_values.append(float(value))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(positions, dict):
                for value in positions.values():
                    try:
                        pos_values.append(float(value))
                    except (TypeError, ValueError):
                        continue
            volatility = 0.0
            if len(pos_values) >= 2:
                deltas = [
                    abs(pos_values[idx] - pos_values[idx - 1])
                    for idx in range(1, len(pos_values))
                ]
                volatility = sum(deltas) / len(deltas)
            monthly_points = 0
            if isinstance(monthly_positions, list):
                monthly_points = len(monthly_positions)
            elif isinstance(monthly_positions, dict):
                monthly_points = len(monthly_positions)
            out.append(
                {
                    "keyword": candidate,
                    "domain": domain,
                    "volatility": volatility,
                    "positions_points": len(pos_values),
                    "monthly_points": monthly_points,
                }
            )
        out.sort(key=lambda item: item.get("volatility", 0.0), reverse=True)
        return out[: max(1, top_n)]

    def _parse_visibility_payload(
        self,
        payload: object,
        fallback_day: date | None = None,
    ) -> list[VisibilityPoint]:
        rows: object = payload
        if isinstance(payload, dict):
            data_payload = payload.get("data")
            if isinstance(data_payload, dict):
                rows = [{"date": key, **value} for key, value in data_payload.items() if isinstance(value, dict)]
            elif isinstance(data_payload, list):
                rows = data_payload
            elif isinstance(payload.get("items"), list):
                rows = payload["items"]
            elif isinstance(payload.get("results"), list):
                rows = payload["results"]
            else:
                rows = []

        points: list[VisibilityPoint] = []
        if not isinstance(rows, list):
            return points

        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_date = row.get("date") or row.get("day")
            raw_visibility = row.get(self.visibility_metric)
            if isinstance(raw_visibility, dict):
                raw_visibility = raw_visibility.get("current")
            if raw_visibility is None:
                raw_visibility = row.get("visibility")
            if isinstance(raw_visibility, dict):
                raw_visibility = raw_visibility.get("current")
            if raw_visibility is None:
                raw_visibility = row.get("value")
            if raw_visibility is None:
                raw_visibility = row.get("top10")
            if raw_visibility is None:
                raw_visibility = row.get("top3")
            if raw_visibility is None:
                raw_visibility = row.get("top50")
            if raw_visibility is None:
                nested_data = row.get("data")
                if isinstance(nested_data, dict):
                    distribution = nested_data.get("positions_distribution_top50")
                    if isinstance(distribution, dict):
                        counts: list[float] = []
                        for rank_raw, value_raw in distribution.items():
                            try:
                                rank = int(str(rank_raw))
                                value = float(value_raw)
                            except (TypeError, ValueError):
                                continue
                            if rank < 1 or rank > 50:
                                continue
                            counts.append((rank, value))

                        if counts:
                            rank_map = {rank: value for rank, value in counts}
                            metric = self.visibility_metric.lower()
                            if metric == "top3":
                                raw_visibility = sum(
                                    rank_map.get(rank, 0.0) for rank in range(1, 4)
                                )
                            elif metric == "top50":
                                raw_visibility = sum(
                                    rank_map.get(rank, 0.0) for rank in range(1, 51)
                                )
                            else:
                                raw_visibility = sum(
                                    rank_map.get(rank, 0.0) for rank in range(1, 11)
                                )
            if raw_date is None and fallback_day is not None:
                raw_date = fallback_day.isoformat()
            if raw_date is None or raw_visibility is None:
                continue

            try:
                day = date.fromisoformat(str(raw_date)[:10])
                visibility = float(raw_visibility)
            except (TypeError, ValueError):
                continue

            points.append(VisibilityPoint(day=day, visibility=visibility))

        points.sort(key=lambda point: point.day)
        return points
