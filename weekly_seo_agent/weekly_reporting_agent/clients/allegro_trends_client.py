from __future__ import annotations

import time
from datetime import date
from typing import Any
from urllib.parse import quote_plus

import requests


class AllegroTrendsClient:
    """Client for Allegro Trends API (token + time-series endpoint)."""

    def __init__(
        self,
        *,
        basic_auth_login: str,
        basic_auth_password: str,
        technical_account_login: str,
        technical_account_password: str,
        oauth_url: str = "https://oauth.allegrogroup.com/auth/oauth/token",
        api_base_url: str = "https://search-trends-service-prod-passive.allegrogroup.com",
        timeout_sec: int = 15,
    ) -> None:
        self.basic_auth_login = basic_auth_login.strip()
        self.basic_auth_password = basic_auth_password.strip()
        self.technical_account_login = technical_account_login.strip()
        self.technical_account_password = technical_account_password.strip()
        self.oauth_url = oauth_url.strip()
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_sec = max(5, int(timeout_sec))

        self._token: str = ""
        self._token_start_unix: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(
            self.basic_auth_login
            and self.basic_auth_password
            and self.technical_account_login
            and self.technical_account_password
            and self.oauth_url
            and self.api_base_url
        )

    def _get_token(self, max_retries: int = 3, retry_delay_sec: int = 3) -> str:
        if not self.configured:
            raise RuntimeError("Allegro Trends API credentials are missing.")

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "password",
            "username": quote_plus(self.technical_account_login),
            "password": quote_plus(self.technical_account_password),
        }

        last_error = "Unknown token error."
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.oauth_url,
                    data=data,
                    headers=headers,
                    auth=(self.basic_auth_login, self.basic_auth_password),
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                payload = response.json()
                token = str(payload.get("access_token", "")).strip()
                if token:
                    self._token = token
                    self._token_start_unix = time.time()
                    return token
                last_error = f"No access_token in response payload: {payload}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

            if attempt < max_retries:
                time.sleep(max(1, retry_delay_sec))

        raise RuntimeError(f"Failed to get Allegro Trends token: {last_error}")

    def _ensure_token(self) -> str:
        now = time.time()
        if self._token and (now - self._token_start_unix) < 3500:
            return self._token
        return self._get_token()

    def _fetch_raw_series(
        self,
        *,
        query: str,
        from_day: date,
        till_day: date,
        interval: str,
        exact: bool,
        escape_query: bool,
        measures: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], int]:
        token = self._ensure_token()
        query_encoded = quote_plus(str(query).strip())
        measures_param = ",".join(part.strip().upper() for part in measures if part.strip())
        url = (
            f"{self.api_base_url}/searches/time-series"
            f"?escapeQuery={'true' if escape_query else 'false'}"
            f"&from={from_day.isoformat()}"
            f"&till={till_day.isoformat()}"
            f"&interval={interval}"
            f"&exact={'true' if exact else 'false'}"
            f"&query={query_encoded}"
            f"&measures={measures_param}"
        )

        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=self.timeout_sec,
        )
        if response.status_code == 401:
            token = self._get_token()
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout_sec,
            )

        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)], response.status_code
        if isinstance(payload, dict):
            series = payload.get("data") or payload.get("rows") or payload.get("results") or []
            if isinstance(series, list):
                return [row for row in series if isinstance(row, dict)], response.status_code
        return [], response.status_code

    @staticmethod
    def _as_float(payload: dict[str, Any], key: str) -> float:
        raw = payload.get(key, 0.0)
        if raw is None:
            return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def fetch_query_summary(
        self,
        *,
        query: str,
        from_day: date,
        till_day: date,
        interval: str = "day",
        exact: bool = False,
        escape_query: bool = True,
        measures: tuple[str, ...] = ("VISIT", "PV", "OFFERS", "GMV", "DEALS"),
    ) -> dict[str, Any]:
        series, status_code = self._fetch_raw_series(
            query=query,
            from_day=from_day,
            till_day=till_day,
            interval=interval,
            exact=exact,
            escape_query=escape_query,
            measures=measures,
        )

        visit = 0.0
        pv = 0.0
        offers = 0.0
        deals = 0.0
        gmv = 0.0
        for row in series:
            visit += self._as_float(row, "visit")
            pv += self._as_float(row, "pv")
            offers += self._as_float(row, "offers")
            deals += self._as_float(row, "deals")
            gmv += self._as_float(row, "gmv")

        return {
            "query": query,
            "from": from_day.isoformat(),
            "till": till_day.isoformat(),
            "interval": interval,
            "visit": visit,
            "pv": pv,
            "offers": offers,
            "deals": deals,
            "gmv": gmv,
            "points": len(series),
            "http_code": status_code,
        }
