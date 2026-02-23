from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Sequence

import httplib2
from google.auth.credentials import Credentials
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2 import service_account
try:
    from google_auth_httplib2 import AuthorizedHttp
except Exception:  # pragma: no cover
    AuthorizedHttp = None
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from weekly_seo_agent.weekly_reporting_agent.models import DateWindow, MetricRow, MetricSummary


class GSCClient:
    """Thin wrapper for Search Console Search Analytics API."""

    SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
    HTTP_TIMEOUT_SEC = 30
    API_RETRIES = 3
    REQUEST_HARD_TIMEOUT_SEC = 45

    def __init__(
        self,
        site_url: str,
        credentials_path: str = "",
        oauth_client_secret_path: str = "",
        oauth_refresh_token: str = "",
        oauth_token_uri: str = "https://oauth2.googleapis.com/token",
        country_filter: str = "",
        row_limit: int = 3000,
    ) -> None:
        self.site_url = site_url
        self.credentials_path = credentials_path
        self.oauth_client_secret_path = oauth_client_secret_path
        self.oauth_refresh_token = oauth_refresh_token
        self.oauth_token_uri = oauth_token_uri
        self.country_filter = self._normalize_country_filter(country_filter)
        self.row_limit = row_limit
        self._service = None

    @staticmethod
    def _normalize_country_filter(country_filter: str) -> str:
        value = country_filter.strip().strip("'\"").lower()
        if not value:
            return ""
        if value == "pl":
            return "pol"
        return value

    def _country_filter_groups(self) -> list[dict] | None:
        if not self.country_filter:
            return None
        return [
            {
                "groupType": "and",
                "filters": [
                    {
                        "dimension": "country",
                        "operator": "equals",
                        "expression": self.country_filter,
                    }
                ],
            }
        ]

    def _build_service(self):
        if self._service is not None:
            return self._service

        credentials = self._build_credentials()
        http = None
        if AuthorizedHttp is not None:
            http = AuthorizedHttp(
                credentials,
                http=httplib2.Http(timeout=self.HTTP_TIMEOUT_SEC),
            )
        if http is not None:
            self._service = build(
                "searchconsole",
                "v1",
                http=http,
                cache_discovery=False,
            )
        else:
            self._service = build(
                "searchconsole",
                "v1",
                credentials=credentials,
                cache_discovery=False,
            )
        return self._service

    @contextmanager
    def _hard_timeout(self, seconds: int):
        # Unix hard-timeout guard for blocking googleapiclient calls.
        if seconds <= 0 or not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"GSC request exceeded hard timeout ({seconds}s).")

        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    def _build_credentials(self) -> Credentials:
        if self.credentials_path:
            payload = self._load_json(self.credentials_path)
            if payload.get("type") == "service_account":
                return service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=self.SCOPES,
                )
            return self._oauth_credentials_from_payload(payload)

        if self.oauth_client_secret_path:
            payload = self._load_json(self.oauth_client_secret_path)
            return self._oauth_credentials_from_payload(payload)

        raise RuntimeError(
            "Missing GSC credentials. Set GSC_CREDENTIALS_PATH (service account or oauth JSON) "
            "or set GSC_OAUTH_CLIENT_SECRET_PATH + GSC_OAUTH_REFRESH_TOKEN."
        )

    def _oauth_credentials_from_payload(self, payload: dict) -> UserCredentials:
        # OAuth JSON can be either {"installed": {...}} or {"web": {...}}.
        client_section = payload.get("installed") or payload.get("web") or payload
        client_id = client_section.get("client_id")
        client_secret = client_section.get("client_secret")
        token_uri = client_section.get("token_uri") or self.oauth_token_uri

        if not (client_id and client_secret):
            raise RuntimeError(
                "OAuth client JSON is missing client_id/client_secret."
            )
        if not self.oauth_refresh_token:
            raise RuntimeError(
                "Missing GSC_OAUTH_REFRESH_TOKEN for OAuth credentials."
            )

        return UserCredentials(
            token=None,
            refresh_token=self.oauth_refresh_token,
            token_uri=token_uri,
            client_id=client_id,
            client_secret=client_secret,
            scopes=self.SCOPES,
        )

    @staticmethod
    def _load_json(path_value: str) -> dict:
        path = Path(path_value)
        if not path.exists():
            raise RuntimeError(f"GSC credentials file not found: {path_value}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in credentials file: {path_value}") from exc

    def fetch_rows(self, window: DateWindow, dimensions: Sequence[str]) -> list[MetricRow]:
        service = self._build_service()
        body = {
            "startDate": window.start.isoformat(),
            "endDate": window.end.isoformat(),
            "dimensions": list(dimensions),
            "rowLimit": self.row_limit,
        }
        filter_groups = self._country_filter_groups()
        if filter_groups:
            body["dimensionFilterGroups"] = filter_groups

        try:
            with self._hard_timeout(self.REQUEST_HARD_TIMEOUT_SEC):
                response = (
                    service.searchanalytics()
                    .query(siteUrl=self.site_url, body=body)
                    .execute(num_retries=self.API_RETRIES)
                )
        except TimeoutError as exc:
            raise RuntimeError(
                f"GSC API timeout for dimensions {dimensions}: {exc}"
            ) from exc
        except HttpError as exc:
            raise RuntimeError(f"GSC API error for dimensions {dimensions}: {exc}") from exc

        rows: list[MetricRow] = []
        for row in response.get("rows", []):
            keys = row.get("keys", [])
            key = " | ".join(keys) if keys else "TOTAL"
            rows.append(
                MetricRow(
                    key=key,
                    clicks=float(row.get("clicks", 0.0)),
                    impressions=float(row.get("impressions", 0.0)),
                    ctr=float(row.get("ctr", 0.0)),
                    position=float(row.get("position", 0.0)),
                )
            )
        return rows

    def fetch_totals(self, window: DateWindow) -> MetricSummary:
        service = self._build_service()
        body = {
            "startDate": window.start.isoformat(),
            "endDate": window.end.isoformat(),
            "rowLimit": 1,
        }
        filter_groups = self._country_filter_groups()
        if filter_groups:
            body["dimensionFilterGroups"] = filter_groups

        try:
            with self._hard_timeout(self.REQUEST_HARD_TIMEOUT_SEC):
                response = (
                    service.searchanalytics()
                    .query(siteUrl=self.site_url, body=body)
                    .execute(num_retries=self.API_RETRIES)
                )
        except TimeoutError as exc:
            raise RuntimeError(f"GSC API totals timeout: {exc}") from exc
        except HttpError as exc:
            raise RuntimeError(f"GSC API totals error: {exc}") from exc

        row = (response.get("rows") or [{}])[0]
        clicks = float(row.get("clicks", 0.0))
        impressions = float(row.get("impressions", 0.0))
        ctr = float(row.get("ctr", (clicks / impressions if impressions else 0.0)))
        position = float(row.get("position", 0.0))
        return MetricSummary(clicks=clicks, impressions=impressions, ctr=ctr, position=position)

    def fetch_daily_clicks(self, window: DateWindow) -> list[tuple[date, float]]:
        rows = self.fetch_rows(window=window, dimensions=["date"])
        result: list[tuple[date, float]] = []
        for row in rows:
            try:
                day = date.fromisoformat(row.key)
            except ValueError:
                continue
            result.append((day, row.clicks))
        result.sort(key=lambda item: item[0])
        return result
