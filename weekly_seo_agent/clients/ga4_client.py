from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

from weekly_seo_agent.models import DateWindow


@dataclass(frozen=True)
class GA4MetricSummary:
    sessions: float | None
    users: float | None
    engaged_sessions: float | None
    transactions: float | None
    revenue: float | None


class GA4Client:
    """Small GA4 Data API client (REST) for weekly SEO reporting context."""

    SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]
    API_BASE = "https://analyticsdata.googleapis.com/v1beta"

    def __init__(
        self,
        property_id: str,
        credentials_path: str,
        country_code: str = "PL",
    ) -> None:
        self.property_id = self._normalize_property_id(property_id)
        self.credentials_path = credentials_path.strip()
        self.country_code = country_code.strip().upper()
        self._session: AuthorizedSession | None = None

    @staticmethod
    def _normalize_property_id(raw: str) -> str:
        value = raw.strip()
        if value.startswith("properties/"):
            value = value.split("/", 1)[1]
        return value

    def _build_session(self) -> AuthorizedSession:
        if self._session is not None:
            return self._session

        if not self.credentials_path:
            raise RuntimeError("GA4 credentials path is missing.")
        path = Path(self.credentials_path)
        if not path.exists():
            raise RuntimeError(f"GA4 credentials file not found: {self.credentials_path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in GA4 credentials file: {self.credentials_path}") from exc

        if payload.get("type") != "service_account":
            raise RuntimeError(
                "GA4 client currently expects service-account credentials "
                "(JSON with type=service_account)."
            )

        creds = service_account.Credentials.from_service_account_file(
            str(path),
            scopes=self.SCOPES,
        )
        self._session = AuthorizedSession(creds)
        return self._session

    def _run_report(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.property_id:
            raise RuntimeError("GA4 property ID is missing.")
        session = self._build_session()
        url = f"{self.API_BASE}/properties/{self.property_id}:runReport"
        response = session.post(url, json=body, timeout=40)
        if not response.ok:
            detail = response.text.strip()
            if len(detail) > 400:
                detail = detail[:397] + "..."
            raise RuntimeError(
                f"GA4 Data API request failed ({response.status_code}): {detail or 'No response body.'}"
            )
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("GA4 Data API returned non-object payload.")

    def _base_country_filter(self) -> dict[str, Any]:
        return {
            "filter": {
                "fieldName": "countryId",
                "stringFilter": {"matchType": "EXACT", "value": self.country_code},
            }
        }

    @staticmethod
    def _metric_value(row: dict[str, Any], index: int) -> float | None:
        values = row.get("metricValues", [])
        if not isinstance(values, list) or index >= len(values):
            return None
        raw = values[index]
        if not isinstance(raw, dict):
            return None
        value = raw.get("value")
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def fetch_summary(self, window: DateWindow) -> GA4MetricSummary:
        body = {
            "dateRanges": [
                {
                    "startDate": window.start.isoformat(),
                    "endDate": window.end.isoformat(),
                }
            ],
            "metrics": [
                {"name": "sessions"},
                {"name": "totalUsers"},
                {"name": "engagedSessions"},
                {"name": "transactions"},
                {"name": "purchaseRevenue"},
            ],
            "dimensionFilter": self._base_country_filter(),
            "limit": 1,
        }
        payload = self._run_report(body)
        rows = payload.get("rows", [])
        if not isinstance(rows, list) or not rows:
            return GA4MetricSummary(
                sessions=None,
                users=None,
                engaged_sessions=None,
                transactions=None,
                revenue=None,
            )
        row = rows[0] if isinstance(rows[0], dict) else {}
        return GA4MetricSummary(
            sessions=self._metric_value(row, 0),
            users=self._metric_value(row, 1),
            engaged_sessions=self._metric_value(row, 2),
            transactions=self._metric_value(row, 3),
            revenue=self._metric_value(row, 4),
        )

    def fetch_channel_performance(
        self,
        window: DateWindow,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        body = {
            "dateRanges": [
                {
                    "startDate": window.start.isoformat(),
                    "endDate": window.end.isoformat(),
                }
            ],
            "dimensions": [{"name": "sessionDefaultChannelGroup"}],
            "metrics": [
                {"name": "sessions"},
                {"name": "totalUsers"},
                {"name": "transactions"},
                {"name": "purchaseRevenue"},
            ],
            "dimensionFilter": self._base_country_filter(),
            "orderBys": [{"metric": {"metricName": "sessions"}, "desc": True}],
            "limit": max(1, int(limit)),
        }
        payload = self._run_report(body)
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            dims = row.get("dimensionValues", [])
            channel = ""
            if isinstance(dims, list) and dims and isinstance(dims[0], dict):
                channel = str(dims[0].get("value", "")).strip()
            if not channel:
                continue
            out.append(
                {
                    "channel": channel,
                    "sessions": self._metric_value(row, 0),
                    "users": self._metric_value(row, 1),
                    "transactions": self._metric_value(row, 2),
                    "revenue": self._metric_value(row, 3),
                }
            )
        return out

    def fetch_top_landing_pages(
        self,
        window: DateWindow,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        body = {
            "dateRanges": [
                {
                    "startDate": window.start.isoformat(),
                    "endDate": window.end.isoformat(),
                }
            ],
            "dimensions": [{"name": "landingPagePlusQueryString"}],
            "metrics": [
                {"name": "sessions"},
                {"name": "transactions"},
                {"name": "purchaseRevenue"},
            ],
            "dimensionFilter": self._base_country_filter(),
            "orderBys": [{"metric": {"metricName": "sessions"}, "desc": True}],
            "limit": max(1, int(limit)),
        }
        payload = self._run_report(body)
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            dims = row.get("dimensionValues", [])
            landing_page = ""
            if isinstance(dims, list) and dims and isinstance(dims[0], dict):
                landing_page = str(dims[0].get("value", "")).strip()
            if not landing_page:
                continue
            out.append(
                {
                    "landing_page": landing_page,
                    "sessions": self._metric_value(row, 0),
                    "transactions": self._metric_value(row, 1),
                    "revenue": self._metric_value(row, 2),
                }
            )
        return out
