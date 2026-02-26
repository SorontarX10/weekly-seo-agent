from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account


class MerchantCenterClient:
    """Thin preflight client for Merchant Center (Merchant API)."""

    SCOPES = ["https://www.googleapis.com/auth/content"]
    ACCOUNTS_API_BASE_URL = "https://merchantapi.googleapis.com/accounts/v1"

    def __init__(
        self,
        credentials_path: str,
        mca_id: str = "",
        account_id_map: dict[str, str] | None = None,
    ) -> None:
        self.credentials_path = str(credentials_path or "").strip()
        self.mca_id = str(mca_id or "").strip()
        self.account_id_map = {
            str(country or "").strip().upper(): str(account_id or "").strip()
            for country, account_id in (account_id_map or {}).items()
            if str(country or "").strip() and str(account_id or "").strip()
        }
        self._session: AuthorizedSession | None = None
        self.service_account_email = self._read_service_account_email()

    def _read_service_account_email(self) -> str:
        payload = self._load_json_file()
        if payload.get("type") != "service_account":
            raise RuntimeError(
                "Merchant Center credentials must be a service-account JSON file."
            )
        return str(payload.get("client_email", "")).strip()

    def _load_json_file(self) -> dict:
        if not self.credentials_path:
            raise RuntimeError(
                "Missing Merchant Center credentials path (MERCHANT_CENTER_CREDENTIALS_PATH)."
            )
        path = Path(self.credentials_path)
        if not path.exists():
            raise RuntimeError(
                f"Merchant Center credentials file not found: {self.credentials_path}"
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in Merchant Center credentials file: {self.credentials_path}"
            ) from exc

    def _build_session(self) -> AuthorizedSession:
        if self._session is not None:
            return self._session

        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=self.SCOPES,
        )
        self._session = AuthorizedSession(creds)
        return self._session

    @staticmethod
    def _extract_account_id(account_payload: dict) -> str:
        direct = str(account_payload.get("accountId", "")).strip()
        if direct:
            return direct
        name = str(account_payload.get("name", "")).strip()
        if name.startswith("accounts/"):
            return name.split("/", 1)[1].strip()
        return ""

    def _merchant_api_get(
        self, path: str, params: dict[str, str | int] | None = None
    ) -> dict:
        session = self._build_session()
        url = f"{self.ACCOUNTS_API_BASE_URL}/{path.lstrip('/')}"
        response = session.get(url, params=params or {}, timeout=45)
        if response.status_code >= 400:
            detail = response.text.strip()
            if response.status_code == 401:
                raise RuntimeError(
                    "Merchant API request failed with 401. Confirm that the GCP project "
                    "used by this service account is registered as a Merchant API developer "
                    "for the target Merchant Center account."
                    + (f" Details: {detail}" if detail else "")
                )
            if response.status_code == 403:
                raise RuntimeError(
                    "Merchant API request failed with 403. Confirm that "
                    "`merchantapi.googleapis.com` is enabled in GCP and the "
                    "service account has Merchant Center access."
                    + (f" Details: {detail}" if detail else "")
                )
            raise RuntimeError(
                f"Merchant API request failed ({response.status_code}) for {url}."
                + (f" Details: {detail}" if detail else "")
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Merchant API returned invalid JSON for {url}."
            ) from exc
        if isinstance(payload, dict):
            return payload
        return {}

    def _fetch_accessible_account_ids(self) -> set[str]:
        # Merchant API: list accounts accessible by current principal.
        account_ids: set[str] = set()
        page_token = ""
        while True:
            params: dict[str, str | int] = {"pageSize": 500}
            if page_token:
                params["pageToken"] = page_token
            payload = self._merchant_api_get("accounts", params=params)
            for row in payload.get("accounts", []) or []:
                normalized = self._extract_account_id(row)
                if normalized:
                    account_ids.add(normalized)
            page_token = str(payload.get("nextPageToken", "")).strip()
            if not page_token:
                break
        return account_ids

    def _fetch_subaccount_ids(self, mca_id: str) -> set[str]:
        parent = str(mca_id or "").strip()
        if not parent:
            return set()

        account_ids: set[str] = set()
        page_token = ""
        provider = f"accounts/{parent}"
        # Merchant API: accounts.listSubaccounts
        while True:
            params: dict[str, str | int] = {"pageSize": 500}
            if page_token:
                params["pageToken"] = page_token
            response = self._merchant_api_get(
                f"{provider}:listSubaccounts",
                params=params,
            )
            for row in response.get("accounts", []) or []:
                normalized = self._extract_account_id(row)
                if normalized:
                    account_ids.add(normalized)
            page_token = str(response.get("nextPageToken", "")).strip()
            if not page_token:
                break
        return account_ids

    def validate_country_access(self, required_countries: Iterable[str]) -> dict:
        countries = [
            str(country or "").strip().upper()
            for country in required_countries
            if str(country or "").strip()
        ]
        required = sorted(set(countries))
        if not required:
            return {
                "ok": True,
                "required_countries": [],
                "configured_country_accounts": {},
                "validated_country_accounts": {},
                "missing_country_mappings": [],
                "inaccessible_country_accounts": {},
                "service_account_email": self.service_account_email,
                "accessible_account_ids": [],
                "mca_subaccount_ids": [],
            }

        configured = {
            country: self.account_id_map.get(country, "").strip()
            for country in required
        }
        missing_country_mappings = sorted(
            [country for country, account_id in configured.items() if not account_id]
        )

        accessible_ids = self._fetch_accessible_account_ids()
        subaccount_ids = self._fetch_subaccount_ids(self.mca_id) if self.mca_id else set()
        known_ids = accessible_ids | subaccount_ids

        validated_country_accounts: dict[str, str] = {}
        inaccessible_country_accounts: dict[str, str] = {}
        for country, account_id in configured.items():
            if not account_id:
                continue
            if account_id in known_ids:
                validated_country_accounts[country] = account_id
            else:
                inaccessible_country_accounts[country] = account_id

        ok = not missing_country_mappings and not inaccessible_country_accounts
        return {
            "ok": ok,
            "required_countries": required,
            "configured_country_accounts": configured,
            "validated_country_accounts": validated_country_accounts,
            "missing_country_mappings": missing_country_mappings,
            "inaccessible_country_accounts": inaccessible_country_accounts,
            "service_account_email": self.service_account_email,
            "accessible_account_ids": sorted(accessible_ids),
            "mca_subaccount_ids": sorted(subaccount_ids),
        }
