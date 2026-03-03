from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2 import id_token as google_id_token


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class AuthUser:
    subject: str
    email: str
    roles: list[str]
    auth_mode: str


class AuthManager:
    def __init__(
        self,
        *,
        mode: str,
        mock_token_roles: dict[str, list[str]],
        oauth_client_id: str,
        oauth_role_map: dict[str, list[str]],
    ):
        self.mode = mode
        self.mock_token_roles = mock_token_roles
        self.oauth_client_id = oauth_client_id
        self.oauth_role_map = oauth_role_map

    @classmethod
    def from_env(cls) -> "AuthManager":
        mode = os.getenv("MANAGER_DOCUMENT_AGENT_AUTH_MODE", "none").strip().lower()
        if mode not in {"none", "mock", "oauth"}:
            mode = "none"

        mock_token_roles = _parse_roles_map(
            os.getenv("MANAGER_DOCUMENT_AGENT_MOCK_TOKEN_ROLES", "{}")
        )
        oauth_role_map = _parse_roles_map(
            os.getenv("MANAGER_DOCUMENT_AGENT_OAUTH_ROLE_MAP", "{}")
        )
        oauth_client_id = os.getenv("MANAGER_DOCUMENT_AGENT_OAUTH_CLIENT_ID", "").strip()

        return cls(
            mode=mode,
            mock_token_roles=mock_token_roles,
            oauth_client_id=oauth_client_id,
            oauth_role_map=oauth_role_map,
        )

    def authenticate(self, authorization_header: Optional[str]) -> AuthUser:
        if self.mode == "none":
            return AuthUser(
                subject="anonymous",
                email="",
                roles=["admin", "author", "reviewer"],
                auth_mode="none",
            )

        token = _extract_bearer_token(authorization_header)
        if self.mode == "mock":
            roles = self.mock_token_roles.get(token)
            if roles is None:
                raise AuthError("Invalid mock token", status_code=401)
            return AuthUser(
                subject=f"mock:{token[:8]}",
                email="mock@example.local",
                roles=roles,
                auth_mode="mock",
            )

        if self.mode == "oauth":
            if not self.oauth_client_id:
                raise AuthError(
                    "OAuth mode enabled but MANAGER_DOCUMENT_AGENT_OAUTH_CLIENT_ID is missing",
                    status_code=500,
                )
            try:
                claims = google_id_token.verify_oauth2_token(
                    token,
                    Request(),
                    self.oauth_client_id,
                )
            except Exception as exc:
                raise AuthError(f"Invalid OAuth token: {exc}", status_code=401) from exc

            email = str(claims.get("email", "")).strip().lower()
            subject = str(claims.get("sub", "")).strip()
            if not subject:
                raise AuthError("OAuth token is missing subject", status_code=401)
            roles = self.oauth_role_map.get(email, ["author"])
            return AuthUser(
                subject=subject,
                email=email,
                roles=roles,
                auth_mode="oauth",
            )

        raise AuthError("Unsupported auth mode", status_code=500)

    def login_mock(self, token: str) -> AuthUser:
        roles = self.mock_token_roles.get(token)
        if roles is None:
            raise AuthError("Invalid mock token", status_code=401)
        return AuthUser(
            subject=f"mock:{token[:8]}",
            email="mock@example.local",
            roles=roles,
            auth_mode="mock",
        )

    def login_oauth(self, id_token: str) -> AuthUser:
        return self.authenticate(f"Bearer {id_token}")



def has_any_role(user: AuthUser, required_roles: list[str]) -> bool:
    return bool(set(user.roles).intersection(required_roles))



def _parse_roles_map(raw: str) -> dict[str, list[str]]:
    text = raw.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            roles = [str(item).strip() for item in value if str(item).strip()]
        else:
            roles = [str(value).strip()] if str(value).strip() else []
        normalized[str(key)] = roles
    return normalized



def _extract_bearer_token(authorization_header: Optional[str]) -> str:
    if not authorization_header:
        raise AuthError("Missing Authorization header", status_code=401)
    value = authorization_header.strip()
    if not value.lower().startswith("bearer "):
        raise AuthError("Authorization header must use Bearer token", status_code=401)
    token = value[7:].strip()
    if not token:
        raise AuthError("Bearer token is empty", status_code=401)
    return token
