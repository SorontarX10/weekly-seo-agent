from __future__ import annotations

import base64
import json
from email.message import EmailMessage
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2 import credentials as oauth_credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build


class GmailClient:
    def __init__(
        self,
        service_account_path: str,
        delegated_user: str,
        sender: str,
        auth_mode: str = "service_account",
        oauth_client_secret_path: str = "",
        oauth_refresh_token: str = "",
        oauth_token_uri: str = "https://oauth2.googleapis.com/token",
    ) -> None:
        scopes = ["https://www.googleapis.com/auth/gmail.send"]
        mode = (auth_mode or "service_account").lower()

        if mode == "oauth":
            if not oauth_client_secret_path or not oauth_refresh_token:
                raise ValueError(
                    "OAuth mode requires GMAIL_OAUTH_CLIENT_SECRET_PATH and "
                    "GMAIL_OAUTH_REFRESH_TOKEN."
                )
            secret_path = Path(oauth_client_secret_path)
            payload = json.loads(secret_path.read_text(encoding="utf-8"))
            section = payload.get("installed") or payload.get("web") or {}
            client_id = section.get("client_id")
            client_secret = section.get("client_secret")
            token_uri = section.get("token_uri") or oauth_token_uri
            if not client_id or not client_secret:
                raise ValueError("OAuth client secret JSON missing client_id/client_secret.")

            credentials = oauth_credentials.Credentials(
                token=None,
                refresh_token=oauth_refresh_token,
                token_uri=token_uri,
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
            )
            credentials.refresh(Request())
            self.sender = sender or delegated_user
        else:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=scopes,
            )
            if delegated_user:
                credentials = credentials.with_subject(delegated_user)
            self.sender = sender or delegated_user or credentials.service_account_email

        if not self.sender:
            raise ValueError("Sender email is required for Gmail.")

        self._service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    def send_message(self, to_email: str, subject: str, body: str) -> dict:
        message = EmailMessage()
        message["To"] = to_email
        message["From"] = self.sender
        message["Subject"] = subject
        message.set_content(body)

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        payload = {"raw": raw}
        return self._service.users().messages().send(userId="me", body=payload).execute()
