from __future__ import annotations

import json
from pathlib import Path
import webbrowser

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class GoogleDriveClient:
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    FOLDER_MIME = "application/vnd.google-apps.folder"
    DOC_MIME = "application/vnd.google-apps.document"
    DOCX_MIME = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    def __init__(
        self,
        client_secret_path: str,
        token_path: str,
        folder_name: str,
        folder_id: str = "",
    ) -> None:
        self.client_secret_path = client_secret_path.strip()
        self.token_path = token_path.strip() or ".google_drive_token.json"
        self.folder_name = folder_name.strip() or "SEO Weekly Reports"
        self.folder_id = folder_id.strip()
        self._service = None
        self._auth_mode = "unknown"

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _load_credentials(self) -> Credentials:
        secret_path = Path(self.client_secret_path)
        if not secret_path.exists():
            raise RuntimeError(
                "Google Drive credentials file not found: "
                f"{self.client_secret_path}"
            )

        secret_payload = json.loads(secret_path.read_text(encoding="utf-8"))
        if secret_payload.get("type") == "service_account":
            self._auth_mode = "service_account"
            return service_account.Credentials.from_service_account_file(
                str(secret_path),
                scopes=self.SCOPES,
            )

        self._auth_mode = "oauth_client"
        creds: Credentials | None = None
        token_file = Path(self.token_path)
        if token_file.exists():
            creds = Credentials.from_authorized_user_file(
                str(token_file), self.SCOPES
            )

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            # GitHub Actions / CI is non-interactive. OAuth browser flow cannot run there.
            if self._running_in_ci():
                raise RuntimeError(
                    "Google Drive OAuth requires a pre-generated token in CI. "
                    "Provide GOOGLE_DRIVE_TOKEN_JSON (or use service account JSON in "
                    "GOOGLE_DRIVE_CLIENT_SECRET_PATH with folder access)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(secret_path), self.SCOPES
            )
            try:
                creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
            except webbrowser.Error as exc:
                raise RuntimeError(
                    "Unable to start browser for Google OAuth flow. "
                    "Run locally to create token file first (GOOGLE_DRIVE_TOKEN_PATH), "
                    "then provide it in CI as GOOGLE_DRIVE_TOKEN_JSON."
                ) from exc

        token_file.write_text(creds.to_json(), encoding="utf-8")
        return creds

    @staticmethod
    def _running_in_ci() -> bool:
        import os
        value = str(os.environ.get("CI", "")).strip().lower()
        return value in {"1", "true", "yes"}

    def _get_service(self):
        if self._service is None:
            credentials = self._load_credentials()
            self._service = build(
                "drive",
                "v3",
                credentials=credentials,
                cache_discovery=False,
            )
        return self._service

    def _find_or_create_folder(self) -> str:
        if self.folder_id:
            return self.folder_id

        service = self._get_service()
        escaped = self._escape_query_value(self.folder_name)
        query = (
            f"mimeType='{self.FOLDER_MIME}' "
            f"and name='{escaped}' "
            "and trashed=false and 'root' in parents"
        )

        response = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id,name)", pageSize=1)
            .execute()
        )
        files = response.get("files", [])
        if files:
            return files[0]["id"]

        created = (
            service.files()
            .create(
                body={"name": self.folder_name, "mimeType": self.FOLDER_MIME},
                fields="id,name",
            )
            .execute()
        )
        folder_id = str(created.get("id", "")).strip()
        if not folder_id:
            raise RuntimeError("Failed to create Google Drive folder.")
        return folder_id

    def _delete_existing_docs(self, folder_id: str, doc_name: str) -> None:
        service = self._get_service()
        escaped = self._escape_query_value(doc_name)
        query = (
            f"mimeType='{self.DOC_MIME}' and name='{escaped}' "
            f"and '{folder_id}' in parents and trashed=false"
        )
        response = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id,name)", pageSize=100)
            .execute()
        )
        files = response.get("files", [])
        for file_row in files:
            file_id = str(file_row.get("id", "")).strip()
            if not file_id:
                continue
            service.files().delete(fileId=file_id).execute()

    def upload_docx_as_google_doc(self, local_docx_path: Path) -> dict:
        if not local_docx_path.exists():
            raise RuntimeError(f"Local DOCX file does not exist: {local_docx_path}")

        folder_id = self._find_or_create_folder()
        doc_name = local_docx_path.stem
        service = self._get_service()

        self._delete_existing_docs(folder_id=folder_id, doc_name=doc_name)

        media = MediaFileUpload(
            str(local_docx_path),
            mimetype=self.DOCX_MIME,
            resumable=False,
        )
        body = {
            "name": doc_name,
            "mimeType": self.DOC_MIME,
            "parents": [folder_id],
        }

        try:
            created = (
                service.files()
                .create(
                    body=body,
                    media_body=media,
                    fields="id,name,webViewLink,mimeType,parents",
                )
                .execute()
            )
        except HttpError as exc:
            payload = str(exc)
            if getattr(exc, "content", None):
                try:
                    payload = f"{payload} {exc.content.decode('utf-8', errors='ignore')}"
                except Exception:
                    pass
            if exc.resp.status == 403 and "storageQuotaExceeded" in payload:
                hint = (
                    "Google Drive quota exceeded for the authenticated account. "
                )
                if self._auth_mode == "service_account":
                    hint += (
                        "Current credentials are service-account based. "
                        "Use OAuth client credentials for a user Drive or point "
                        "GOOGLE_DRIVE_FOLDER_ID to a Shared Drive folder where this "
                        "service account has write access."
                    )
                raise RuntimeError(hint) from exc
            raise

        if not created.get("id"):
            raise RuntimeError("Failed to upload/convert DOCX to Google Docs.")
        return created
