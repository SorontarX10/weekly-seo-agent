from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlparse
import webbrowser
from typing import Callable, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


class ManagerGoogleDriveClient:
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    FOLDER_MIME = "application/vnd.google-apps.folder"
    DOC_MIME = "application/vnd.google-apps.document"
    SHEET_MIME = "application/vnd.google-apps.spreadsheet"
    DOCX_MIME = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    FILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")

    MIME_TO_EXTENSION = {
        DOCX_MIME: ".docx",
        XLSX_MIME: ".xlsx",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "application/csv": ".csv",
        "text/tab-separated-values": ".tsv",
    }

    EXTENSION_TO_CANONICAL_MIME = {
        ".docx": DOCX_MIME,
        ".xlsx": XLSX_MIME,
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
    }
    LISTABLE_MIME_TYPES = {
        DOC_MIME,
        SHEET_MIME,
        DOCX_MIME,
        XLSX_MIME,
        "application/pdf",
        "text/plain",
        "text/csv",
        "application/csv",
        "text/tab-separated-values",
    }

    def __init__(
        self,
        *,
        folder_name: str,
        folder_id: str = "",
        client_secret_path: str = "",
        token_path: str = ".google_drive_token.json",
        oauth_client_id: str = "",
        oauth_client_secret: str = "",
        oauth_token_json: str = "",
        token_store_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.folder_name = folder_name.strip() or "Manager Documents"
        self.folder_id = folder_id.strip()

        self.client_secret_path = client_secret_path.strip()
        self.token_path = token_path.strip() or ".google_drive_token.json"

        self.oauth_client_id = oauth_client_id.strip()
        self.oauth_client_secret = oauth_client_secret.strip()
        self.oauth_token_json = oauth_token_json.strip()
        self._token_store_callback = token_store_callback

        self._service = None
        self._auth_mode = "unknown"

    @classmethod
    def from_file_paths(
        cls,
        *,
        client_secret_path: str,
        token_path: str,
        folder_name: str,
        folder_id: str = "",
    ) -> "ManagerGoogleDriveClient":
        return cls(
            folder_name=folder_name,
            folder_id=folder_id,
            client_secret_path=client_secret_path,
            token_path=token_path,
        )

    @classmethod
    def from_oauth_config(
        cls,
        *,
        client_id: str,
        client_secret: str,
        token_json: str,
        folder_name: str,
        folder_id: str = "",
        token_store_callback: Optional[Callable[[str], None]] = None,
    ) -> "ManagerGoogleDriveClient":
        return cls(
            folder_name=folder_name,
            folder_id=folder_id,
            oauth_client_id=client_id,
            oauth_client_secret=client_secret,
            oauth_token_json=token_json,
            token_store_callback=token_store_callback,
        )

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _load_credentials(self) -> Credentials:
        if self.oauth_client_id and self.oauth_client_secret and self.oauth_token_json:
            return self._load_credentials_from_oauth_values()
        return self._load_credentials_from_paths()

    def _load_credentials_from_oauth_values(self) -> Credentials:
        self._auth_mode = "oauth_embedded"
        try:
            token_payload = json.loads(self.oauth_token_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Stored OAuth token payload is invalid JSON") from exc

        creds = Credentials.from_authorized_user_info(token_payload, self.SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            if self._token_store_callback is not None:
                self._token_store_callback(creds.to_json())
        if not creds or not creds.valid:
            raise RuntimeError("Stored OAuth token is invalid or expired")
        return creds

    def _load_credentials_from_paths(self) -> Credentials:
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
            creds = Credentials.from_authorized_user_file(str(token_file), self.SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), self.SCOPES)
            try:
                creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
            except webbrowser.Error as exc:
                raise RuntimeError(
                    "Unable to open browser for OAuth flow. "
                    "Generate token file first or use quick OAuth connect."
                ) from exc

        token_file.write_text(creds.to_json(), encoding="utf-8")
        return creds

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
            raise RuntimeError("Failed to create Google Drive folder")
        return folder_id

    def _delete_existing_docs(self, folder_id: str, doc_name: str) -> int:
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
        deleted_count = 0
        for file_row in files:
            file_id = str(file_row.get("id", "")).strip()
            if not file_id:
                continue
            service.files().delete(fileId=file_id).execute()
            deleted_count += 1
        return deleted_count

    def _verify_uploaded_doc(
        self,
        *,
        file_id: str,
        expected_name: str,
        expected_folder_id: str,
    ) -> dict:
        service = self._get_service()
        metadata = (
            service.files()
            .get(
                fileId=file_id,
                fields="id,name,parents,trashed,createdTime,modifiedTime,webViewLink",
            )
            .execute()
        )
        resolved_name = str(metadata.get("name", "")).strip()
        parents = metadata.get("parents", [])
        parent_ids = [str(item).strip() for item in parents if str(item).strip()]
        is_trashed = bool(metadata.get("trashed", False))

        if is_trashed:
            raise RuntimeError(f"Uploaded Google Doc is trashed: {file_id}")
        if expected_name and resolved_name != expected_name:
            raise RuntimeError(
                "Uploaded Google Doc title mismatch: "
                f"expected '{expected_name}' got '{resolved_name}'"
            )
        if expected_folder_id and expected_folder_id not in parent_ids:
            raise RuntimeError(
                "Uploaded Google Doc folder mismatch: "
                f"expected parent '{expected_folder_id}', got '{parent_ids}'"
            )
        return {
            "exists": True,
            "id": file_id,
            "name": resolved_name,
            "folder_id": expected_folder_id,
            "parents": parent_ids,
            "created_time": str(metadata.get("createdTime", "")).strip(),
            "modified_time": str(metadata.get("modifiedTime", "")).strip(),
            "web_view_link": str(metadata.get("webViewLink", "")).strip(),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_file_id(self, file_ref: str) -> str:
        normalized = file_ref.strip()
        if not normalized:
            raise RuntimeError("Google Drive file reference is empty")
        if self.FILE_ID_RE.fullmatch(normalized):
            return normalized

        parsed = urlparse(normalized)
        if parsed.scheme in {"http", "https"}:
            query = parse_qs(parsed.query)
            query_id = (query.get("id") or [""])[0].strip()
            if self.FILE_ID_RE.fullmatch(query_id):
                return query_id

            for pattern in (
                r"/d/([A-Za-z0-9_-]{10,})",
                r"/folders/([A-Za-z0-9_-]{10,})",
            ):
                match = re.search(pattern, parsed.path)
                if match:
                    return match.group(1)

        raise RuntimeError(
            "Could not parse Google Drive file ID. "
            "Provide file ID or full file URL."
        )

    def _download_request_bytes(self, request) -> bytes:
        buffer = BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()

    def _resolved_filename(
        self,
        *,
        source_name: str,
        resolved_mime_type: str,
        forced_extension: str = "",
    ) -> str:
        base_name = Path(source_name).name.strip() or "drive_attachment"
        extension = Path(base_name).suffix.lower()
        desired_extension = (
            forced_extension.strip().lower()
            or self.MIME_TO_EXTENSION.get(resolved_mime_type, "")
        )
        if not extension and desired_extension:
            return f"{base_name}{desired_extension}"
        return base_name

    def _canonical_mime_for_filename(self, filename: str, fallback: str) -> str:
        extension = Path(filename).suffix.lower()
        return self.EXTENSION_TO_CANONICAL_MIME.get(
            extension,
            fallback.strip() or "application/octet-stream",
        )

    def list_files(
        self,
        *,
        query: str = "",
        limit: int = 25,
        scope: str = "all",
    ) -> list[dict]:
        service = self._get_service()
        safe_limit = min(max(int(limit), 1), 100)
        q_parts: list[str] = ["trashed=false"]
        mime_conditions = sorted(
            [f"mimeType='{mime_type}'" for mime_type in self.LISTABLE_MIME_TYPES]
        )
        q_parts.append("(" + " or ".join(mime_conditions) + ")")

        normalized_scope = scope.strip().lower()
        if normalized_scope == "folder":
            folder_id = self._find_or_create_folder()
            q_parts.append(f"'{folder_id}' in parents")

        name_query = query.strip()
        if name_query:
            escaped_query = self._escape_query_value(name_query)
            q_parts.append(f"name contains '{escaped_query}'")

        response = (
            service.files()
            .list(
                q=" and ".join(q_parts),
                spaces="drive",
                fields="files(id,name,mimeType,modifiedTime,webViewLink)",
                pageSize=safe_limit,
                orderBy="modifiedTime desc",
            )
            .execute()
        )
        rows = response.get("files", [])
        out: list[dict] = []
        for row in rows:
            file_id = str(row.get("id", "")).strip()
            if not file_id:
                continue
            out.append(
                {
                    "file_id": file_id,
                    "name": str(row.get("name", "")).strip(),
                    "mime_type": str(row.get("mimeType", "")).strip(),
                    "modified_time": str(row.get("modifiedTime", "")).strip(),
                    "web_view_link": str(row.get("webViewLink", "")).strip(),
                }
            )
        return out

    def verify_connection(self) -> dict:
        folder_id = self._find_or_create_folder()
        return {
            "ok": True,
            "folder_id": folder_id,
            "folder_name": self.folder_name,
            "auth_mode": self._auth_mode,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

    def upload_docx_as_google_doc(
        self,
        local_docx_path: Path,
        *,
        document_name: str = "",
    ) -> dict:
        if not local_docx_path.exists():
            raise RuntimeError(f"Local DOCX file does not exist: {local_docx_path}")

        folder_id = self._find_or_create_folder()
        doc_name = document_name.strip() or local_docx_path.stem
        service = self._get_service()

        replaced_docs = self._delete_existing_docs(folder_id=folder_id, doc_name=doc_name)

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

        created: dict | None = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
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
                break
            except HttpError as exc:
                retryable = exc.resp.status in {429, 500, 502, 503, 504}
                if retryable and attempt < 3:
                    time.sleep(float(attempt) * 1.5)
                    continue
                last_error = exc
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                break

        if created is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to upload/convert DOCX to Google Docs")

        created_id = str(created.get("id", "")).strip()
        if not created_id:
            raise RuntimeError("Failed to upload/convert DOCX to Google Docs")

        verification = self._verify_uploaded_doc(
            file_id=created_id,
            expected_name=doc_name,
            expected_folder_id=folder_id,
        )
        created["folder_id"] = folder_id
        created["replaced_docs_count"] = int(replaced_docs)
        created["verification"] = verification
        return created

    def download_attachment(self, *, file_ref: str) -> dict:
        file_id = self._extract_file_id(file_ref)
        service = self._get_service()
        metadata = (
            service.files()
            .get(fileId=file_id, fields="id,name,mimeType,trashed,webViewLink")
            .execute()
        )
        is_trashed = bool(metadata.get("trashed", False))
        if is_trashed:
            raise RuntimeError(f"Google Drive file is trashed: {file_id}")

        source_name = str(metadata.get("name", "")).strip() or f"drive_{file_id}"
        source_mime = str(metadata.get("mimeType", "")).strip()
        forced_extension = ""
        resolved_mime = source_mime

        if source_mime == self.DOC_MIME:
            request = service.files().export_media(fileId=file_id, mimeType=self.DOCX_MIME)
            resolved_mime = self.DOCX_MIME
            forced_extension = ".docx"
        elif source_mime == self.SHEET_MIME:
            request = service.files().export_media(fileId=file_id, mimeType=self.XLSX_MIME)
            resolved_mime = self.XLSX_MIME
            forced_extension = ".xlsx"
        elif source_mime.startswith("application/vnd.google-apps."):
            raise RuntimeError(
                f"Google file type '{source_mime}' is not supported for attachment import"
            )
        else:
            request = service.files().get_media(fileId=file_id)

        raw_bytes = self._download_request_bytes(request)
        filename = self._resolved_filename(
            source_name=source_name,
            resolved_mime_type=resolved_mime,
            forced_extension=forced_extension,
        )
        attachment_mime = self._canonical_mime_for_filename(filename, resolved_mime)

        return {
            "file_id": file_id,
            "filename": filename,
            "mime_type": attachment_mime,
            "raw_bytes": raw_bytes,
            "source_mime_type": source_mime,
            "source_name": source_name,
            "web_view_link": str(metadata.get("webViewLink", "")).strip(),
        }
