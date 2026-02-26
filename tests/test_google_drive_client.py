from __future__ import annotations

import json
from pathlib import Path

from googleapiclient.errors import HttpError
from httplib2 import Response

from weekly_seo_agent.clients.google_drive_client import GoogleDriveClient


class _FakeRequest:
    def __init__(self, payload: dict):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeErrorRequest:
    def __init__(self, error: Exception):
        self.error = error

    def execute(self):
        raise self.error


class _FakeFiles:
    def __init__(self):
        self.created: list[tuple[dict, object, str | None]] = []
        self.deleted: list[str] = []
        self.queries: list[str] = []
        self.last_doc_name: str = ""
        self.last_doc_parent: str = ""

    def list(self, q=None, **kwargs):
        self.queries.append(q or "")
        if "application/vnd.google-apps.folder" in (q or ""):
            return _FakeRequest({"files": []})
        if "application/vnd.google-apps.document" in (q or ""):
            return _FakeRequest({"files": [{"id": "old-doc-id", "name": "old"}]})
        return _FakeRequest({"files": []})

    def create(self, body=None, media_body=None, fields=None):
        self.created.append((body or {}, media_body, fields))
        if (body or {}).get("mimeType") == "application/vnd.google-apps.folder":
            return _FakeRequest({"id": "folder-123", "name": body.get("name")})
        self.last_doc_name = str((body or {}).get("name", ""))
        parents = (body or {}).get("parents") or []
        if isinstance(parents, list) and parents:
            self.last_doc_parent = str(parents[0])
        return _FakeRequest(
            {
                "id": "doc-999",
                "name": body.get("name"),
                "webViewLink": "https://docs.google.com/document/d/doc-999/edit",
            }
        )

    def delete(self, fileId=None):
        if fileId:
            self.deleted.append(fileId)
        return _FakeRequest({})

    def get(self, fileId=None, fields=None):
        return _FakeRequest(
            {
                "id": str(fileId or ""),
                "name": self.last_doc_name or "doc",
                "parents": [self.last_doc_parent or "folder-123"],
                "trashed": False,
                "createdTime": "2026-02-26T10:00:00Z",
                "modifiedTime": "2026-02-26T10:00:05Z",
                "webViewLink": "https://docs.google.com/document/d/doc-999/edit",
            }
        )


class _FakeDriveService:
    def __init__(self):
        self._files = _FakeFiles()

    def files(self):
        return self._files


class _FakeFilesQuotaExceeded(_FakeFiles):
    def create(self, body=None, media_body=None, fields=None):
        self.created.append((body or {}, media_body, fields))
        if (body or {}).get("mimeType") == "application/vnd.google-apps.folder":
            return _FakeRequest({"id": "folder-123", "name": body.get("name")})
        response = Response({"status": "403", "reason": "Forbidden"})
        payload = b'{"error":{"errors":[{"reason":"storageQuotaExceeded"}]}}'
        return _FakeErrorRequest(HttpError(response, payload, uri="https://www.googleapis.com/upload/drive/v3/files"))


class _FakeDriveServiceQuotaExceeded:
    def __init__(self):
        self._files = _FakeFilesQuotaExceeded()

    def files(self):
        return self._files


def test_escape_query_value() -> None:
    raw = "SEO Weekly Reports O'Hara"
    escaped = GoogleDriveClient._escape_query_value(raw)
    assert escaped == "SEO Weekly Reports O\\'Hara"


def test_upload_docx_creates_google_doc_and_replaces_existing(monkeypatch, tmp_path: Path) -> None:
    local_docx = tmp_path / "2026_02_10_seo_weekly_report.docx"
    local_docx.write_bytes(b"fake-docx")

    client = GoogleDriveClient(
        client_secret_path="secret.json",
        token_path=".google_drive_token.json",
        folder_name="SEO Weekly Reports",
        folder_id="",
    )
    fake_service = _FakeDriveService()
    monkeypatch.setattr(client, "_get_service", lambda: fake_service)

    created = client.upload_docx_as_google_doc(local_docx)

    assert created["id"] == "doc-999"
    assert fake_service._files.deleted == ["old-doc-id"]

    folder_create = next(
        body for body, _, _ in fake_service._files.created if body.get("mimeType") == "application/vnd.google-apps.folder"
    )
    assert folder_create["name"] == "SEO Weekly Reports"

    doc_create = next(
        body for body, _, _ in fake_service._files.created if body.get("mimeType") == "application/vnd.google-apps.document"
    )
    assert doc_create["name"] == local_docx.stem
    assert doc_create["parents"] == ["folder-123"]


def test_upload_docx_uses_explicit_folder_id(monkeypatch, tmp_path: Path) -> None:
    local_docx = tmp_path / "2026_02_17_seo_weekly_report.docx"
    local_docx.write_bytes(b"fake-docx")

    client = GoogleDriveClient(
        client_secret_path="secret.json",
        token_path=".google_drive_token.json",
        folder_name="SEO Weekly Reports",
        folder_id="explicit-folder",
    )
    fake_service = _FakeDriveService()
    monkeypatch.setattr(client, "_get_service", lambda: fake_service)

    client.upload_docx_as_google_doc(local_docx)

    assert all(
        "application/vnd.google-apps.folder" not in query
        for query in fake_service._files.queries
    )

    doc_create = next(
        body for body, _, _ in fake_service._files.created if body.get("mimeType") == "application/vnd.google-apps.document"
    )
    assert doc_create["parents"] == ["explicit-folder"]


def test_service_account_credentials_path(monkeypatch, tmp_path: Path) -> None:
    secret_file = tmp_path / "service_secret.json"
    secret_file.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "proj",
                "private_key_id": "kid",
                "private_key": "-----BEGIN PRIVATE KEY-----\\nabc\\n-----END PRIVATE KEY-----\\n",
                "client_email": "svc@example.iam.gserviceaccount.com",
                "client_id": "123",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        ),
        encoding="utf-8",
    )

    marker = object()
    captured: dict[str, object] = {}

    def fake_from_service_account_file(path: str, scopes: list[str]):
        captured["path"] = path
        captured["scopes"] = scopes
        return marker

    monkeypatch.setattr(
        "weekly_seo_agent.clients.google_drive_client.service_account.Credentials.from_service_account_file",
        fake_from_service_account_file,
    )

    client = GoogleDriveClient(
        client_secret_path=str(secret_file),
        token_path=str(tmp_path / "token.json"),
        folder_name="SEO Weekly Reports",
    )
    creds = client._load_credentials()

    assert creds is marker
    assert captured["path"] == str(secret_file)


def test_upload_docx_quota_error_has_service_account_hint(monkeypatch, tmp_path: Path) -> None:
    local_docx = tmp_path / "2026_02_10_seo_weekly_report.docx"
    local_docx.write_bytes(b"fake-docx")

    client = GoogleDriveClient(
        client_secret_path="secret.json",
        token_path=".google_drive_token.json",
        folder_name="SEO Weekly Reports",
    )
    client._auth_mode = "service_account"
    fake_service = _FakeDriveServiceQuotaExceeded()
    monkeypatch.setattr(client, "_get_service", lambda: fake_service)

    try:
        client.upload_docx_as_google_doc(local_docx)
    except RuntimeError as exc:
        message = str(exc)
        assert "quota exceeded" in message
        assert "service-account based" in message
    else:
        raise AssertionError("Expected RuntimeError for storageQuotaExceeded")
