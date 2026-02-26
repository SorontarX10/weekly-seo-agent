from __future__ import annotations

from pathlib import Path

import pytest

from weekly_seo_agent.weekly_reporting_agent.clients.google_drive_client import (
    GoogleDriveClient,
)


class _FakeRequest:
    def __init__(self, payload: dict):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeFiles:
    def __init__(self, *, mismatch_name: bool = False):
        self.mismatch_name = mismatch_name
        self.deleted: list[str] = []
        self.created: list[dict] = []

    def list(self, q=None, **kwargs):
        query = str(q or "")
        if "application/vnd.google-apps.folder" in query:
            return _FakeRequest({"files": []})
        if "application/vnd.google-apps.document" in query:
            return _FakeRequest({"files": [{"id": "old-doc-1", "name": "old"}]})
        return _FakeRequest({"files": []})

    def create(self, body=None, media_body=None, fields=None):
        payload = body or {}
        self.created.append(payload)
        if payload.get("mimeType") == "application/vnd.google-apps.folder":
            return _FakeRequest({"id": "folder-abc", "name": payload.get("name", "")})
        return _FakeRequest(
            {
                "id": "doc-123",
                "name": payload.get("name", ""),
                "webViewLink": "https://docs.google.com/document/d/doc-123/edit",
            }
        )

    def delete(self, fileId=None):
        if fileId:
            self.deleted.append(str(fileId))
        return _FakeRequest({})

    def get(self, fileId=None, fields=None):
        name = "unexpected-name" if self.mismatch_name else "2026_02_26_pl_seo_weekly_report"
        return _FakeRequest(
            {
                "id": str(fileId or ""),
                "name": name,
                "parents": ["folder-abc"],
                "trashed": False,
                "createdTime": "2026-02-26T10:00:00Z",
                "modifiedTime": "2026-02-26T10:00:05Z",
                "webViewLink": "https://docs.google.com/document/d/doc-123/edit",
            }
        )


class _FakeDriveService:
    def __init__(self, *, mismatch_name: bool = False):
        self._files = _FakeFiles(mismatch_name=mismatch_name)

    def files(self):
        return self._files


def test_upload_docx_returns_verification_payload(monkeypatch, tmp_path: Path) -> None:
    local_docx = tmp_path / "2026_02_26_pl_seo_weekly_report.docx"
    local_docx.write_bytes(b"fake-docx")

    client = GoogleDriveClient(
        client_secret_path="secret.json",
        token_path=".google_drive_token.json",
        folder_name="SEO Weekly Reports",
    )
    fake_service = _FakeDriveService()
    monkeypatch.setattr(client, "_get_service", lambda: fake_service)

    uploaded = client.upload_docx_as_google_doc(local_docx)
    verification = uploaded.get("verification", {})

    assert uploaded["id"] == "doc-123"
    assert int(uploaded.get("replaced_docs_count", 0)) == 1
    assert verification.get("exists") is True
    assert verification.get("name") == local_docx.stem
    assert verification.get("folder_id") == "folder-abc"
    assert bool(verification.get("verified_at"))


def test_upload_docx_raises_on_verification_name_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    local_docx = tmp_path / "2026_02_26_pl_seo_weekly_report.docx"
    local_docx.write_bytes(b"fake-docx")

    client = GoogleDriveClient(
        client_secret_path="secret.json",
        token_path=".google_drive_token.json",
        folder_name="SEO Weekly Reports",
    )
    fake_service = _FakeDriveService(mismatch_name=True)
    monkeypatch.setattr(client, "_get_service", lambda: fake_service)

    with pytest.raises(RuntimeError):
        client.upload_docx_as_google_doc(local_docx)
