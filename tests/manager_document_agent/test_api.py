from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from fastapi.testclient import TestClient

import weekly_seo_agent.manager_document_agent.api as api_module
from weekly_seo_agent.manager_document_agent.api import create_app
from weekly_seo_agent.manager_document_agent.models import (
    Attachment,
    ExportStatus,
    ExportType,
    ExtractionStatus,
)


def _create_payload(title: str) -> dict:
    return {
        "title": title,
        "doc_type": "MANAGEMENT_BRIEF",
        "target_audience": "Board",
        "language": "pl",
        "objective": "Summarize key points",
        "tone": "formal",
        "constraints": "No confidential details",
        "current_content": "Initial",
    }


def _build_docx_bytes(paragraphs: list[str]) -> bytes:
    document = DocxDocument()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _seed_oauth_drive_integration(client: TestClient, *, folder_name: str = "Manager Documents", folder_id: str = "") -> None:
    client.app.state.document_service.upsert_integration_setting(
        provider="google_drive",
        config_json=json.dumps(
            {
                "credential_mode": "oauth_quick_connect",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "token_json": '{"token":"fake"}',
                "folder_name": folder_name,
                "folder_id": folder_id,
            }
        ),
    )


def test_documents_crud_and_revisions(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Q1 Summary"))
        assert created.status_code == 200
        created_body = created.json()
        document_id = created_body["id"]
        assert created_body["status"] == "IN_PROGRESS"

        listed = client.get("/documents")
        assert listed.status_code == 200
        assert len(listed.json()) == 1

        fetched = client.get(f"/documents/{document_id}")
        assert fetched.status_code == 200
        assert fetched.json()["title"] == "Q1 Summary"

        updated = client.patch(
            f"/documents/{document_id}",
            json={"current_content": "Updated manually", "change_type": "MANUAL"},
        )
        assert updated.status_code == 200
        assert updated.json()["current_content"] == "Updated manually"

        revisions = client.get(f"/documents/{document_id}/revisions")
        assert revisions.status_code == 200
        revision_numbers = [row["revision_no"] for row in revisions.json()]
        assert revision_numbers == [1, 2]


def test_finalize_blocks_updates_and_last_opened(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        first = client.post("/documents", json=_create_payload("First"))
        second = client.post("/documents", json=_create_payload("Second"))

        first_id = first.json()["id"]
        second_id = second.json()["id"]

        finalized = client.post(f"/documents/{first_id}/finalize")
        assert finalized.status_code == 200
        assert finalized.json()["status"] == "FINALIZED"

        blocked = client.patch(f"/documents/{first_id}", json={"current_content": "Should fail"})
        assert blocked.status_code == 409

        last_opened = client.get("/documents/last-opened")
        assert last_opened.status_code == 200
        assert last_opened.json()["id"] in {first_id, second_id}


def test_archive_document_endpoint_moves_document_to_archive(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Archive now"))
        document_id = created.json()["id"]

        archived = client.post(f"/documents/{document_id}/archive")
        assert archived.status_code == 200
        assert archived.json()["status"] == "ARCHIVED"

        fetched = client.get(f"/documents/{document_id}")
        assert fetched.status_code == 200
        assert fetched.json()["status"] == "ARCHIVED"

        archive = client.get("/documents/archive")
        assert archive.status_code == 200
        assert any(row["id"] == document_id for row in archive.json())


def test_delete_document_endpoint_removes_document_and_attachment_files(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Delete me"))
        document_id = created.json()["id"]

        uploaded = client.post(
            f"/documents/{document_id}/attachments",
            files={"file": ("note.txt", b"Attachment content", "text/plain")},
        )
        assert uploaded.status_code == 200
        stored_path = uploaded.json()["storage_path"]

        deleted = client.delete(f"/documents/{document_id}")
        assert deleted.status_code == 200
        body = deleted.json()
        assert body["deleted"] is True
        assert body["document_id"] == document_id
        assert body["removed_attachment_files"] >= 1

        fetched = client.get(f"/documents/{document_id}")
        assert fetched.status_code == 404

        assert not Path(stored_path).exists()


def test_archive_endpoint_returns_archived_documents(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Archive me"))
        document_id = created.json()["id"]
        finalized = client.post(f"/documents/{document_id}/finalize")
        assert finalized.status_code == 200

        old_date = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        app.state.document_service._connection.execute(
            "UPDATE documents SET finalized_at = ?, updated_at = ? WHERE id = ?",
            (old_date, old_date, document_id),
        )
        app.state.document_service._connection.commit()

        run_archive = client.post("/archive/run?older_than_days=30")
        assert run_archive.status_code == 200
        assert run_archive.json()["archived_count"] == 1

        archive = client.get("/documents/archive")
        assert archive.status_code == 200
        archive_docs = archive.json()
        assert len(archive_docs) == 1
        assert archive_docs[0]["id"] == document_id
        assert archive_docs[0]["status"] == "ARCHIVED"


def test_upload_attachment_and_list(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Attachment doc"))
        document_id = created.json()["id"]

        uploaded = client.post(
            f"/documents/{document_id}/attachments",
            files={"file": ("note.txt", b"Attachment content", "text/plain")},
        )
        assert uploaded.status_code == 200
        body = uploaded.json()
        assert body["document_id"] == document_id
        assert body["extraction_status"] == "OK"
        assert "Attachment content" in body["extracted_text"]

        listed = client.get(f"/documents/{document_id}/attachments")
        assert listed.status_code == 200
        attachments = listed.json()
        assert len(attachments) == 1
        assert attachments[0]["filename"] == "note.txt"


def test_upload_attachment_returns_validation_error(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Attachment doc"))
        document_id = created.json()["id"]

        uploaded = client.post(
            f"/documents/{document_id}/attachments",
            files={"file": ("script.exe", b"binary", "application/octet-stream")},
        )
        assert uploaded.status_code == 400
        assert "Unsupported extension" in uploaded.json()["detail"]


def test_build_attachment_summary_extracts_high_signal_facts():
    attachment = Attachment(
        id=1,
        document_id="doc-1",
        filename="DEO - SEO & GEO Outlook 2026.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        size_bytes=1234,
        storage_path="/tmp/file.docx",
        extraction_status=ExtractionStatus.OK,
        extracted_text=(
            "2026 DEO Outlook\n"
            "In February'26 AI Organic Traffic delivered 4.5% of the whole Organic GMV.\n"
            "2026 Strategic Goal: Protect and grow organic GMV while positioning Allegro as a discovery leader.\n"
            "I. Technical & Index Foundations (Protect the Core)\n"
            "II. Authority & Brand Leadership in AI Ecosystems\n"
            "III. GEO: Generative Engine Optimization (Win Beyond the SERP)\n"
            "Owner\tTimeline\tStatus\n"
            "Adam\tQ2-Q4'26\tIn progress\n"
        ),
        extraction_error="",
        created_at=datetime.now(timezone.utc),
    )
    summary = api_module._build_attachment_summary([attachment])
    assert "4.5%" in summary
    assert "Technical & Index Foundations" in summary
    assert "Q2-Q4'26" in summary
    assert len(summary) < 3500


def test_build_attachment_summary_avoids_dangling_ellipsis_for_long_lines():
    long_line = (
        "In 2026 Allegro scales discovery operations across SEO and GEO with a cross-functional "
        "delivery model focused on measurable GMV impact, structured governance cadence, owner-level "
        "accountability, and consistent KPI reviews across markets and product areas to secure durable growth."
    )
    attachment = Attachment(
        id=2,
        document_id="doc-2",
        filename="long_note.txt",
        mime_type="text/plain",
        size_bytes=2345,
        storage_path="/tmp/long_note.txt",
        extraction_status=ExtractionStatus.OK,
        extracted_text=long_line * 3,
        extraction_error="",
        created_at=datetime.now(timezone.utc),
    )
    summary = api_module._build_attachment_summary([attachment])
    assert "..." not in summary
    assert "cross-functional" in summary.lower()


def test_build_attachment_summary_includes_followup_program_lines_after_colon():
    attachment = Attachment(
        id=3,
        document_id="doc-3",
        filename="strategy.txt",
        mime_type="text/plain",
        size_bytes=3456,
        storage_path="/tmp/strategy.txt",
        extraction_status=ExtractionStatus.OK,
        extracted_text=(
            "This roadmap is built around five strategic programs:\n"
            "I. Technical & Index Foundations (Protect the Core)\n"
            "II. Authority & Brand Leadership in AI Ecosystems\n"
            "III. GEO: Generative Engine Optimization\n"
        ),
        extraction_error="",
        created_at=datetime.now(timezone.utc),
    )
    summary = api_module._build_attachment_summary([attachment])
    assert "five strategic programs" in summary.lower()
    assert "Technical & Index Foundations" in summary
    assert "Authority & Brand Leadership" in summary


def test_import_drive_attachment_requires_integration_settings(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Drive import doc"))
        document_id = created.json()["id"]

        imported = client.post(
            f"/documents/{document_id}/attachments/import-drive",
            json={"file_ref": "https://drive.google.com/file/d/abc123def456/view"},
        )
        assert imported.status_code == 400
        assert "not configured" in imported.json()["detail"].lower()


def test_import_drive_attachment_uses_drive_client_and_persists_attachment(tmp_path, monkeypatch):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    class FakeDriveClient:
        def download_attachment(self, *, file_ref: str) -> dict:
            assert file_ref == "drive-file-1"
            return {
                "filename": "brief.txt",
                "mime_type": "text/plain",
                "raw_bytes": b"Imported from Google Drive",
            }

    monkeypatch.setattr(
        api_module,
        "_build_google_drive_client_from_config",
        lambda *, service, config: FakeDriveClient(),
    )

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)

        created = client.post("/documents", json=_create_payload("Drive import doc"))
        document_id = created.json()["id"]

        imported = client.post(
            f"/documents/{document_id}/attachments/import-drive",
            json={"file_ref": "drive-file-1"},
        )
        assert imported.status_code == 200
        body = imported.json()
        assert body["filename"] == "brief.txt"
        assert body["mime_type"] == "text/plain"
        assert body["extraction_status"] == "OK"
        assert "Imported from Google Drive" in body["extracted_text"]

        listed = client.get(f"/documents/{document_id}/attachments")
        assert listed.status_code == 200
        assert len(listed.json()) == 1


def test_download_current_docx_endpoint_returns_binary_file(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        exports_dir=tmp_path / "exports",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Download me"))
        document_id = created.json()["id"]

        downloaded = client.get(f"/documents/{document_id}/export/docx/download")
        assert downloaded.status_code == 200
        assert (
            downloaded.headers["content-type"]
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(downloaded.content) > 100

        history = client.get(f"/documents/{document_id}/exports")
        assert history.status_code == 200
        assert history.json()[0]["export_type"] == "DOCX"


def test_import_edited_file_updates_document_content(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Import edited"))
        document_id = created.json()["id"]

        edited_docx = _build_docx_bytes(
            ["Executive Summary", "Imported content from edited DOCX"]
        )
        imported = client.post(
            f"/documents/{document_id}/imports/edited-file",
            files={
                "file": (
                    "edited.docx",
                    edited_docx,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert imported.status_code == 200
        body = imported.json()
        assert "Imported content from edited DOCX" in body["current_content"]

        revisions = client.get(f"/documents/{document_id}/revisions")
        assert revisions.status_code == 200
        assert revisions.json()[-1]["prompt"].startswith("import:edited-file:")


def test_sync_edited_document_from_drive_updates_document_content(tmp_path, monkeypatch):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    docx_bytes = _build_docx_bytes(["Synced from Drive", "Updated paragraph"])

    class FakeDriveClient:
        def download_attachment(self, *, file_ref: str) -> dict:
            assert file_ref == "drive-file-123"
            return {
                "filename": "drive_version.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "raw_bytes": docx_bytes,
            }

    monkeypatch.setattr(
        api_module,
        "_build_google_drive_client_from_config",
        lambda *, service, config: FakeDriveClient(),
    )

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)
        created = client.post("/documents", json=_create_payload("Sync edited"))
        document_id = created.json()["id"]

        synced = client.post(
            f"/documents/{document_id}/imports/edited-drive",
            json={"file_ref": "drive-file-123"},
        )
        assert synced.status_code == 200
        body = synced.json()
        assert "Synced from Drive" in body["current_content"]

        revisions = client.get(f"/documents/{document_id}/revisions")
        assert revisions.status_code == 200
        assert revisions.json()[-1]["prompt"].startswith("import:edited-drive:")


def test_google_doc_sync_requires_linked_drive_export(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)
        created = client.post("/documents", json=_create_payload("No linked doc"))
        document_id = created.json()["id"]

        synced = client.post(f"/documents/{document_id}/google-doc/sync")
        assert synced.status_code == 409
        assert "export drive first" in synced.json()["detail"].lower()


def test_google_doc_sync_uses_last_linked_export(tmp_path, monkeypatch):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")
    docx_bytes = _build_docx_bytes(["Linked Google Doc", "Synced content"])

    class FakeDriveClient:
        def download_attachment(self, *, file_ref: str) -> dict:
            assert file_ref == "https://docs.google.com/document/d/linked-doc-1/edit"
            return {
                "filename": "linked.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "raw_bytes": docx_bytes,
            }

    monkeypatch.setattr(
        api_module,
        "_build_google_drive_client_from_config",
        lambda *, service, config: FakeDriveClient(),
    )

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)
        created = client.post("/documents", json=_create_payload("Linked sync"))
        document_id = created.json()["id"]
        client.app.state.document_service.add_export_record(
            document_id=document_id,
            export_type=ExportType.GOOGLE_DRIVE,
            status=ExportStatus.SUCCESS,
            external_url="https://docs.google.com/document/d/linked-doc-1/edit",
        )

        synced = client.post(f"/documents/{document_id}/google-doc/sync")
        assert synced.status_code == 200
        assert "Synced content" in synced.json()["current_content"]


def test_google_doc_update_requires_linked_drive_export(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)
        created = client.post("/documents", json=_create_payload("No linked update"))
        document_id = created.json()["id"]

        updated = client.post(f"/documents/{document_id}/google-doc/update")
        assert updated.status_code == 409
        assert "export drive first" in updated.json()["detail"].lower()


def test_google_doc_update_exports_current_content_to_drive(tmp_path, monkeypatch):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        exports_dir=tmp_path / "exports",
    )
    captured: dict[str, str] = {}

    class FakeDriveClient:
        def upload_docx_as_google_doc(self, local_docx_path: Path, *, document_name: str = "") -> dict:
            assert local_docx_path.exists()
            captured["document_name"] = document_name
            return {"webViewLink": "https://docs.google.com/document/d/updated-doc/edit"}

    monkeypatch.setattr(
        api_module,
        "_build_google_drive_client_from_config",
        lambda *, service, config: FakeDriveClient(),
    )

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)
        created = client.post("/documents", json=_create_payload("Linked update"))
        document_id = created.json()["id"]
        client.app.state.document_service.add_export_record(
            document_id=document_id,
            export_type=ExportType.GOOGLE_DRIVE,
            status=ExportStatus.SUCCESS,
            external_url="https://docs.google.com/document/d/linked-doc-2/edit",
        )

        updated = client.post(f"/documents/{document_id}/google-doc/update")
        assert updated.status_code == 200
        assert updated.json()["status"] == "SUCCESS"
        assert updated.json()["external_url"].startswith("https://docs.google.com/document/d/updated-doc")
        assert captured["document_name"]


def test_create_new_document_from_docx_import(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        docx_bytes = _build_docx_bytes(
            ["## Imported Strategy", "This content comes from uploaded document."]
        )
        created = client.post(
            "/documents/import/docx",
            files={
                "file": (
                    "imported_strategy.docx",
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert created.status_code == 200
        body = created.json()
        assert body["status"] == "IN_PROGRESS"
        assert body["doc_type"] == "DOCUMENTATION"
        assert "This content comes from uploaded document." in body["current_content"]


def test_create_new_document_from_google_doc_import(tmp_path, monkeypatch):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")
    imported_bytes = _build_docx_bytes(["Google Doc Title", "Imported from linked Google Doc"])

    class FakeDriveClient:
        def download_attachment(self, *, file_ref: str) -> dict:
            assert file_ref == "https://docs.google.com/document/d/google-doc-42/edit"
            return {
                "filename": "google_import.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "raw_bytes": imported_bytes,
            }

    monkeypatch.setattr(
        api_module,
        "_build_google_drive_client_from_config",
        lambda *, service, config: FakeDriveClient(),
    )

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)
        created = client.post(
            "/documents/import/google-doc",
            json={"file_ref": "https://docs.google.com/document/d/google-doc-42/edit"},
        )
        assert created.status_code == 200
        body = created.json()
        assert body["status"] == "IN_PROGRESS"
        assert body["doc_type"] == "DOCUMENTATION"
        assert "Imported from linked Google Doc" in body["current_content"]


def test_list_google_drive_files_endpoint(tmp_path, monkeypatch):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    class FakeDriveClient:
        def list_files(self, *, query: str, limit: int, scope: str) -> list[dict]:
            assert query == "outlook"
            assert limit == 30
            assert scope == "all"
            return [
                {
                    "file_id": "abc123",
                    "name": "DEO Outlook 2026.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "modified_time": "2026-03-02T10:00:00Z",
                    "web_view_link": "https://drive.google.com/file/d/abc123/view",
                }
            ]

    monkeypatch.setattr(
        api_module,
        "_build_google_drive_client_from_config",
        lambda *, service, config: FakeDriveClient(),
    )

    with TestClient(app) as client:
        _seed_oauth_drive_integration(client)

        listed = client.get("/integrations/google-drive/files?query=outlook&limit=30&scope=all")
        assert listed.status_code == 200
        body = listed.json()
        assert len(body["files"]) == 1
        assert body["files"][0]["file_id"] == "abc123"
        assert body["files"][0]["name"] == "DEO Outlook 2026.docx"


def test_web_research_endpoint_returns_rows(tmp_path, monkeypatch):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    monkeypatch.setattr(
        api_module,
        "run_web_research",
        lambda **kwargs: {
            "query": kwargs["query"],
            "region": kwargs["region"],
            "provider": "duckduckgo_instant_answer",
            "warning": "",
            "summary_text": "# Web Research: test\n\n## Results\n- sample",
            "items": [
                {
                    "title": "Sample title",
                    "url": "https://example.com/article",
                    "snippet": "Sample snippet",
                    "source": "duckduckgo_related",
                    "page_excerpt": "Fetched page content",
                    "page_error": "",
                }
            ],
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/research/web",
            json={
                "query": "SEO GEO outlook 2026",
                "region": "us-en",
                "max_results": 5,
                "fetch_pages": True,
                "max_pages": 2,
                "page_char_limit": 1200,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "duckduckgo_instant_answer"
        assert body["query"] == "SEO GEO outlook 2026"
        assert len(body["items"]) == 1
        assert body["items"][0]["url"] == "https://example.com/article"


def test_web_research_suggestions_endpoint_returns_queries(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        response = client.post(
            "/research/web/suggestions",
            json={
                "title": "SEO & GEO Outlook 2026",
                "objective": "Prepare management-ready strategy with KPIs and risks.",
                "doc_type": "MANAGEMENT_BRIEF",
                "target_audience": "Management",
                "language": "en",
                "conversation": "User: emphasize AI visibility and CRVisits in 90 day plan.",
                "max_suggestions": 5,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["language"] == "en"
        assert 2 <= len(body["suggestions"]) <= 5
        assert all(isinstance(query, str) and query.strip() for query in body["suggestions"])


def test_attach_web_research_creates_text_attachment(tmp_path, monkeypatch):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    monkeypatch.setattr(
        api_module,
        "run_web_research",
        lambda **kwargs: {
            "query": kwargs["query"],
            "region": kwargs["region"],
            "provider": "duckduckgo_instant_answer",
            "warning": "",
            "summary_text": (
                "# Web Research: test\n\n"
                "## Results\n"
                "- URL: https://example.com\n"
                "- Snippet: Research snippet"
            ),
            "items": [],
        },
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Research attachment doc"))
        document_id = created.json()["id"]

        attached = client.post(
            f"/documents/{document_id}/attachments/research-web",
            json={
                "query": "AI SEO outlook",
                "region": "us-en",
                "max_results": 5,
                "fetch_pages": True,
                "max_pages": 2,
                "page_char_limit": 1200,
            },
        )
        assert attached.status_code == 200
        body = attached.json()
        assert body["filename"].startswith("web_research_")
        assert body["filename"].endswith(".txt")
        assert body["extraction_status"] == "OK"
        assert "Web Research" in body["extracted_text"]

        listed = client.get(f"/documents/{document_id}/attachments")
        assert listed.status_code == 200
        assert len(listed.json()) == 1


def test_ui_root_serves_application_shell(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Manager Document Agent POC" in response.text
        assert "Create New" in response.text
