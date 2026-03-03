from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from weekly_seo_agent.manager_document_agent.api import create_app


def test_full_user_flow_create_edit_finalize_archive(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
        exports_dir=tmp_path / "exports",
        audit_dir=tmp_path / "audit",
    )

    with TestClient(app) as client:
        created = client.post(
            "/documents",
            json={
                "title": "E2E Flow",
                "doc_type": "MANAGEMENT_BRIEF",
                "target_audience": "Management",
                "language": "pl",
                "objective": "Capture full flow",
                "tone": "formal",
                "constraints": "No secrets",
                "current_content": "Initial draft",
            },
        )
        assert created.status_code == 200
        document_id = created.json()["id"]

        uploaded = client.post(
            f"/documents/{document_id}/attachments",
            files={"file": ("brief.txt", b"Attachment context", "text/plain")},
        )
        assert uploaded.status_code == 200

        outlined = client.post(
            f"/documents/{document_id}/outline",
            json={"instructions": "Create manager-ready outline"},
        )
        assert outlined.status_code == 200

        edited = client.patch(
            f"/documents/{document_id}",
            json={"current_content": outlined.json()["current_content"] + "\n\nManual edit."},
        )
        assert edited.status_code == 200

        finalized = client.post(f"/documents/{document_id}/finalize")
        assert finalized.status_code == 200
        assert finalized.json()["status"] == "FINALIZED"

        old_date = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        app.state.document_service._connection.execute(
            "UPDATE documents SET finalized_at = ?, updated_at = ? WHERE id = ?",
            (old_date, old_date, document_id),
        )
        app.state.document_service._connection.commit()

        archived = client.post("/archive/run?older_than_days=30")
        assert archived.status_code == 200
        assert archived.json()["archived_count"] == 1

        archive_list = client.get("/documents/archive")
        assert archive_list.status_code == 200
        ids = [doc["id"] for doc in archive_list.json()]
        assert document_id in ids
