from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from weekly_seo_agent.manager_document_agent.api import create_app


def _create_payload(title: str, content: str = "Initial") -> dict:
    return {
        "title": title,
        "doc_type": "MANAGEMENT_BRIEF",
        "target_audience": "Management",
        "language": "pl",
        "objective": "Summarize progress",
        "tone": "formal",
        "constraints": "Keep it concise",
        "current_content": content,
    }


def test_audit_export_generates_jsonl(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        exports_dir=tmp_path / "exports",
        audit_dir=tmp_path / "audit",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Audit doc"))
        document_id = created.json()["id"]

        client.patch(
            f"/documents/{document_id}",
            json={"current_content": "Updated", "change_type": "MANUAL"},
        )
        client.post(f"/documents/{document_id}/export/docx")

        exported = client.post(f"/audit/export?document_id={document_id}")
        assert exported.status_code == 200
        body = exported.json()
        assert body["event_count"] > 0
        assert Path(body["file_path"]).exists()


def test_audit_export_returns_404_for_unknown_document(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        exported = client.post("/audit/export?document_id=missing-doc")
        assert exported.status_code == 404
