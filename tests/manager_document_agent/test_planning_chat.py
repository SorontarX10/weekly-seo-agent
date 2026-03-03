from __future__ import annotations

from fastapi.testclient import TestClient

from weekly_seo_agent.manager_document_agent.api import create_app


def test_planning_chat_requires_approval_before_document_create(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        started = client.post(
            "/document-planning/sessions",
            json={"doc_type": "MANAGEMENT_BRIEF", "target_audience": "Management", "language": "pl"},
        )
        assert started.status_code == 200
        session_id = started.json()["id"]

        blocked = client.post(
            f"/document-planning/sessions/{session_id}/create-document",
            json={"include_chat_summary": True},
        )
        assert blocked.status_code == 409
        assert "approved" in blocked.json()["detail"].lower()


def test_planning_chat_end_to_end_creates_document_after_approval(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        started = client.post(
            "/document-planning/sessions",
            json={
                "title": "",
                "doc_type": "MANAGEMENT_BRIEF",
                "target_audience": "Management",
                "language": "pl",
                "objective": "",
                "tone": "formal",
                "constraints": "",
            },
        )
        assert started.status_code == 200
        session = started.json()
        session_id = session["id"]
        assert session["status"] == "COLLECTING"
        assert not session["ready_to_create"]

        messages = [
            "Przygotuj dokument o strategii SEO i GEO na 2026 z naciskiem na decyzje zarządcze.",
            "Podkreśl wpływ na GMV, ryzyka wdrożenia i plan 90 dni.",
            "Potrzebujemy akceptacji priorytetów i budżetu.",
            "Odbiorca: Management, typ: MANAGEMENT_BRIEF, język: en.",
            "Ton formalny, max 2 strony, bez danych poufnych.",
        ]
        for message in messages:
            turned = client.post(
                f"/document-planning/sessions/{session_id}/messages",
                json={"message": message},
            )
            assert turned.status_code == 200
            session = turned.json()

        assert session["status"] == "READY_FOR_APPROVAL"
        assert len(session["suggested_points"]) >= 4
        assert not session["ready_to_create"]

        approved = client.post(
            f"/document-planning/sessions/{session_id}/approve",
            json={"approved_points": []},
        )
        assert approved.status_code == 200
        session = approved.json()
        assert session["status"] == "APPROVED"
        assert session["ready_to_create"]
        assert session["approved_points"]

        created = client.post(
            f"/document-planning/sessions/{session_id}/create-document",
            json={"include_chat_summary": True},
        )
        assert created.status_code == 200
        body = created.json()
        doc = body["document"]
        assert doc["id"]
        assert doc["status"] == "IN_PROGRESS"
        assert doc["language"] == "en"
        assert "Approved Planning Points" in doc["current_content"]
        assert "Planning Chat Summary" in doc["current_content"]

        session_after_create = body["session"]
        assert session_after_create["status"] == "DOCUMENT_CREATED"
        assert session_after_create["created_document_id"] == doc["id"]

        fetched = client.get(f"/documents/{doc['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == doc["id"]

        # Repeated create on same session should return already created document.
        repeated = client.post(
            f"/document-planning/sessions/{session_id}/create-document",
            json={"include_chat_summary": True},
        )
        assert repeated.status_code == 200
        assert repeated.json()["document"]["id"] == doc["id"]
