from __future__ import annotations

from fastapi.testclient import TestClient

from weekly_seo_agent.manager_document_agent.api import create_app


def _create_payload(title: str, content: str = "Initial content") -> dict:
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


def test_outline_generation_uses_instructions_and_attachments(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Sprint update", ""))
        document_id = created.json()["id"]

        uploaded = client.post(
            f"/documents/{document_id}/attachments",
            files={"file": ("brief.txt", b"Team achieved 92% target.", "text/plain")},
        )
        assert uploaded.status_code == 200

        outlined = client.post(
            f"/documents/{document_id}/outline",
            json={"instructions": "Focus on blockers and next steps"},
        )
        assert outlined.status_code == 200
        content = outlined.json()["current_content"]
        assert "#" in content
        assert len(content) > 200
        lowered = content.lower()
        assert (
            "blocker" in lowered
            or "next step" in lowered
            or "ryzyk" in lowered
            or "blokad" in lowered
            or "kolejne kroki" in lowered
        )
        assert "92%" in content or "team achieved" in lowered


def test_rewrite_full_and_selection_create_ai_revisions(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        created = client.post(
            "/documents",
            json=_create_payload("Review", "Alpha Beta Gamma"),
        )
        document_id = created.json()["id"]

        rewritten = client.post(
            f"/documents/{document_id}/rewrite",
            json={"prompt": "Add formal management framing"},
        )
        assert rewritten.status_code == 200
        full_content = rewritten.json()["current_content"]
        assert full_content != "Alpha Beta Gamma"
        assert len(full_content) > 100
        assert "AI Revision" not in full_content

        latest = rewritten.json()["current_content"]
        start = max(0, len(latest) // 4)
        end = min(len(latest), start + 30)
        partial = client.post(
            f"/documents/{document_id}/rewrite-selection",
            json={
                "selection_start": start,
                "selection_end": end,
                "prompt": "Rewrite with stronger action language",
            },
        )
        assert partial.status_code == 200
        partial_content = partial.json()["current_content"]
        assert partial_content != latest
        assert len(partial_content) > 100

        revisions = client.get(f"/documents/{document_id}/revisions")
        assert revisions.status_code == 200
        change_types = [row["change_type"] for row in revisions.json()]
        assert change_types[-2:] == ["AI_FULL", "AI_PARTIAL"]


def test_rewrite_blocked_after_finalize(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Locked doc"))
        document_id = created.json()["id"]

        finalized = client.post(f"/documents/{document_id}/finalize")
        assert finalized.status_code == 200

        blocked = client.post(
            f"/documents/{document_id}/rewrite",
            json={"prompt": "Change this"},
        )
        assert blocked.status_code == 409


def test_fallback_outline_includes_attachment_facts(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
    )

    with TestClient(app) as client:
        # Force deterministic fallback path.
        app.state.ai_service._llm = None

        created = client.post("/documents", json=_create_payload("SEO & GEO Outlook 2026", ""))
        document_id = created.json()["id"]

        uploaded = client.post(
            f"/documents/{document_id}/attachments",
            files={
                "file": (
                    "deo.txt",
                    (
                        b"In February 2026 AI Organic Traffic delivered 4.5% of Organic GMV.\n"
                        b"2026 Strategic Goal: Protect and grow organic GMV.\n"
                        b"I. Technical & Index Foundations (Protect the Core)\n"
                    ),
                    "text/plain",
                )
            },
        )
        assert uploaded.status_code == 200

        outlined = client.post(
            f"/documents/{document_id}/outline",
            json={"instructions": "Management ready"},
        )
        assert outlined.status_code == 200
        content = outlined.json()["current_content"]
        assert "Source Facts From Attachments" in content
        assert "4.5%" in content
        assert "Technical & Index Foundations" in content
