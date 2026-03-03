from __future__ import annotations

from fastapi.testclient import TestClient

from weekly_seo_agent.manager_document_agent.api import create_app


def test_mock_login_and_rbac_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("MANAGER_DOCUMENT_AGENT_AUTH_MODE", "mock")
    monkeypatch.setenv(
        "MANAGER_DOCUMENT_AGENT_MOCK_TOKEN_ROLES",
        '{"author-token":["author"],"reviewer-token":["reviewer"],"admin-token":["admin"]}',
    )

    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        login = client.post("/auth/login/mock", json={"token": "author-token"})
        assert login.status_code == 200
        assert login.json()["roles"] == ["author"]

        missing_auth = client.get("/documents")
        assert missing_auth.status_code == 401

        reviewer_create = client.post(
            "/documents",
            json={
                "title": "Blocked",
                "doc_type": "MANAGEMENT_BRIEF",
                "target_audience": "Management",
                "language": "pl",
            },
            headers={"Authorization": "Bearer reviewer-token"},
        )
        assert reviewer_create.status_code == 403

        author_create = client.post(
            "/documents",
            json={
                "title": "Allowed",
                "doc_type": "MANAGEMENT_BRIEF",
                "target_audience": "Management",
                "language": "pl",
            },
            headers={"Authorization": "Bearer author-token"},
        )
        assert author_create.status_code == 200

        reviewer_read = client.get(
            "/documents",
            headers={"Authorization": "Bearer reviewer-token"},
        )
        assert reviewer_read.status_code == 200

        metrics_as_author = client.get(
            "/metrics/dashboard",
            headers={"Authorization": "Bearer author-token"},
        )
        assert metrics_as_author.status_code == 403

        metrics_as_admin = client.get(
            "/metrics/dashboard",
            headers={"Authorization": "Bearer admin-token"},
        )
        assert metrics_as_admin.status_code == 200


def test_oauth_login_endpoint_disabled_outside_oauth_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MANAGER_DOCUMENT_AGENT_AUTH_MODE", "none")
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        login = client.post("/auth/login/oauth", json={"id_token": "dummy"})
        assert login.status_code == 400
