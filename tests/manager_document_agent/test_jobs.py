from __future__ import annotations

import time

from fastapi.testclient import TestClient

from weekly_seo_agent.manager_document_agent.api import create_app


def _create_payload(title: str, content: str = "Alpha Beta Gamma") -> dict:
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


def _wait_for_job(client: TestClient, job_id: int, timeout_sec: float = 20.0) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"SUCCESS", "FAILED"}:
            return body
        time.sleep(0.1)
    raise AssertionError(f"Job {job_id} did not finish in {timeout_sec}s")


def test_async_attachment_parsing_job_flow(tmp_path):
    app = create_app(
        db_path=tmp_path / "manager_agent_api.db",
        attachments_dir=tmp_path / "attachments",
        exports_dir=tmp_path / "exports",
    )

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Async parse"))
        document_id = created.json()["id"]

        queued = client.post(
            f"/documents/{document_id}/attachments/async",
            files={"file": ("note.txt", b"Queued parsing payload", "text/plain")},
        )
        assert queued.status_code == 200
        queued_body = queued.json()
        assert queued_body["job"]["job_type"] == "parse_attachment"
        assert queued_body["related_attachment_id"] is not None

        final_job = _wait_for_job(client, queued_body["job"]["id"])
        assert final_job["status"] == "SUCCESS"

        attachments = client.get(f"/documents/{document_id}/attachments")
        assert attachments.status_code == 200
        assert attachments.json()[0]["extraction_status"] in {"OK", "PARTIAL"}


def test_async_rewrite_and_metrics_dashboard(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        created = client.post("/documents", json=_create_payload("Async rewrite"))
        document_id = created.json()["id"]

        queued = client.post(
            f"/documents/{document_id}/rewrite/async",
            json={"prompt": "Add stronger management framing"},
            headers={"X-Request-ID": "req-123"},
        )
        assert queued.status_code == 200
        job_id = queued.json()["job"]["id"]
        assert queued.json()["job"]["request_id"] == "req-123"

        final_job = _wait_for_job(client, job_id)
        assert final_job["status"] == "SUCCESS"

        metrics = client.get("/metrics/dashboard")
        assert metrics.status_code == 200
        body = metrics.json()
        assert "requests" in body
        assert "jobs" in body
        assert body["jobs"]["total_jobs"] >= 1


def test_enqueue_nightly_archive_job(tmp_path):
    app = create_app(db_path=tmp_path / "manager_agent_api.db")

    with TestClient(app) as client:
        queued = client.post("/jobs/archive-nightly?older_than_days=30")
        assert queued.status_code == 200
        job = queued.json()["job"]
        assert job["job_type"] == "archive_scan"

        final_job = _wait_for_job(client, job["id"])
        assert final_job["status"] == "SUCCESS"
