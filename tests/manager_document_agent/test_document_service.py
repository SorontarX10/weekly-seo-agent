from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weekly_seo_agent.manager_document_agent import DocumentLockedError, DocumentService
from weekly_seo_agent.manager_document_agent.models import DocumentStatus


@pytest.fixture()
def service(tmp_path):
    db_path = tmp_path / "manager_document_agent.db"
    svc = DocumentService(str(db_path))
    try:
        yield svc
    finally:
        svc.close()


def test_create_and_update_document(service: DocumentService):
    doc = service.create_document(
        title="Weekly Test Summary",
        doc_type="TEST_SUMMARY",
        target_audience="Management",
        language="pl",
        objective="Summarize QA test run",
        tone="formal",
        constraints="Do not mention internal codenames",
        current_content="Initial draft",
    )

    assert doc.status == DocumentStatus.IN_PROGRESS
    assert doc.current_content == "Initial draft"

    updated = service.update_document(
        doc.id,
        current_content="Updated draft",
    )
    assert updated.current_content == "Updated draft"

    revisions = service.get_revisions(doc.id)
    assert [rev.revision_no for rev in revisions] == [1, 2]


def test_finalize_locks_document(service: DocumentService):
    doc = service.create_document(
        title="Board update",
        doc_type="MANAGEMENT_BRIEF",
        target_audience="Board",
        language="en",
        objective="Quarterly update",
        tone="concise",
        constraints="Use bullet points",
        current_content="Draft",
    )

    finalized = service.finalize_document(doc.id)
    assert finalized.status == DocumentStatus.FINALIZED
    assert finalized.finalized_at is not None

    with pytest.raises(DocumentLockedError):
        service.update_document(doc.id, current_content="Should fail")


def test_archive_job_archives_finalized_older_than_30_days(service: DocumentService):
    doc = service.create_document(
        title="Release report",
        doc_type="TECH_DOC",
        target_audience="Engineering",
        language="en",
        objective="Release notes",
        tone="neutral",
        constraints="Include risks",
        current_content="Done",
    )
    service.finalize_document(doc.id)

    old_date = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    service._connection.execute(
        "UPDATE documents SET finalized_at = ?, updated_at = ? WHERE id = ?",
        (old_date, old_date, doc.id),
    )
    service._connection.commit()

    archived_count = service.run_archive_job()
    assert archived_count == 1

    archived = service.get_document(doc.id)
    assert archived.status == DocumentStatus.ARCHIVED
    assert archived.archived_at is not None

    archived_docs = service.list_archive()
    assert len(archived_docs) == 1
    assert archived_docs[0].id == doc.id
