from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from weekly_seo_agent.manager_document_agent.models import ExportStatus, ExportType
from weekly_seo_agent.manager_document_agent.service import DocumentService


def test_document_service_handles_parallel_read_calls(tmp_path):
    service = DocumentService(str(tmp_path / "manager_agent_api.db"))
    try:
        document = service.create_document(
            title="Parallel reads",
            doc_type="MANAGEMENT_BRIEF",
            target_audience="Management",
            language="pl",
            objective="Thread-safe reads",
            tone="formal",
            constraints="",
            current_content="# Draft\n\nContent",
        )
        service.add_export_record(
            document_id=document.id,
            export_type=ExportType.DOCX,
            status=ExportStatus.SUCCESS,
            file_path="/tmp/file.docx",
        )

        def _read_bundle() -> tuple[int, int, int, str]:
            exports = service.list_export_records(document.id)
            revisions = service.get_revisions(document.id)
            attachments = service.list_attachments(document.id)
            fetched = service.get_document(document.id)
            return (len(exports), len(revisions), len(attachments), fetched.id)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: _read_bundle(), range(40)))

        for exports_count, revisions_count, attachments_count, fetched_id in results:
            assert exports_count >= 1
            assert revisions_count >= 1
            assert attachments_count >= 0
            assert fetched_id == document.id
    finally:
        service.close()
