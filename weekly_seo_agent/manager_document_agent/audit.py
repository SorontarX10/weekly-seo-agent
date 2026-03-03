from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

from .service import DocumentService


def export_audit_log_jsonl(
    *,
    service: DocumentService,
    output_dir: Path,
    document_id: Optional[str] = None,
) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    scope = document_id or "all"
    output_path = output_dir / f"audit_{scope}_{timestamp}.jsonl"

    events: list[dict] = []
    documents = [service.get_document(document_id)] if document_id else service.list_documents()

    for document in documents:
        revisions = service.get_revisions(document.id)
        attachments = service.list_attachments(document.id)
        exports = service.list_export_records(document.id)

        for revision in revisions:
            events.append(
                {
                    "event_type": "revision",
                    "document_id": document.id,
                    "timestamp": revision.created_at.isoformat(),
                    "payload": {
                        "revision_no": revision.revision_no,
                        "change_type": revision.change_type.value,
                        "prompt": revision.prompt,
                    },
                }
            )

        for attachment in attachments:
            events.append(
                {
                    "event_type": "attachment",
                    "document_id": document.id,
                    "timestamp": attachment.created_at.isoformat(),
                    "payload": {
                        "filename": attachment.filename,
                        "extraction_status": attachment.extraction_status.value,
                        "extraction_error": attachment.extraction_error,
                    },
                }
            )

        for export in exports:
            events.append(
                {
                    "event_type": "export",
                    "document_id": document.id,
                    "timestamp": export.created_at.isoformat(),
                    "payload": {
                        "export_type": export.export_type.value,
                        "status": export.status.value,
                        "external_url": export.external_url,
                        "error_message": export.error_message,
                    },
                }
            )

    jobs = service.list_jobs(limit=1000)
    for job in jobs:
        payload = json.loads(job.payload_json) if job.payload_json else {}
        payload_document_id = str(payload.get("document_id", ""))
        if document_id and payload_document_id != document_id:
            continue
        events.append(
            {
                "event_type": "job",
                "document_id": payload_document_id,
                "timestamp": job.updated_at.isoformat(),
                "payload": {
                    "job_id": job.id,
                    "job_type": job.job_type,
                    "status": job.status.value,
                    "attempts": job.attempts,
                    "failure_class": job.failure_class,
                    "error_message": job.error_message,
                    "request_id": job.request_id,
                },
            }
        )

    events.sort(key=lambda item: item["timestamp"])
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    return output_path, len(events)
