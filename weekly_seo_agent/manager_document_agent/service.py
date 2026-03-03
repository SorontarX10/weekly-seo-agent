from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

from .db import connect, init_db
from .models import (
    Attachment,
    Document,
    DocumentRevision,
    DocumentStatus,
    ExportRecord,
    ExportStatus,
    ExportType,
    ExtractionStatus,
    IntegrationSetting,
    JobFailureClass,
    JobRecord,
    JobStatus,
    RevisionChangeType,
)


class NotFoundError(Exception):
    """Raised when the requested document does not exist."""


class DocumentLockedError(Exception):
    """Raised when trying to edit non-editable document."""


def synchronized_db_method(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class DocumentService:
    def __init__(self, db_path: str):
        self._connection = connect(db_path)
        init_db(self._connection)
        self._lock = threading.RLock()

    @synchronized_db_method
    def close(self) -> None:
        self._connection.close()

    @synchronized_db_method
    def create_document(
        self,
        *,
        title: str,
        doc_type: str,
        target_audience: str,
        language: str,
        objective: str,
        tone: str,
        constraints: str,
        current_content: str = "",
    ) -> Document:
        now = self._now_iso()
        document_id = str(uuid.uuid4())
        self._connection.execute(
            """
            INSERT INTO documents (
                id, title, doc_type, target_audience, language,
                objective, tone, constraints, status, current_content,
                last_opened_at, finalized_at, archived_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                title,
                doc_type,
                target_audience,
                language,
                objective,
                tone,
                constraints,
                DocumentStatus.IN_PROGRESS.value,
                current_content,
                now,
                None,
                None,
                now,
                now,
            ),
        )
        self._add_revision(
            document_id=document_id,
            change_type=RevisionChangeType.SYSTEM,
            prompt=None,
            content_snapshot=current_content,
            created_at=now,
        )
        self._connection.commit()
        return self.get_document(document_id)

    @synchronized_db_method
    def list_documents(self, status: Optional[DocumentStatus] = None) -> list[Document]:
        query = "SELECT * FROM documents"
        params: tuple[str, ...] = tuple()
        if status:
            query += " WHERE status = ?"
            params = (status.value,)
        query += " ORDER BY updated_at DESC"
        rows = self._connection.execute(query, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    @synchronized_db_method
    def list_archive(self) -> list[Document]:
        rows = self._connection.execute(
            "SELECT * FROM documents WHERE status = ? ORDER BY archived_at DESC, updated_at DESC",
            (DocumentStatus.ARCHIVED.value,),
        ).fetchall()
        return [self._row_to_document(row) for row in rows]

    @synchronized_db_method
    def get_document(self, document_id: str) -> Document:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Document '{document_id}' was not found")
        return self._row_to_document(row)

    @synchronized_db_method
    def update_document(
        self,
        document_id: str,
        *,
        title: Optional[str] = None,
        target_audience: Optional[str] = None,
        objective: Optional[str] = None,
        tone: Optional[str] = None,
        constraints: Optional[str] = None,
        current_content: Optional[str] = None,
        change_type: RevisionChangeType = RevisionChangeType.MANUAL,
        prompt: Optional[str] = None,
    ) -> Document:
        document = self.get_document(document_id)
        self._ensure_editable(document)

        updated_title = title if title is not None else document.title
        updated_target = target_audience if target_audience is not None else document.target_audience
        updated_objective = objective if objective is not None else document.objective
        updated_tone = tone if tone is not None else document.tone
        updated_constraints = constraints if constraints is not None else document.constraints
        updated_content = current_content if current_content is not None else document.current_content
        now = self._now_iso()

        self._connection.execute(
            """
            UPDATE documents
            SET title = ?, target_audience = ?, objective = ?, tone = ?,
                constraints = ?, current_content = ?, updated_at = ?, last_opened_at = ?
            WHERE id = ?
            """,
            (
                updated_title,
                updated_target,
                updated_objective,
                updated_tone,
                updated_constraints,
                updated_content,
                now,
                now,
                document_id,
            ),
        )
        self._add_revision(
            document_id=document_id,
            change_type=change_type,
            prompt=prompt,
            content_snapshot=updated_content,
            created_at=now,
        )
        self._connection.commit()
        return self.get_document(document_id)

    @synchronized_db_method
    def finalize_document(self, document_id: str) -> Document:
        document = self.get_document(document_id)
        self._ensure_editable(document)
        now = self._now_iso()
        self._connection.execute(
            """
            UPDATE documents
            SET status = ?, finalized_at = ?, updated_at = ?, last_opened_at = ?
            WHERE id = ?
            """,
            (DocumentStatus.FINALIZED.value, now, now, now, document_id),
        )
        self._add_revision(
            document_id=document_id,
            change_type=RevisionChangeType.SYSTEM,
            prompt="finalize",
            content_snapshot=document.current_content,
            created_at=now,
        )
        self._connection.commit()
        return self.get_document(document_id)

    @synchronized_db_method
    def archive_document(self, document_id: str) -> Document:
        document = self.get_document(document_id)
        if document.status == DocumentStatus.ARCHIVED:
            return document

        now = self._now_iso()
        self._connection.execute(
            """
            UPDATE documents
            SET status = ?, archived_at = ?, updated_at = ?, last_opened_at = ?
            WHERE id = ?
            """,
            (DocumentStatus.ARCHIVED.value, now, now, now, document_id),
        )
        self._add_revision(
            document_id=document_id,
            change_type=RevisionChangeType.SYSTEM,
            prompt="archive:manual",
            content_snapshot=document.current_content,
            created_at=now,
        )
        self._connection.commit()
        return self.get_document(document_id)

    @synchronized_db_method
    def delete_document(self, document_id: str) -> None:
        self.get_document(document_id)
        self._connection.execute("DELETE FROM attachments WHERE document_id = ?", (document_id,))
        self._connection.execute("DELETE FROM exports WHERE document_id = ?", (document_id,))
        self._connection.execute("DELETE FROM document_revisions WHERE document_id = ?", (document_id,))
        self._connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self._connection.commit()

    @synchronized_db_method
    def run_archive_job(self, older_than_days: int = 30) -> int:
        threshold = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        threshold_iso = threshold.isoformat()
        now = self._now_iso()
        cursor = self._connection.execute(
            """
            UPDATE documents
            SET status = ?, archived_at = ?, updated_at = ?
            WHERE status = ? AND finalized_at IS NOT NULL AND finalized_at < ?
            """,
            (
                DocumentStatus.ARCHIVED.value,
                now,
                now,
                DocumentStatus.FINALIZED.value,
                threshold_iso,
            ),
        )
        archived_count = cursor.rowcount or 0
        if archived_count > 0:
            archived_docs = self._connection.execute(
                """
                SELECT id, current_content FROM documents
                WHERE status = ? AND archived_at = ?
                """,
                (DocumentStatus.ARCHIVED.value, now),
            ).fetchall()
            for row in archived_docs:
                self._add_revision(
                    document_id=row["id"],
                    change_type=RevisionChangeType.SYSTEM,
                    prompt="archive",
                    content_snapshot=row["current_content"],
                    created_at=now,
                )
        self._connection.commit()
        return archived_count

    @synchronized_db_method
    def get_revisions(self, document_id: str) -> list[DocumentRevision]:
        self.get_document(document_id)
        rows = self._connection.execute(
            """
            SELECT * FROM document_revisions
            WHERE document_id = ?
            ORDER BY revision_no ASC
            """,
            (document_id,),
        ).fetchall()
        return [self._row_to_revision(row) for row in rows]

    @synchronized_db_method
    def add_attachment(
        self,
        *,
        document_id: str,
        filename: str,
        mime_type: str,
        size_bytes: int,
        storage_path: str,
        extraction_status: ExtractionStatus,
        extracted_text: str,
        extraction_error: str,
    ) -> Attachment:
        self.get_document(document_id)
        now = self._now_iso()
        cursor = self._connection.execute(
            """
            INSERT INTO attachments (
                document_id, filename, mime_type, size_bytes, storage_path,
                extraction_status, extracted_text, extraction_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                mime_type,
                size_bytes,
                storage_path,
                extraction_status.value,
                extracted_text,
                extraction_error,
                now,
            ),
        )
        self._connection.execute(
            "UPDATE documents SET updated_at = ?, last_opened_at = ? WHERE id = ?",
            (now, now, document_id),
        )
        self._connection.commit()
        attachment_id = cursor.lastrowid
        row = self._connection.execute(
            "SELECT * FROM attachments WHERE id = ?", (attachment_id,)
        ).fetchone()
        return self._row_to_attachment(row)

    @synchronized_db_method
    def add_attachment_pending(
        self,
        *,
        document_id: str,
        filename: str,
        mime_type: str,
        size_bytes: int,
        storage_path: str,
    ) -> Attachment:
        return self.add_attachment(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            storage_path=storage_path,
            extraction_status=ExtractionStatus.PENDING,
            extracted_text="",
            extraction_error="",
        )

    @synchronized_db_method
    def update_attachment_extraction(
        self,
        *,
        attachment_id: int,
        extraction_status: ExtractionStatus,
        extracted_text: str,
        extraction_error: str,
    ) -> Attachment:
        now = self._now_iso()
        row = self._connection.execute(
            "SELECT * FROM attachments WHERE id = ?",
            (attachment_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Attachment '{attachment_id}' was not found")

        self._connection.execute(
            """
            UPDATE attachments
            SET extraction_status = ?, extracted_text = ?, extraction_error = ?
            WHERE id = ?
            """,
            (extraction_status.value, extracted_text, extraction_error, attachment_id),
        )
        self._connection.execute(
            "UPDATE documents SET updated_at = ?, last_opened_at = ? WHERE id = ?",
            (now, now, row["document_id"]),
        )
        self._connection.commit()
        updated_row = self._connection.execute(
            "SELECT * FROM attachments WHERE id = ?",
            (attachment_id,),
        ).fetchone()
        return self._row_to_attachment(updated_row)

    @synchronized_db_method
    def get_attachment(self, attachment_id: int) -> Attachment:
        row = self._connection.execute(
            "SELECT * FROM attachments WHERE id = ?",
            (attachment_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Attachment '{attachment_id}' was not found")
        return self._row_to_attachment(row)

    @synchronized_db_method
    def list_attachments(self, document_id: str) -> list[Attachment]:
        self.get_document(document_id)
        rows = self._connection.execute(
            """
            SELECT * FROM attachments
            WHERE document_id = ?
            ORDER BY created_at DESC
            """,
            (document_id,),
        ).fetchall()
        return [self._row_to_attachment(row) for row in rows]

    @synchronized_db_method
    def add_export_record(
        self,
        *,
        document_id: str,
        export_type: ExportType,
        status: ExportStatus,
        external_url: str = "",
        file_path: str = "",
        error_message: str = "",
    ) -> ExportRecord:
        self.get_document(document_id)
        now = self._now_iso()
        cursor = self._connection.execute(
            """
            INSERT INTO exports (
                document_id, export_type, status, external_url, file_path, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                export_type.value,
                status.value,
                external_url,
                file_path,
                error_message,
                now,
            ),
        )
        self._connection.commit()
        row = self._connection.execute(
            "SELECT * FROM exports WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return self._row_to_export_record(row)

    @synchronized_db_method
    def list_export_records(self, document_id: str) -> list[ExportRecord]:
        self.get_document(document_id)
        rows = self._connection.execute(
            """
            SELECT * FROM exports
            WHERE document_id = ?
            ORDER BY created_at DESC
            """,
            (document_id,),
        ).fetchall()
        return [self._row_to_export_record(row) for row in rows]

    @synchronized_db_method
    def upsert_integration_setting(self, *, provider: str, config_json: str) -> IntegrationSetting:
        now = self._now_iso()
        self._connection.execute(
            """
            INSERT INTO integration_settings (provider, config_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at = excluded.updated_at
            """,
            (provider, config_json, now),
        )
        self._connection.commit()
        row = self._connection.execute(
            "SELECT * FROM integration_settings WHERE provider = ?",
            (provider,),
        ).fetchone()
        return self._row_to_integration_setting(row)

    @synchronized_db_method
    def get_integration_setting(self, provider: str) -> Optional[IntegrationSetting]:
        row = self._connection.execute(
            "SELECT * FROM integration_settings WHERE provider = ?",
            (provider,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_integration_setting(row)

    @synchronized_db_method
    def enqueue_job(
        self,
        *,
        job_type: str,
        payload: dict,
        max_attempts: int = 3,
        request_id: str = "",
    ) -> JobRecord:
        now = self._now_iso()
        payload_json = json.dumps(payload, ensure_ascii=True)
        cursor = self._connection.execute(
            """
            INSERT INTO jobs (
                job_type, status, payload_json, result_json, error_message, failure_class,
                attempts, max_attempts, next_run_at, request_id, created_at, updated_at,
                started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_type,
                JobStatus.PENDING.value,
                payload_json,
                "",
                "",
                "",
                0,
                max_attempts,
                now,
                request_id,
                now,
                now,
                None,
                None,
            ),
        )
        self._connection.commit()
        return self.get_job(int(cursor.lastrowid))

    @synchronized_db_method
    def claim_next_job(self) -> Optional[JobRecord]:
        now = self._now_iso()
        self._connection.execute("BEGIN IMMEDIATE")
        row = self._connection.execute(
            """
            SELECT * FROM jobs
            WHERE status = ? AND next_run_at <= ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (JobStatus.PENDING.value, now),
        ).fetchone()
        if row is None:
            self._connection.commit()
            return None

        attempts = int(row["attempts"]) + 1
        self._connection.execute(
            """
            UPDATE jobs
            SET status = ?, attempts = ?, started_at = COALESCE(started_at, ?), updated_at = ?
            WHERE id = ?
            """,
            (JobStatus.RUNNING.value, attempts, now, now, row["id"]),
        )
        self._connection.commit()
        updated_row = self._connection.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (row["id"],),
        ).fetchone()
        return self._row_to_job(updated_row)

    @synchronized_db_method
    def mark_job_success(self, *, job_id: int, result: dict) -> JobRecord:
        now = self._now_iso()
        self._connection.execute(
            """
            UPDATE jobs
            SET status = ?, result_json = ?, error_message = '', failure_class = '',
                updated_at = ?, finished_at = ?
            WHERE id = ?
            """,
            (JobStatus.SUCCESS.value, json.dumps(result, ensure_ascii=True), now, now, job_id),
        )
        self._connection.commit()
        return self.get_job(job_id)

    @synchronized_db_method
    def mark_job_failure(
        self,
        *,
        job_id: int,
        error_message: str,
        failure_class: JobFailureClass,
        retry_delay_seconds: int = 10,
    ) -> JobRecord:
        current = self.get_job(job_id)
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        should_retry = (
            failure_class == JobFailureClass.RETRYABLE
            and current.attempts < current.max_attempts
        )
        if should_retry:
            next_run_at = (now + timedelta(seconds=retry_delay_seconds)).isoformat()
            status = JobStatus.PENDING.value
            finished_at = None
        else:
            next_run_at = now_iso
            status = JobStatus.FAILED.value
            finished_at = now_iso

        self._connection.execute(
            """
            UPDATE jobs
            SET status = ?, error_message = ?, failure_class = ?, updated_at = ?,
                next_run_at = ?, finished_at = ?
            WHERE id = ?
            """,
            (
                status,
                error_message,
                failure_class.value,
                now_iso,
                next_run_at,
                finished_at,
                job_id,
            ),
        )
        self._connection.commit()
        return self.get_job(job_id)

    @synchronized_db_method
    def get_job(self, job_id: int) -> JobRecord:
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Job '{job_id}' was not found")
        return self._row_to_job(row)

    @synchronized_db_method
    def list_jobs(self, *, limit: int = 100) -> list[JobRecord]:
        rows = self._connection.execute(
            """
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_job(row) for row in rows]

    @synchronized_db_method
    def queue_depth(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS cnt FROM jobs WHERE status = ?",
            (JobStatus.PENDING.value,),
        ).fetchone()
        return int(row["cnt"])

    @synchronized_db_method
    def job_metrics(self) -> dict:
        totals = self._connection.execute(
            """
            SELECT status, COUNT(*) AS cnt
            FROM jobs
            GROUP BY status
            """
        ).fetchall()
        by_status = {row["status"]: int(row["cnt"]) for row in totals}
        row = self._connection.execute(
            """
            SELECT AVG((julianday(finished_at) - julianday(started_at)) * 86400000.0) AS avg_ms
            FROM jobs
            WHERE started_at IS NOT NULL AND finished_at IS NOT NULL
            """
        ).fetchone()
        avg_ms = float(row["avg_ms"]) if row and row["avg_ms"] is not None else 0.0
        total_jobs = sum(by_status.values())
        failed_jobs = by_status.get(JobStatus.FAILED.value, 0)
        failure_rate = (failed_jobs / total_jobs) if total_jobs else 0.0
        return {
            "total_jobs": total_jobs,
            "queue_depth": self.queue_depth(),
            "by_status": by_status,
            "avg_job_latency_ms": avg_ms,
            "job_failure_rate": failure_rate,
        }

    @synchronized_db_method
    def _add_revision(
        self,
        *,
        document_id: str,
        change_type: RevisionChangeType,
        prompt: Optional[str],
        content_snapshot: str,
        created_at: str,
    ) -> None:
        current = self._connection.execute(
            "SELECT COALESCE(MAX(revision_no), 0) AS max_revision FROM document_revisions WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        next_revision = int(current["max_revision"]) + 1
        self._connection.execute(
            """
            INSERT INTO document_revisions (
                document_id, revision_no, change_type, prompt, content_snapshot, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, next_revision, change_type.value, prompt, content_snapshot, created_at),
        )

    @staticmethod
    def _ensure_editable(document: Document) -> None:
        if document.status != DocumentStatus.IN_PROGRESS:
            raise DocumentLockedError(
                f"Document '{document.id}' is locked because status is {document.status.value}"
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        return datetime.fromisoformat(value)

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            title=row["title"],
            doc_type=row["doc_type"],
            target_audience=row["target_audience"],
            language=row["language"],
            objective=row["objective"],
            tone=row["tone"],
            constraints=row["constraints"],
            status=DocumentStatus(row["status"]),
            current_content=row["current_content"],
            last_opened_at=self._parse_dt(row["last_opened_at"]),
            finalized_at=self._parse_dt(row["finalized_at"]),
            archived_at=self._parse_dt(row["archived_at"]),
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
        )

    def _row_to_revision(self, row: sqlite3.Row) -> DocumentRevision:
        return DocumentRevision(
            id=row["id"],
            document_id=row["document_id"],
            revision_no=row["revision_no"],
            change_type=RevisionChangeType(row["change_type"]),
            prompt=row["prompt"],
            content_snapshot=row["content_snapshot"],
            created_at=self._parse_dt(row["created_at"]),
        )

    def _row_to_attachment(self, row: sqlite3.Row) -> Attachment:
        return Attachment(
            id=row["id"],
            document_id=row["document_id"],
            filename=row["filename"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            storage_path=row["storage_path"],
            extraction_status=ExtractionStatus(row["extraction_status"]),
            extracted_text=row["extracted_text"],
            extraction_error=row["extraction_error"],
            created_at=self._parse_dt(row["created_at"]),
        )

    def _row_to_export_record(self, row: sqlite3.Row) -> ExportRecord:
        return ExportRecord(
            id=row["id"],
            document_id=row["document_id"],
            export_type=ExportType(row["export_type"]),
            status=ExportStatus(row["status"]),
            external_url=row["external_url"],
            file_path=row["file_path"],
            error_message=row["error_message"],
            created_at=self._parse_dt(row["created_at"]),
        )

    def _row_to_integration_setting(self, row: sqlite3.Row) -> IntegrationSetting:
        return IntegrationSetting(
            provider=row["provider"],
            config_json=row["config_json"],
            updated_at=self._parse_dt(row["updated_at"]),
        )

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            job_type=row["job_type"],
            status=JobStatus(row["status"]),
            payload_json=row["payload_json"],
            result_json=row["result_json"],
            error_message=row["error_message"],
            failure_class=row["failure_class"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            next_run_at=self._parse_dt(row["next_run_at"]),
            request_id=row["request_id"],
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
            started_at=self._parse_dt(row["started_at"]),
            finished_at=self._parse_dt(row["finished_at"]),
        )
