from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
import time
from typing import Callable

from .ai import AIContentError, AIService, OutlineContext
from .drive_client import ManagerGoogleDriveClient
from .exporting import build_drive_export_name, export_markdown_like_to_docx
from .logging_utils import log_event
from .models import ExportStatus, ExportType, JobFailureClass, RevisionChangeType
from .parsers import parse_attachment
from .service import DocumentLockedError, DocumentService, NotFoundError

JOB_PARSE_ATTACHMENT = "parse_attachment"
JOB_GENERATE_OUTLINE = "generate_outline"
JOB_REWRITE_FULL = "rewrite_full"
JOB_REWRITE_SELECTION = "rewrite_selection"
JOB_EXPORT_DOCX = "export_docx"
JOB_EXPORT_DRIVE = "export_drive"
JOB_ARCHIVE_SCAN = "archive_scan"


@dataclass(slots=True)
class JobRunnerConfig:
    db_path: str
    exports_dir: Path
    poll_interval_sec: float = 1.0


class JobRunner:
    def __init__(self, config: JobRunnerConfig):
        self._config = config
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="manager-job-runner", daemon=True)
        self._thread.start()
        log_event("job_runner_started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        log_event("job_runner_stopped")

    def _run_loop(self) -> None:
        service = DocumentService(self._config.db_path)
        ai_service = AIService()
        handlers: dict[str, Callable[[dict], dict]] = {
            JOB_PARSE_ATTACHMENT: lambda payload: self._handle_parse_attachment(service, payload),
            JOB_GENERATE_OUTLINE: lambda payload: self._handle_generate_outline(service, ai_service, payload),
            JOB_REWRITE_FULL: lambda payload: self._handle_rewrite_full(service, ai_service, payload),
            JOB_REWRITE_SELECTION: lambda payload: self._handle_rewrite_selection(service, ai_service, payload),
            JOB_EXPORT_DOCX: lambda payload: self._handle_export_docx(service, payload),
            JOB_EXPORT_DRIVE: lambda payload: self._handle_export_drive(service, payload),
            JOB_ARCHIVE_SCAN: lambda payload: self._handle_archive_scan(service, payload),
        }

        try:
            while not self._stop_event.is_set():
                job = service.claim_next_job()
                if job is None:
                    time.sleep(self._config.poll_interval_sec)
                    continue

                payload = json.loads(job.payload_json)
                log_event(
                    "job_claimed",
                    job_id=job.id,
                    job_type=job.job_type,
                    attempts=job.attempts,
                )

                handler = handlers.get(job.job_type)
                if handler is None:
                    service.mark_job_failure(
                        job_id=job.id,
                        error_message=f"Unsupported job type '{job.job_type}'",
                        failure_class=JobFailureClass.PERMANENT,
                    )
                    continue

                try:
                    result = handler(payload)
                    service.mark_job_success(job_id=job.id, result=result)
                    log_event("job_succeeded", job_id=job.id, job_type=job.job_type)
                except Exception as exc:  # pragma: no cover - runtime fallback
                    failure_class = _classify_failure(exc)
                    service.mark_job_failure(
                        job_id=job.id,
                        error_message=str(exc),
                        failure_class=failure_class,
                        retry_delay_seconds=max(5, job.attempts * 5),
                    )
                    log_event(
                        "job_failed",
                        job_id=job.id,
                        job_type=job.job_type,
                        failure_class=failure_class.value,
                        error=str(exc),
                    )
        finally:
            service.close()

    def _handle_parse_attachment(self, service: DocumentService, payload: dict) -> dict:
        attachment_id = int(payload["attachment_id"])
        attachment = service.get_attachment(attachment_id)
        raw_bytes = Path(attachment.storage_path).read_bytes()
        parsed = parse_attachment(attachment.filename, raw_bytes)
        updated = service.update_attachment_extraction(
            attachment_id=attachment_id,
            extraction_status=parsed.extraction_status,
            extracted_text=parsed.extracted_text,
            extraction_error=parsed.extraction_error,
        )
        return {
            "attachment_id": updated.id,
            "extraction_status": updated.extraction_status.value,
        }

    def _handle_generate_outline(
        self,
        service: DocumentService,
        ai_service: AIService,
        payload: dict,
    ) -> dict:
        document_id = str(payload["document_id"])
        instructions = str(payload.get("instructions", ""))
        document = service.get_document(document_id)
        attachments = service.list_attachments(document_id)
        summary = _build_attachment_summary(attachments)
        outline = ai_service.generate_outline(
            document,
            OutlineContext(instructions=instructions, attachments_summary=summary),
        )
        updated = service.update_document(
            document_id,
            current_content=outline,
            change_type=RevisionChangeType.AI_FULL,
            prompt=f"outline:{instructions.strip()}",
        )
        return {"document_id": updated.id, "content_len": len(updated.current_content)}

    def _handle_rewrite_full(
        self,
        service: DocumentService,
        ai_service: AIService,
        payload: dict,
    ) -> dict:
        document_id = str(payload["document_id"])
        prompt = str(payload["prompt"])
        document = service.get_document(document_id)
        rewritten = ai_service.rewrite_full(document, prompt)
        updated = service.update_document(
            document_id,
            current_content=rewritten,
            change_type=RevisionChangeType.AI_FULL,
            prompt=prompt,
        )
        return {"document_id": updated.id, "content_len": len(updated.current_content)}

    def _handle_rewrite_selection(
        self,
        service: DocumentService,
        ai_service: AIService,
        payload: dict,
    ) -> dict:
        document_id = str(payload["document_id"])
        selection_start = int(payload["selection_start"])
        selection_end = int(payload["selection_end"])
        prompt = str(payload["prompt"])
        document = service.get_document(document_id)
        content = document.current_content
        if selection_end <= selection_start:
            raise ValueError("selection_end must be greater than selection_start")
        if selection_end > len(content):
            raise ValueError("selection_end exceeds current content length")

        selected = content[selection_start:selection_end]
        left = content[max(0, selection_start - 500):selection_start]
        right = content[selection_end:selection_end + 500]
        rewritten = ai_service.rewrite_selection(
            selected_text=selected,
            prompt=prompt,
            left_context=left,
            right_context=right,
        )
        merged = content[:selection_start] + rewritten + content[selection_end:]
        updated = service.update_document(
            document_id,
            current_content=merged,
            change_type=RevisionChangeType.AI_PARTIAL,
            prompt=prompt,
        )
        return {"document_id": updated.id, "content_len": len(updated.current_content)}

    def _handle_export_docx(self, service: DocumentService, payload: dict) -> dict:
        document_id = str(payload["document_id"])
        document = service.get_document(document_id)
        try:
            exported = export_markdown_like_to_docx(
                title=document.title,
                content=document.current_content,
                output_dir=self._config.exports_dir,
                document_id=document.id,
            )
            record = service.add_export_record(
                document_id=document_id,
                export_type=ExportType.DOCX,
                status=ExportStatus.SUCCESS,
                file_path=str(exported),
            )
        except Exception as exc:
            service.add_export_record(
                document_id=document_id,
                export_type=ExportType.DOCX,
                status=ExportStatus.FAILED,
                error_message=str(exc),
            )
            raise
        return {"export_id": record.id, "file_path": record.file_path}

    def _handle_export_drive(self, service: DocumentService, payload: dict) -> dict:
        document_id = str(payload["document_id"])
        document = service.get_document(document_id)
        setting = service.get_integration_setting("google_drive")
        if setting is None:
            service.add_export_record(
                document_id=document_id,
                export_type=ExportType.GOOGLE_DRIVE,
                status=ExportStatus.FAILED,
                error_message="Google Drive settings are missing",
            )
            raise RuntimeError("Google Drive settings are missing")
        config = json.loads(setting.config_json)

        try:
            exported = export_markdown_like_to_docx(
                title=document.title,
                content=document.current_content,
                output_dir=self._config.exports_dir,
                document_id=document.id,
            )
            drive_document_name = build_drive_export_name(
                title=document.title,
                document_id=document.id,
            )
            drive_client = _build_drive_client_for_job(service=service, config=config)
            result = drive_client.upload_docx_as_google_doc(
                exported,
                document_name=drive_document_name,
            )
            external_url = str(result.get("webViewLink", "")).strip()
            if not external_url:
                verification = result.get("verification", {})
                external_url = str(verification.get("web_view_link", "")).strip()

            record = service.add_export_record(
                document_id=document_id,
                export_type=ExportType.GOOGLE_DRIVE,
                status=ExportStatus.SUCCESS,
                external_url=external_url,
                file_path=str(exported),
            )
        except Exception as exc:
            service.add_export_record(
                document_id=document_id,
                export_type=ExportType.GOOGLE_DRIVE,
                status=ExportStatus.FAILED,
                error_message=str(exc),
            )
            raise
        return {"export_id": record.id, "external_url": external_url}

    @staticmethod
    def _handle_archive_scan(service: DocumentService, payload: dict) -> dict:
        older_than_days = int(payload.get("older_than_days", 30))
        archived_count = service.run_archive_job(older_than_days=older_than_days)
        return {"archived_count": archived_count}


def _build_attachment_summary(attachments) -> str:
    max_total_chars = 70000
    attachments_with_text = [
        attachment
        for attachment in attachments
        if str(getattr(attachment, "extracted_text", "") or "").strip()
    ]
    if not attachments_with_text:
        return ""
    per_attachment_cap = min(16000, max(2500, max_total_chars // max(1, len(attachments_with_text))))

    lines: list[str] = []
    for attachment in attachments_with_text:
        extracted_text = str(getattr(attachment, "extracted_text", "") or "").strip()
        lines.append(f"- {attachment.filename}")
        selected = _select_attachment_lines(
            extracted_text,
            limit=28,
            per_attachment_cap=per_attachment_cap,
        )
        if selected:
            lines.extend([f"  - {line}" for line in selected])
        lines = _trim_lines(lines, max_chars=max_total_chars)
        if _lines_char_len(lines) >= max_total_chars:
            break
    return "\n".join(lines)


def _select_attachment_lines(text: str, *, limit: int, per_attachment_cap: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        normalized = re.sub(r"\s+", " ", raw_line.replace("\t", " | ").strip()).strip("- ").strip()
        if len(normalized) < 18:
            continue
        if len(normalized) > 900:
            normalized = normalized[:900].rstrip()
            if " " in normalized:
                normalized = normalized.rsplit(" ", 1)[0].rstrip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
        if len(selected) >= limit:
            break

    selected.extend(_context_windows(text, per_attachment_cap=per_attachment_cap))
    output: list[str] = []
    seen_output: set[str] = set()
    for line in selected:
        normalized = re.sub(r"\s+", " ", str(line or "").strip())
        if not normalized or normalized in seen_output:
            continue
        seen_output.add(normalized)
        output.append(normalized)
    return _trim_lines(output, max_chars=per_attachment_cap)


def _context_windows(text: str, *, per_attachment_cap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    if len(normalized) <= per_attachment_cap:
        return [normalized]
    window = max(600, min(2400, per_attachment_cap // 4))
    mid_start = max(0, (len(normalized) // 2) - (window // 2))
    return [
        f"Context (start): {normalized[:window].rstrip()}",
        f"Context (middle): {normalized[mid_start:mid_start + window].strip()}",
        f"Context (end): {normalized[-window:].lstrip()}",
    ]


def _trim_lines(lines: list[str], *, max_chars: int) -> list[str]:
    output: list[str] = []
    running = 0
    for line in lines:
        proposed = running + len(line) + 1
        if proposed > max_chars:
            break
        output.append(line)
        running = proposed
    return output


def _lines_char_len(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def _classify_failure(exc: Exception) -> JobFailureClass:
    if isinstance(exc, (NotFoundError, DocumentLockedError, AIContentError, ValueError)):
        return JobFailureClass.PERMANENT

    message = str(exc).lower()
    retryable_fragments = [
        "timeout",
        "tempor",
        "429",
        "rate limit",
        "502",
        "503",
        "504",
        "connection reset",
    ]
    if any(fragment in message for fragment in retryable_fragments):
        return JobFailureClass.RETRYABLE
    return JobFailureClass.PERMANENT


def _build_drive_client_for_job(
    *,
    service: DocumentService,
    config: dict,
) -> ManagerGoogleDriveClient:
    mode = str(config.get("credential_mode", "")).strip().lower()
    if mode == "oauth_quick_connect":
        client_id = str(config.get("client_id", "")).strip()
        client_secret = str(config.get("client_secret", "")).strip()
        token_json = str(config.get("token_json", "")).strip()
        if not client_id or not client_secret or not token_json:
            raise RuntimeError("Quick connect OAuth configuration is incomplete")

        def _persist_token(new_token_json: str) -> None:
            updated = dict(config)
            updated["token_json"] = new_token_json
            service.upsert_integration_setting(
                provider="google_drive",
                config_json=json.dumps(updated),
            )

        return ManagerGoogleDriveClient.from_oauth_config(
            client_id=client_id,
            client_secret=client_secret,
            token_json=token_json,
            folder_name=str(config.get("folder_name", "Manager Documents")),
            folder_id=str(config.get("folder_id", "")),
            token_store_callback=_persist_token,
        )

    raise RuntimeError(
        "Google Drive integration is OAuth-only. "
        "Reconnect with quick connect OAuth."
    )
