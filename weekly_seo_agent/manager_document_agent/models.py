from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class DocumentStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    FINALIZED = "FINALIZED"
    ARCHIVED = "ARCHIVED"


class RevisionChangeType(str, Enum):
    MANUAL = "MANUAL"
    AI_FULL = "AI_FULL"
    AI_PARTIAL = "AI_PARTIAL"
    SYSTEM = "SYSTEM"


class ExtractionStatus(str, Enum):
    PENDING = "PENDING"
    OK = "OK"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class ExportType(str, Enum):
    DOCX = "DOCX"
    GOOGLE_DRIVE = "GOOGLE_DRIVE"


class ExportStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class JobFailureClass(str, Enum):
    RETRYABLE = "RETRYABLE"
    PERMANENT = "PERMANENT"


@dataclass(slots=True)
class Document:
    id: str
    title: str
    doc_type: str
    target_audience: str
    language: str
    objective: str
    tone: str
    constraints: str
    status: DocumentStatus
    current_content: str
    last_opened_at: Optional[datetime]
    finalized_at: Optional[datetime]
    archived_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class DocumentRevision:
    id: int
    document_id: str
    revision_no: int
    change_type: RevisionChangeType
    prompt: Optional[str]
    content_snapshot: str
    created_at: datetime


@dataclass(slots=True)
class Attachment:
    id: int
    document_id: str
    filename: str
    mime_type: str
    size_bytes: int
    storage_path: str
    extraction_status: ExtractionStatus
    extracted_text: str
    extraction_error: str
    created_at: datetime


@dataclass(slots=True)
class ExportRecord:
    id: int
    document_id: str
    export_type: ExportType
    status: ExportStatus
    external_url: str
    file_path: str
    error_message: str
    created_at: datetime


@dataclass(slots=True)
class IntegrationSetting:
    provider: str
    config_json: str
    updated_at: datetime


@dataclass(slots=True)
class JobRecord:
    id: int
    job_type: str
    status: JobStatus
    payload_json: str
    result_json: str
    error_message: str
    failure_class: str
    attempts: int
    max_attempts: int
    next_run_at: datetime
    request_id: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
