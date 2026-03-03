from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import os
import re
import time
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel, Field

from .ai import AIContentError, AIService, OutlineContext
from .auth import AuthError, AuthManager, AuthUser, has_any_role
from .audit import export_audit_log_jsonl
from .drive_client import ManagerGoogleDriveClient
from .exporting import build_drive_export_name, export_markdown_like_to_docx
from .job_runner import (
    JOB_ARCHIVE_SCAN,
    JOB_EXPORT_DOCX,
    JOB_EXPORT_DRIVE,
    JOB_GENERATE_OUTLINE,
    JOB_PARSE_ATTACHMENT,
    JOB_REWRITE_FULL,
    JOB_REWRITE_SELECTION,
    JobRunner,
    JobRunnerConfig,
)
from .logging_utils import log_event
from .metrics import RuntimeMetrics
from .models import (
    Attachment,
    Document,
    DocumentStatus,
    ExportRecord,
    ExportStatus,
    ExportType,
    ExtractionStatus,
    JobRecord,
    JobStatus,
    RevisionChangeType,
)
from .parsers import (
    AttachmentValidationError,
    parse_attachment,
    validate_attachment,
    validate_attachment_content,
)
from .planning import (
    PLANNING_STATUS_APPROVED,
    PLANNING_STATUS_DOCUMENT_CREATED,
    new_planning_session,
    planning_session_approve,
    planning_session_build_document_payload,
    planning_session_mark_document_created,
    planning_session_user_turn,
    prune_planning_sessions,
)
from .service import DocumentLockedError, DocumentService, NotFoundError
from .web_research import run_web_research

DEFAULT_DB_PATH = Path("outputs/manager_document_agent.db")
DEFAULT_ATTACHMENTS_DIR = Path("outputs/manager_document_agent/attachments")
DEFAULT_EXPORTS_DIR = Path("outputs/manager_document_agent/exports")
DEFAULT_AUDIT_DIR = Path("outputs/manager_document_agent/audit")


class CreateDocumentRequest(BaseModel):
    title: str = Field(min_length=1)
    doc_type: str = Field(min_length=1)
    target_audience: str = Field(min_length=1)
    language: str = Field(min_length=1)
    objective: str = Field(default="")
    tone: str = Field(default="")
    constraints: str = Field(default="")
    current_content: str = Field(default="")


class UpdateDocumentRequest(BaseModel):
    title: Optional[str] = None
    target_audience: Optional[str] = None
    objective: Optional[str] = None
    tone: Optional[str] = None
    constraints: Optional[str] = None
    current_content: Optional[str] = None
    change_type: RevisionChangeType = RevisionChangeType.MANUAL
    prompt: Optional[str] = None


class OutlineRequest(BaseModel):
    instructions: str = Field(default="")


class RewriteRequest(BaseModel):
    prompt: str = Field(min_length=1)


class RewriteSelectionRequest(BaseModel):
    selection_start: int = Field(ge=0)
    selection_end: int = Field(ge=0)
    prompt: str = Field(min_length=1)


class DriveAttachmentImportRequest(BaseModel):
    file_ref: str = Field(min_length=1)


class WebResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    region: str = Field(default="us-en")
    max_results: int = Field(default=6, ge=1, le=15)
    fetch_pages: bool = Field(default=True)
    max_pages: int = Field(default=3, ge=0, le=8)
    page_char_limit: int = Field(default=1800, ge=200, le=6000)


class WebResearchItemResponse(BaseModel):
    title: str
    url: str
    snippet: str
    source: str
    page_excerpt: str = ""
    page_error: str = ""


class WebResearchResponse(BaseModel):
    query: str
    provider: str
    region: str
    warning: str
    summary_text: str
    items: list[WebResearchItemResponse]


class DocumentResponse(BaseModel):
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


class DocumentDeleteResponse(BaseModel):
    deleted: bool
    document_id: str
    removed_attachment_files: int = 0


class DocumentPlanningBriefResponse(BaseModel):
    title: str
    doc_type: str
    target_audience: str
    language: str
    objective: str
    tone: str
    constraints: str
    emphasis: str = ""
    decisions_needed: str = ""
    must_include: str = ""


class DocumentPlanningMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class DocumentPlanningSessionResponse(BaseModel):
    id: str
    status: str
    step_index: int
    ready_to_create: bool
    created_document_id: str = ""
    suggested_points: list[str]
    approved_points: list[str]
    brief: DocumentPlanningBriefResponse
    messages: list[DocumentPlanningMessageResponse]
    created_at: str
    updated_at: str


class DocumentPlanningStartRequest(BaseModel):
    title: str = Field(default="")
    doc_type: str = Field(default="MANAGEMENT_BRIEF")
    target_audience: str = Field(default="Management")
    language: str = Field(default="pl")
    objective: str = Field(default="")
    tone: str = Field(default="formal")
    constraints: str = Field(default="")


class DocumentPlanningMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class DocumentPlanningApproveRequest(BaseModel):
    approved_points: list[str] = Field(default_factory=list)


class DocumentPlanningCreateDocumentRequest(BaseModel):
    include_chat_summary: bool = Field(default=True)


class DocumentPlanningCreateDocumentResponse(BaseModel):
    session: DocumentPlanningSessionResponse
    document: DocumentResponse


class RevisionResponse(BaseModel):
    id: int
    document_id: str
    revision_no: int
    change_type: RevisionChangeType
    prompt: Optional[str]
    content_snapshot: str
    created_at: datetime


class AttachmentResponse(BaseModel):
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


class ArchiveJobResponse(BaseModel):
    archived_count: int


class GoogleDriveSettingsRequest(BaseModel):
    folder_name: str = Field(default="Manager Documents")
    folder_id: str = Field(default="")


class GoogleDriveSettingsResponse(BaseModel):
    provider: str
    config: dict
    updated_at: datetime


class GoogleDriveQuickConnectStartRequest(BaseModel):
    folder_name: str = Field(default="Manager Documents")
    folder_id: str = Field(default="")
    client_id: str = Field(default="")
    client_secret: str = Field(default="")
    redirect_uri: str = Field(default="")


class GoogleDriveQuickConnectStartResponse(BaseModel):
    authorization_url: str
    state: str
    redirect_uri: str
    source: str


class GoogleDriveQuickConnectVerifyResponse(BaseModel):
    ok: bool
    folder_id: str
    folder_name: str
    auth_mode: str
    verified_at: str


class GoogleDriveFileItemResponse(BaseModel):
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    web_view_link: str


class GoogleDriveFileListResponse(BaseModel):
    files: list[GoogleDriveFileItemResponse]


class ExportRecordResponse(BaseModel):
    id: int
    document_id: str
    export_type: ExportType
    status: ExportStatus
    external_url: str
    file_path: str
    error_message: str
    created_at: datetime


class JobResponse(BaseModel):
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


class EnqueuedJobResponse(BaseModel):
    job: JobResponse
    related_attachment_id: Optional[int] = None


class AuditExportResponse(BaseModel):
    file_path: str
    event_count: int


class AuthLoginMockRequest(BaseModel):
    token: str = Field(min_length=1)


class AuthLoginOAuthRequest(BaseModel):
    id_token: str = Field(min_length=1)


class AuthUserResponse(BaseModel):
    subject: str
    email: str
    roles: list[str]
    auth_mode: str


def create_app(
    db_path: str | Path = DEFAULT_DB_PATH,
    attachments_dir: str | Path = DEFAULT_ATTACHMENTS_DIR,
    exports_dir: str | Path = DEFAULT_EXPORTS_DIR,
    audit_dir: str | Path = DEFAULT_AUDIT_DIR,
    job_poll_interval_sec: float = 0.2,
) -> FastAPI:
    resolved_db = Path(db_path)
    resolved_attachments_dir = Path(attachments_dir)
    resolved_exports_dir = Path(exports_dir)
    resolved_audit_dir = Path(audit_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_db.parent.mkdir(parents=True, exist_ok=True)
        resolved_attachments_dir.mkdir(parents=True, exist_ok=True)
        resolved_exports_dir.mkdir(parents=True, exist_ok=True)
        resolved_audit_dir.mkdir(parents=True, exist_ok=True)
        app.state.document_service = DocumentService(str(resolved_db))
        app.state.ai_service = AIService()
        app.state.auth_manager = AuthManager.from_env()
        app.state.attachments_dir = resolved_attachments_dir
        app.state.exports_dir = resolved_exports_dir
        app.state.audit_dir = resolved_audit_dir
        app.state.drive_quick_connect_sessions = {}
        app.state.planning_sessions = {}
        app.state.metrics = RuntimeMetrics()
        app.state.job_runner = JobRunner(
            JobRunnerConfig(
                db_path=str(resolved_db),
                exports_dir=resolved_exports_dir,
                poll_interval_sec=job_poll_interval_sec,
            )
        )
        app.state.job_runner.start()
        try:
            yield
        finally:
            app.state.job_runner.stop()
            app.state.document_service.close()

    app = FastAPI(title="Manager Document Agent API", version="0.1.0", lifespan=lifespan)

    def _get_service() -> DocumentService:
        return app.state.document_service

    def _get_attachments_dir() -> Path:
        return app.state.attachments_dir

    def _get_exports_dir() -> Path:
        return app.state.exports_dir

    def _get_audit_dir() -> Path:
        return app.state.audit_dir

    def _get_ai_service() -> AIService:
        return app.state.ai_service

    def _get_metrics() -> RuntimeMetrics:
        return app.state.metrics

    def _get_auth_manager() -> AuthManager:
        return app.state.auth_manager

    def _get_planning_sessions() -> dict[str, dict]:
        return app.state.planning_sessions

    @app.middleware("http")
    async def request_metrics_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "").strip() or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        try:
            auth_manager: AuthManager = app.state.auth_manager
            if _requires_auth(request.url.path):
                user = auth_manager.authenticate(request.headers.get("Authorization"))
                required_roles = _required_roles_for_request(
                    method=request.method,
                    path=request.url.path,
                )
                if required_roles and not has_any_role(user, required_roles):
                    raise AuthError(
                        f"Forbidden: requires any role in {required_roles}",
                        status_code=403,
                    )
                request.state.auth_user = user
            response = await call_next(request)
            status_code = response.status_code
        except AuthError as exc:
            status_code = exc.status_code
            latency_ms = (time.perf_counter() - started) * 1000.0
            app.state.metrics.observe_request(status_code=status_code, latency_ms=latency_ms)
            log_event(
                "http_request",
                request_id=request_id,
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
                latency_ms=round(latency_ms, 2),
                error=str(exc),
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": str(exc)},
                headers={"X-Request-ID": request_id},
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            app.state.metrics.observe_request(status_code=status_code, latency_ms=latency_ms)
            log_event(
                "http_request",
                request_id=request_id,
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
                latency_ms=round(latency_ms, 2),
                error=str(exc),
            )
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        app.state.metrics.observe_request(status_code=status_code, latency_ms=latency_ms)
        response.headers["X-Request-ID"] = request_id
        log_event(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
        )
        return response

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def ui_root() -> HTMLResponse:
        ui_path = Path(__file__).parent / "ui" / "index.html"
        if not ui_path.exists():
            return HTMLResponse("<h1>Manager Document Agent UI not found</h1>", status_code=404)
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))

    @app.get("/auth/me", response_model=AuthUserResponse)
    def auth_me(request: Request) -> AuthUserResponse:
        user = getattr(request.state, "auth_user", None)
        if user is None:
            user = app.state.auth_manager.authenticate(request.headers.get("Authorization"))
        return _to_auth_user_response(user)

    @app.post("/auth/login/mock", response_model=AuthUserResponse)
    def auth_login_mock(
        payload: AuthLoginMockRequest,
        auth_manager: AuthManager = Depends(_get_auth_manager),
    ) -> AuthUserResponse:
        if auth_manager.mode != "mock":
            raise HTTPException(
                status_code=400,
                detail="Mock login is available only in MANAGER_DOCUMENT_AGENT_AUTH_MODE=mock",
            )
        user = auth_manager.login_mock(payload.token)
        return _to_auth_user_response(user)

    @app.post("/auth/login/oauth", response_model=AuthUserResponse)
    def auth_login_oauth(
        payload: AuthLoginOAuthRequest,
        auth_manager: AuthManager = Depends(_get_auth_manager),
    ) -> AuthUserResponse:
        if auth_manager.mode != "oauth":
            raise HTTPException(
                status_code=400,
                detail="OAuth login is available only in MANAGER_DOCUMENT_AGENT_AUTH_MODE=oauth",
            )
        user = auth_manager.login_oauth(payload.id_token)
        return _to_auth_user_response(user)

    @app.post(
        "/document-planning/sessions",
        response_model=DocumentPlanningSessionResponse,
    )
    def start_document_planning_session(
        payload: DocumentPlanningStartRequest,
        planning_sessions: dict[str, dict] = Depends(_get_planning_sessions),
    ) -> DocumentPlanningSessionResponse:
        prune_planning_sessions(sessions=planning_sessions)
        session = new_planning_session(initial_brief=payload.model_dump())
        planning_sessions[session["id"]] = session
        return _to_document_planning_session_response(session)

    @app.get(
        "/document-planning/sessions/{session_id}",
        response_model=DocumentPlanningSessionResponse,
    )
    def get_document_planning_session(
        session_id: str,
        planning_sessions: dict[str, dict] = Depends(_get_planning_sessions),
    ) -> DocumentPlanningSessionResponse:
        session = planning_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Planning session '{session_id}' was not found")
        return _to_document_planning_session_response(session)

    @app.post(
        "/document-planning/sessions/{session_id}/messages",
        response_model=DocumentPlanningSessionResponse,
    )
    def post_document_planning_message(
        session_id: str,
        payload: DocumentPlanningMessageRequest,
        planning_sessions: dict[str, dict] = Depends(_get_planning_sessions),
    ) -> DocumentPlanningSessionResponse:
        session = planning_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Planning session '{session_id}' was not found")
        try:
            planning_session_user_turn(session=session, user_message=payload.message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_document_planning_session_response(session)

    @app.post(
        "/document-planning/sessions/{session_id}/approve",
        response_model=DocumentPlanningSessionResponse,
    )
    def approve_document_planning_points(
        session_id: str,
        payload: DocumentPlanningApproveRequest,
        planning_sessions: dict[str, dict] = Depends(_get_planning_sessions),
    ) -> DocumentPlanningSessionResponse:
        session = planning_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Planning session '{session_id}' was not found")
        try:
            planning_session_approve(session=session, approved_points=payload.approved_points)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _to_document_planning_session_response(session)

    @app.post(
        "/document-planning/sessions/{session_id}/create-document",
        response_model=DocumentPlanningCreateDocumentResponse,
    )
    def create_document_from_planning_session(
        session_id: str,
        payload: DocumentPlanningCreateDocumentRequest,
        service: DocumentService = Depends(_get_service),
        planning_sessions: dict[str, dict] = Depends(_get_planning_sessions),
    ) -> DocumentPlanningCreateDocumentResponse:
        session = planning_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Planning session '{session_id}' was not found")
        if str(session.get("status", "")) == PLANNING_STATUS_DOCUMENT_CREATED:
            existing_id = str(session.get("created_document_id", "")).strip()
            if existing_id:
                try:
                    existing = service.get_document(existing_id)
                    return DocumentPlanningCreateDocumentResponse(
                        session=_to_document_planning_session_response(session),
                        document=_to_document_response(existing),
                    )
                except NotFoundError:
                    pass
            raise HTTPException(
                status_code=409,
                detail="Document was already created for this planning session",
            )
        if str(session.get("status", "")) != PLANNING_STATUS_APPROVED:
            raise HTTPException(
                status_code=409,
                detail="Planning session must be approved before creating a document",
            )
        try:
            document_payload = planning_session_build_document_payload(
                session=session,
                include_chat_summary=payload.include_chat_summary,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        created = service.create_document(**document_payload)
        planning_session_mark_document_created(session=session, document_id=created.id)
        return DocumentPlanningCreateDocumentResponse(
            session=_to_document_planning_session_response(session),
            document=_to_document_response(created),
        )

    @app.post("/documents", response_model=DocumentResponse)
    def create_document(
        payload: CreateDocumentRequest,
        service: DocumentService = Depends(_get_service),
    ) -> DocumentResponse:
        document = service.create_document(
            title=payload.title,
            doc_type=payload.doc_type,
            target_audience=payload.target_audience,
            language=payload.language,
            objective=payload.objective,
            tone=payload.tone,
            constraints=payload.constraints,
            current_content=payload.current_content,
        )
        return _to_document_response(document)

    @app.get("/documents", response_model=list[DocumentResponse])
    def list_documents(
        status: Optional[DocumentStatus] = None,
        service: DocumentService = Depends(_get_service),
    ) -> list[DocumentResponse]:
        documents = service.list_documents(status=status)
        return [_to_document_response(document) for document in documents]

    @app.get("/documents/archive", response_model=list[DocumentResponse])
    def list_archive(
        service: DocumentService = Depends(_get_service),
    ) -> list[DocumentResponse]:
        documents = service.list_archive()
        return [_to_document_response(document) for document in documents]

    @app.get("/documents/last-opened", response_model=Optional[DocumentResponse])
    def last_opened_document(
        service: DocumentService = Depends(_get_service),
    ) -> Optional[DocumentResponse]:
        documents = service.list_documents()
        if not documents:
            return None
        return _to_document_response(documents[0])

    @app.get("/documents/{document_id}", response_model=DocumentResponse)
    def get_document(
        document_id: str,
        service: DocumentService = Depends(_get_service),
    ) -> DocumentResponse:
        try:
            document = service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _to_document_response(document)

    @app.patch("/documents/{document_id}", response_model=DocumentResponse)
    def update_document(
        document_id: str,
        payload: UpdateDocumentRequest,
        service: DocumentService = Depends(_get_service),
    ) -> DocumentResponse:
        try:
            document = service.update_document(
                document_id,
                title=payload.title,
                target_audience=payload.target_audience,
                objective=payload.objective,
                tone=payload.tone,
                constraints=payload.constraints,
                current_content=payload.current_content,
                change_type=payload.change_type,
                prompt=payload.prompt,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DocumentLockedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _to_document_response(document)

    @app.post("/documents/{document_id}/outline", response_model=DocumentResponse)
    def generate_outline(
        document_id: str,
        payload: OutlineRequest,
        service: DocumentService = Depends(_get_service),
        ai_service: AIService = Depends(_get_ai_service),
    ) -> DocumentResponse:
        try:
            document = service.get_document(document_id)
            attachments = service.list_attachments(document_id)
            attachment_summary = _build_attachment_summary(attachments)
            outline = ai_service.generate_outline(
                document,
                OutlineContext(
                    instructions=payload.instructions,
                    attachments_summary=attachment_summary,
                ),
            )
            updated = service.update_document(
                document_id,
                current_content=outline,
                change_type=RevisionChangeType.AI_FULL,
                prompt=f"outline:{payload.instructions.strip()}",
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DocumentLockedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AIContentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _to_document_response(updated)

    @app.post("/documents/{document_id}/outline/async", response_model=EnqueuedJobResponse)
    def generate_outline_async(
        document_id: str,
        payload: OutlineRequest,
        request: Request,
        service: DocumentService = Depends(_get_service),
    ) -> EnqueuedJobResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        job = service.enqueue_job(
            job_type=JOB_GENERATE_OUTLINE,
            payload={
                "document_id": document_id,
                "instructions": payload.instructions,
            },
            request_id=str(getattr(request.state, "request_id", "")),
        )
        return EnqueuedJobResponse(job=_to_job_response(job))

    @app.post("/documents/{document_id}/rewrite", response_model=DocumentResponse)
    def rewrite_document(
        document_id: str,
        payload: RewriteRequest,
        service: DocumentService = Depends(_get_service),
        ai_service: AIService = Depends(_get_ai_service),
    ) -> DocumentResponse:
        try:
            document = service.get_document(document_id)
            rewritten = ai_service.rewrite_full(document, payload.prompt)
            updated = service.update_document(
                document_id,
                current_content=rewritten,
                change_type=RevisionChangeType.AI_FULL,
                prompt=payload.prompt,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DocumentLockedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AIContentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _to_document_response(updated)

    @app.post("/documents/{document_id}/rewrite/async", response_model=EnqueuedJobResponse)
    def rewrite_document_async(
        document_id: str,
        payload: RewriteRequest,
        request: Request,
        service: DocumentService = Depends(_get_service),
    ) -> EnqueuedJobResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        job = service.enqueue_job(
            job_type=JOB_REWRITE_FULL,
            payload={"document_id": document_id, "prompt": payload.prompt},
            request_id=str(getattr(request.state, "request_id", "")),
        )
        return EnqueuedJobResponse(job=_to_job_response(job))

    @app.post("/documents/{document_id}/rewrite-selection", response_model=DocumentResponse)
    def rewrite_selection(
        document_id: str,
        payload: RewriteSelectionRequest,
        service: DocumentService = Depends(_get_service),
        ai_service: AIService = Depends(_get_ai_service),
    ) -> DocumentResponse:
        try:
            document = service.get_document(document_id)
            content = document.current_content
            if payload.selection_end <= payload.selection_start:
                raise HTTPException(
                    status_code=400,
                    detail="selection_end must be greater than selection_start",
                )
            if payload.selection_end > len(content):
                raise HTTPException(
                    status_code=400,
                    detail="selection_end exceeds current content length",
                )

            selected_text = content[payload.selection_start:payload.selection_end]
            left = content[max(0, payload.selection_start - 500):payload.selection_start]
            right = content[payload.selection_end:payload.selection_end + 500]
            rewritten_fragment = ai_service.rewrite_selection(
                selected_text=selected_text,
                prompt=payload.prompt,
                left_context=left,
                right_context=right,
            )
            merged_content = (
                content[:payload.selection_start]
                + rewritten_fragment
                + content[payload.selection_end:]
            )
            updated = service.update_document(
                document_id,
                current_content=merged_content,
                change_type=RevisionChangeType.AI_PARTIAL,
                prompt=payload.prompt,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DocumentLockedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AIContentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _to_document_response(updated)

    @app.post("/documents/{document_id}/rewrite-selection/async", response_model=EnqueuedJobResponse)
    def rewrite_selection_async(
        document_id: str,
        payload: RewriteSelectionRequest,
        request: Request,
        service: DocumentService = Depends(_get_service),
    ) -> EnqueuedJobResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        job = service.enqueue_job(
            job_type=JOB_REWRITE_SELECTION,
            payload={
                "document_id": document_id,
                "selection_start": payload.selection_start,
                "selection_end": payload.selection_end,
                "prompt": payload.prompt,
            },
            request_id=str(getattr(request.state, "request_id", "")),
        )
        return EnqueuedJobResponse(job=_to_job_response(job))

    @app.post("/documents/{document_id}/finalize", response_model=DocumentResponse)
    def finalize_document(
        document_id: str,
        service: DocumentService = Depends(_get_service),
    ) -> DocumentResponse:
        try:
            document = service.finalize_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DocumentLockedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _to_document_response(document)

    @app.post("/documents/{document_id}/archive", response_model=DocumentResponse)
    def archive_document(
        document_id: str,
        service: DocumentService = Depends(_get_service),
    ) -> DocumentResponse:
        try:
            document = service.archive_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _to_document_response(document)

    @app.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
    def delete_document(
        document_id: str,
        service: DocumentService = Depends(_get_service),
    ) -> DocumentDeleteResponse:
        try:
            attachments = service.list_attachments(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        removed_files = 0
        cleaned_dirs: set[Path] = set()
        for attachment in attachments:
            path = Path(str(attachment.storage_path))
            if path.is_file():
                try:
                    path.unlink()
                    removed_files += 1
                except OSError:
                    pass
            cleaned_dirs.add(path.parent)

        for directory in cleaned_dirs:
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass

        try:
            service.delete_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return DocumentDeleteResponse(
            deleted=True,
            document_id=document_id,
            removed_attachment_files=removed_files,
        )

    @app.get("/documents/{document_id}/revisions", response_model=list[RevisionResponse])
    def get_document_revisions(
        document_id: str,
        service: DocumentService = Depends(_get_service),
    ) -> list[RevisionResponse]:
        try:
            revisions = service.get_revisions(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [
            RevisionResponse(
                id=revision.id,
                document_id=revision.document_id,
                revision_no=revision.revision_no,
                change_type=revision.change_type,
                prompt=revision.prompt,
                content_snapshot=revision.content_snapshot,
                created_at=revision.created_at,
            )
            for revision in revisions
        ]

    @app.post("/documents/{document_id}/attachments", response_model=AttachmentResponse)
    async def upload_attachment(
        document_id: str,
        file: UploadFile = File(...),
        service: DocumentService = Depends(_get_service),
        attachments_dir: Path = Depends(_get_attachments_dir),
    ) -> AttachmentResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if file.filename is None:
            raise HTTPException(status_code=400, detail="Attachment must include filename")

        raw_bytes = await file.read()
        mime_type = file.content_type or "application/octet-stream"
        try:
            validate_attachment(file.filename, mime_type, len(raw_bytes))
            validate_attachment_content(file.filename, raw_bytes)
        except AttachmentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        document_dir = attachments_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(file.filename)
        stored_name = f"{uuid.uuid4()}_{safe_name}"
        storage_path = document_dir / stored_name
        storage_path.write_bytes(raw_bytes)

        parsed = parse_attachment(file.filename, raw_bytes)
        attachment = service.add_attachment(
            document_id=document_id,
            filename=file.filename,
            mime_type=mime_type,
            size_bytes=len(raw_bytes),
            storage_path=str(storage_path),
            extraction_status=parsed.extraction_status,
            extracted_text=parsed.extracted_text,
            extraction_error=parsed.extraction_error,
        )
        return _to_attachment_response(attachment)

    @app.post("/documents/{document_id}/attachments/async", response_model=EnqueuedJobResponse)
    async def upload_attachment_async(
        document_id: str,
        request: Request,
        file: UploadFile = File(...),
        service: DocumentService = Depends(_get_service),
        attachments_dir: Path = Depends(_get_attachments_dir),
    ) -> EnqueuedJobResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if file.filename is None:
            raise HTTPException(status_code=400, detail="Attachment must include filename")

        raw_bytes = await file.read()
        mime_type = file.content_type or "application/octet-stream"
        try:
            validate_attachment(file.filename, mime_type, len(raw_bytes))
            validate_attachment_content(file.filename, raw_bytes)
        except AttachmentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        document_dir = attachments_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(file.filename)
        stored_name = f"{uuid.uuid4()}_{safe_name}"
        storage_path = document_dir / stored_name
        storage_path.write_bytes(raw_bytes)

        attachment = service.add_attachment_pending(
            document_id=document_id,
            filename=file.filename,
            mime_type=mime_type,
            size_bytes=len(raw_bytes),
            storage_path=str(storage_path),
        )
        job = service.enqueue_job(
            job_type=JOB_PARSE_ATTACHMENT,
            payload={"attachment_id": attachment.id},
            request_id=str(getattr(request.state, "request_id", "")),
        )
        return EnqueuedJobResponse(
            job=_to_job_response(job),
            related_attachment_id=attachment.id,
        )

    @app.post("/research/web", response_model=WebResearchResponse)
    def research_web(
        payload: WebResearchRequest,
    ) -> WebResearchResponse:
        try:
            result = run_web_research(
                query=payload.query,
                region=payload.region,
                max_results=payload.max_results,
                fetch_pages=payload.fetch_pages,
                max_pages=payload.max_pages,
                page_char_limit=payload.page_char_limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Web research failed: {exc}",
            ) from exc

        items = [
            WebResearchItemResponse(
                title=str(item.get("title", "")).strip(),
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("snippet", "")).strip(),
                source=str(item.get("source", "duckduckgo")).strip() or "duckduckgo",
                page_excerpt=str(item.get("page_excerpt", "")).strip(),
                page_error=str(item.get("page_error", "")).strip(),
            )
            for item in result.get("items", [])
            if isinstance(item, dict)
        ]
        return WebResearchResponse(
            query=str(result.get("query", payload.query)).strip(),
            provider=str(result.get("provider", "duckduckgo_instant_answer")).strip(),
            region=str(result.get("region", payload.region)).strip(),
            warning=str(result.get("warning", "")).strip(),
            summary_text=str(result.get("summary_text", "")).strip(),
            items=items,
        )

    @app.post(
        "/documents/{document_id}/attachments/research-web",
        response_model=AttachmentResponse,
    )
    def attach_web_research(
        document_id: str,
        payload: WebResearchRequest,
        service: DocumentService = Depends(_get_service),
        attachments_dir: Path = Depends(_get_attachments_dir),
    ) -> AttachmentResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            result = run_web_research(
                query=payload.query,
                region=payload.region,
                max_results=payload.max_results,
                fetch_pages=payload.fetch_pages,
                max_pages=payload.max_pages,
                page_char_limit=payload.page_char_limit,
            )
            summary_text = str(result.get("summary_text", "")).strip()
            if not summary_text:
                raise RuntimeError("Web research returned empty summary")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Web research attachment failed: {exc}",
            ) from exc

        filename = f"web_research_{_safe_slug(payload.query)}.txt"
        raw_bytes = summary_text.encode("utf-8")
        mime_type = "text/plain"
        try:
            validate_attachment(filename, mime_type, len(raw_bytes))
            validate_attachment_content(filename, raw_bytes)
        except AttachmentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        document_dir = attachments_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4()}_{_safe_filename(filename)}"
        storage_path = document_dir / stored_name
        storage_path.write_bytes(raw_bytes)

        parsed = parse_attachment(filename, raw_bytes)
        attachment = service.add_attachment(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(raw_bytes),
            storage_path=str(storage_path),
            extraction_status=parsed.extraction_status,
            extracted_text=parsed.extracted_text,
            extraction_error=parsed.extraction_error,
        )
        return _to_attachment_response(attachment)

    @app.get("/documents/{document_id}/attachments", response_model=list[AttachmentResponse])
    def list_attachments(
        document_id: str,
        service: DocumentService = Depends(_get_service),
    ) -> list[AttachmentResponse]:
        try:
            attachments = service.list_attachments(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [_to_attachment_response(attachment) for attachment in attachments]

    @app.post(
        "/documents/{document_id}/attachments/import-drive",
        response_model=AttachmentResponse,
    )
    def import_attachment_from_google_drive(
        document_id: str,
        payload: DriveAttachmentImportRequest,
        service: DocumentService = Depends(_get_service),
        attachments_dir: Path = Depends(_get_attachments_dir),
    ) -> AttachmentResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        setting = service.get_integration_setting("google_drive")
        if setting is None:
            raise HTTPException(
                status_code=400,
                detail="Google Drive integration is not configured yet.",
            )
        config = json.loads(setting.config_json)

        try:
            drive_client = _build_google_drive_client_from_config(
                service=service,
                config=config,
            )
            downloaded = drive_client.download_attachment(file_ref=payload.file_ref)
            filename = str(downloaded.get("filename", "")).strip()
            if not filename:
                raise RuntimeError("Google Drive did not provide attachment filename")

            mime_type = str(downloaded.get("mime_type", "")).strip()
            raw = downloaded.get("raw_bytes", b"")
            raw_bytes = bytes(raw if isinstance(raw, (bytes, bytearray)) else b"")

            validate_attachment(filename, mime_type, len(raw_bytes))
            validate_attachment_content(filename, raw_bytes)
        except AttachmentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Google Drive attachment import failed: {exc}",
            ) from exc

        document_dir = attachments_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(filename)
        stored_name = f"{uuid.uuid4()}_{safe_name}"
        storage_path = document_dir / stored_name
        storage_path.write_bytes(raw_bytes)

        parsed = parse_attachment(filename, raw_bytes)
        attachment = service.add_attachment(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=len(raw_bytes),
            storage_path=str(storage_path),
            extraction_status=parsed.extraction_status,
            extracted_text=parsed.extracted_text,
            extraction_error=parsed.extraction_error,
        )
        return _to_attachment_response(attachment)

    @app.post("/integrations/google-drive", response_model=GoogleDriveSettingsResponse)
    def save_google_drive_settings(
        payload: GoogleDriveSettingsRequest,
        service: DocumentService = Depends(_get_service),
    ) -> GoogleDriveSettingsResponse:
        setting = service.get_integration_setting("google_drive")
        if setting is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Google Drive is OAuth-only. "
                    "Run quick connect first: /integrations/google-drive/quick-connect/start"
                ),
            )

        config = json.loads(setting.config_json)
        mode = str(config.get("credential_mode", "")).strip().lower()
        if mode != "oauth_quick_connect":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Legacy credential mode is no longer supported. "
                    "Reconnect with OAuth quick connect."
                ),
            )
        config["folder_name"] = payload.folder_name
        config["folder_id"] = payload.folder_id
        setting = service.upsert_integration_setting(
            provider="google_drive",
            config_json=json.dumps(config),
        )
        return GoogleDriveSettingsResponse(
            provider=setting.provider,
            config=_public_google_drive_config(config),
            updated_at=setting.updated_at,
        )

    @app.get("/integrations/google-drive", response_model=Optional[GoogleDriveSettingsResponse])
    def get_google_drive_settings(
        service: DocumentService = Depends(_get_service),
    ) -> Optional[GoogleDriveSettingsResponse]:
        setting = service.get_integration_setting("google_drive")
        if setting is None:
            return None
        config = json.loads(setting.config_json)
        return GoogleDriveSettingsResponse(
            provider=setting.provider,
            config=_public_google_drive_config(config),
            updated_at=setting.updated_at,
        )

    @app.get(
        "/integrations/google-drive/files",
        response_model=GoogleDriveFileListResponse,
    )
    def list_google_drive_files(
        query: str = "",
        limit: int = 25,
        scope: str = "all",
        service: DocumentService = Depends(_get_service),
    ) -> GoogleDriveFileListResponse:
        normalized_scope = scope.strip().lower()
        if normalized_scope not in {"all", "folder"}:
            raise HTTPException(
                status_code=400,
                detail="scope must be one of: all, folder",
            )
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

        setting = service.get_integration_setting("google_drive")
        if setting is None:
            raise HTTPException(
                status_code=400,
                detail="Google Drive integration is not configured yet.",
            )
        config = json.loads(setting.config_json)

        try:
            client = _build_google_drive_client_from_config(
                service=service,
                config=config,
            )
            rows = client.list_files(
                query=query,
                limit=limit,
                scope=normalized_scope,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Google Drive file listing failed: {exc}",
            ) from exc

        return GoogleDriveFileListResponse(
            files=[
                GoogleDriveFileItemResponse(
                    file_id=str(row.get("file_id", "")).strip(),
                    name=str(row.get("name", "")).strip(),
                    mime_type=str(row.get("mime_type", "")).strip(),
                    modified_time=str(row.get("modified_time", "")).strip(),
                    web_view_link=str(row.get("web_view_link", "")).strip(),
                )
                for row in rows
                if str(row.get("file_id", "")).strip()
            ]
        )

    @app.post(
        "/integrations/google-drive/quick-connect/start",
        response_model=GoogleDriveQuickConnectStartResponse,
    )
    def start_google_drive_quick_connect(
        payload: GoogleDriveQuickConnectStartRequest,
        request: Request,
    ) -> GoogleDriveQuickConnectStartResponse:
        client_id, client_secret, source = _resolve_google_oauth_client_credentials(
            payload.client_id,
            payload.client_secret,
        )
        if not client_id or not client_secret:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Google OAuth client credentials not found. "
                    "Provide client_id/client_secret or set environment variables "
                    "GOOGLE_DRIVE_OAUTH_CLIENT_ID/GOOGLE_DRIVE_OAUTH_CLIENT_SECRET."
                ),
            )

        host = request.headers.get("host", "127.0.0.1:8081")
        default_redirect = f"http://{host}/integrations/google-drive/quick-connect/callback"
        redirect_uri = payload.redirect_uri.strip() or default_redirect
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=ManagerGoogleDriveClient.SCOPES,
        )
        flow.redirect_uri = redirect_uri
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            prompt="consent",
        )

        sessions = app.state.drive_quick_connect_sessions
        sessions[state] = {
            "client_id": client_id,
            "client_secret": client_secret,
            "folder_name": payload.folder_name,
            "folder_id": payload.folder_id,
            "redirect_uri": redirect_uri,
            "created_at": datetime.now().isoformat(),
        }

        return GoogleDriveQuickConnectStartResponse(
            authorization_url=authorization_url,
            state=state,
            redirect_uri=redirect_uri,
            source=source,
        )

    @app.get(
        "/integrations/google-drive/quick-connect/callback",
        include_in_schema=False,
        response_class=HTMLResponse,
    )
    def google_drive_quick_connect_callback(
        code: str = "",
        state: str = "",
        error: str = "",
        service: DocumentService = Depends(_get_service),
    ) -> HTMLResponse:
        if error:
            return HTMLResponse(
                f"<h2>Google Drive connection failed</h2><p>{error}</p>",
                status_code=400,
            )
        if not code or not state:
            return HTMLResponse(
                "<h2>Google Drive connection failed</h2><p>Missing OAuth code/state.</p>",
                status_code=400,
            )

        sessions = app.state.drive_quick_connect_sessions
        session = sessions.pop(state, None)
        if session is None:
            return HTMLResponse(
                "<h2>Google Drive connection failed</h2><p>OAuth session expired. Start quick connect again.</p>",
                status_code=400,
            )

        try:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": session["client_id"],
                        "client_secret": session["client_secret"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                },
                scopes=ManagerGoogleDriveClient.SCOPES,
            )
            flow.redirect_uri = session["redirect_uri"]
            flow.fetch_token(code=code)
            token_json = flow.credentials.to_json()

            config = {
                "credential_mode": "oauth_quick_connect",
                "client_id": session["client_id"],
                "client_secret": session["client_secret"],
                "token_json": token_json,
                "folder_name": session["folder_name"],
                "folder_id": session["folder_id"],
            }
            service.upsert_integration_setting(
                provider="google_drive",
                config_json=json.dumps(config),
            )
        except Exception as exc:
            return HTMLResponse(
                "<h2>Google Drive connection failed</h2>"
                f"<p>{str(exc)}</p>",
                status_code=500,
            )

        return HTMLResponse(
            "<h2>Google Drive connected</h2>"
            "<p>You can return to the app and use Export Drive.</p>",
            status_code=200,
        )

    @app.post(
        "/integrations/google-drive/quick-connect/verify",
        response_model=GoogleDriveQuickConnectVerifyResponse,
    )
    def verify_google_drive_quick_connect(
        service: DocumentService = Depends(_get_service),
    ) -> GoogleDriveQuickConnectVerifyResponse:
        setting = service.get_integration_setting("google_drive")
        if setting is None:
            raise HTTPException(
                status_code=400,
                detail="Google Drive integration is not configured yet.",
            )

        config = json.loads(setting.config_json)
        try:
            client = _build_google_drive_client_from_config(
                service=service,
                config=config,
            )
            verification = client.verify_connection()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Google Drive verification failed: {exc}",
            ) from exc

        return GoogleDriveQuickConnectVerifyResponse(**verification)

    @app.post("/documents/{document_id}/export/docx", response_model=ExportRecordResponse)
    def export_docx(
        document_id: str,
        service: DocumentService = Depends(_get_service),
        exports_dir: Path = Depends(_get_exports_dir),
    ) -> ExportRecordResponse:
        try:
            document = service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            exported_path = export_markdown_like_to_docx(
                title=document.title,
                content=document.current_content,
                output_dir=exports_dir,
                document_id=document.id,
            )
            record = service.add_export_record(
                document_id=document_id,
                export_type=ExportType.DOCX,
                status=ExportStatus.SUCCESS,
                file_path=str(exported_path),
            )
        except Exception as exc:  # pragma: no cover - defensive
            record = service.add_export_record(
                document_id=document_id,
                export_type=ExportType.DOCX,
                status=ExportStatus.FAILED,
                error_message=str(exc),
            )
            raise HTTPException(
                status_code=500,
                detail=f"DOCX export failed: {record.error_message}",
            ) from exc
        return _to_export_record_response(record)

    @app.post("/documents/{document_id}/export/docx/async", response_model=EnqueuedJobResponse)
    def export_docx_async(
        document_id: str,
        request: Request,
        service: DocumentService = Depends(_get_service),
    ) -> EnqueuedJobResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        job = service.enqueue_job(
            job_type=JOB_EXPORT_DOCX,
            payload={"document_id": document_id},
            request_id=str(getattr(request.state, "request_id", "")),
        )
        return EnqueuedJobResponse(job=_to_job_response(job))

    @app.post("/documents/{document_id}/export/drive", response_model=ExportRecordResponse)
    def export_google_drive(
        document_id: str,
        service: DocumentService = Depends(_get_service),
        exports_dir: Path = Depends(_get_exports_dir),
    ) -> ExportRecordResponse:
        try:
            document = service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        setting = service.get_integration_setting("google_drive")
        if setting is None:
            message = (
                "Google Drive settings are missing. Configure /integrations/google-drive first."
            )
            record = service.add_export_record(
                document_id=document_id,
                export_type=ExportType.GOOGLE_DRIVE,
                status=ExportStatus.FAILED,
                error_message=message,
            )
            raise HTTPException(status_code=400, detail=record.error_message)

        config = json.loads(setting.config_json)
        try:
            docx_path = export_markdown_like_to_docx(
                title=document.title,
                content=document.current_content,
                output_dir=exports_dir,
                document_id=document.id,
            )
            drive_document_name = build_drive_export_name(
                title=document.title,
                document_id=document.id,
            )
            drive_client = _build_google_drive_client_from_config(
                service=service,
                config=config,
            )
            result = drive_client.upload_docx_as_google_doc(
                docx_path,
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
                file_path=str(docx_path),
            )
        except Exception as exc:
            error_message = str(exc).strip() or "Unknown Google Drive error"
            record = service.add_export_record(
                document_id=document_id,
                export_type=ExportType.GOOGLE_DRIVE,
                status=ExportStatus.FAILED,
                error_message=error_message,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Google Drive export failed: {record.error_message}",
            ) from exc
        return _to_export_record_response(record)

    @app.post("/documents/{document_id}/export/drive/async", response_model=EnqueuedJobResponse)
    def export_google_drive_async(
        document_id: str,
        request: Request,
        service: DocumentService = Depends(_get_service),
    ) -> EnqueuedJobResponse:
        try:
            service.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        job = service.enqueue_job(
            job_type=JOB_EXPORT_DRIVE,
            payload={"document_id": document_id},
            request_id=str(getattr(request.state, "request_id", "")),
        )
        return EnqueuedJobResponse(job=_to_job_response(job))

    @app.get("/documents/{document_id}/exports", response_model=list[ExportRecordResponse])
    def list_exports(
        document_id: str,
        service: DocumentService = Depends(_get_service),
    ) -> list[ExportRecordResponse]:
        try:
            records = service.list_export_records(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [_to_export_record_response(record) for record in records]

    @app.post("/jobs/archive-nightly", response_model=EnqueuedJobResponse)
    def enqueue_nightly_archive_job(
        request: Request,
        older_than_days: int = 30,
        service: DocumentService = Depends(_get_service),
    ) -> EnqueuedJobResponse:
        job = service.enqueue_job(
            job_type=JOB_ARCHIVE_SCAN,
            payload={"older_than_days": older_than_days},
            request_id=str(getattr(request.state, "request_id", "")),
        )
        return EnqueuedJobResponse(job=_to_job_response(job))

    @app.get("/jobs", response_model=list[JobResponse])
    def list_jobs(
        limit: int = 100,
        service: DocumentService = Depends(_get_service),
    ) -> list[JobResponse]:
        jobs = service.list_jobs(limit=limit)
        return [_to_job_response(job) for job in jobs]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(
        job_id: int,
        service: DocumentService = Depends(_get_service),
    ) -> JobResponse:
        try:
            job = service.get_job(job_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _to_job_response(job)

    @app.get("/metrics/dashboard")
    def metrics_dashboard(
        service: DocumentService = Depends(_get_service),
        metrics: RuntimeMetrics = Depends(_get_metrics),
    ) -> dict:
        return metrics.snapshot(service)

    @app.post("/audit/export", response_model=AuditExportResponse)
    def export_audit_log(
        document_id: Optional[str] = None,
        service: DocumentService = Depends(_get_service),
        audit_dir: Path = Depends(_get_audit_dir),
    ) -> AuditExportResponse:
        try:
            file_path, event_count = export_audit_log_jsonl(
                service=service,
                output_dir=audit_dir,
                document_id=document_id,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return AuditExportResponse(file_path=str(file_path), event_count=event_count)

    @app.post("/archive/run", response_model=ArchiveJobResponse)
    def run_archive_job(
        older_than_days: int = 30,
        service: DocumentService = Depends(_get_service),
    ) -> ArchiveJobResponse:
        archived_count = service.run_archive_job(older_than_days=older_than_days)
        return ArchiveJobResponse(archived_count=archived_count)

    return app


def _to_document_planning_session_response(session: dict) -> DocumentPlanningSessionResponse:
    brief = dict(session.get("brief", {}))
    messages_raw = session.get("messages", [])
    messages = [
        DocumentPlanningMessageResponse(
            role=str(item.get("role", "")).strip() or "assistant",
            content=str(item.get("content", "")).strip(),
            created_at=str(item.get("created_at", "")).strip(),
        )
        for item in messages_raw
        if isinstance(item, dict)
    ]
    status = str(session.get("status", "")).strip()
    return DocumentPlanningSessionResponse(
        id=str(session.get("id", "")).strip(),
        status=status,
        step_index=int(session.get("step_index", 0)),
        ready_to_create=status in {PLANNING_STATUS_APPROVED, PLANNING_STATUS_DOCUMENT_CREATED},
        created_document_id=str(session.get("created_document_id", "")).strip(),
        suggested_points=[
            str(item).strip()
            for item in session.get("suggested_points", [])
            if str(item).strip()
        ],
        approved_points=[
            str(item).strip()
            for item in session.get("approved_points", [])
            if str(item).strip()
        ],
        brief=DocumentPlanningBriefResponse(
            title=str(brief.get("title", "")).strip(),
            doc_type=str(brief.get("doc_type", "")).strip(),
            target_audience=str(brief.get("target_audience", "")).strip(),
            language=str(brief.get("language", "")).strip(),
            objective=str(brief.get("objective", "")).strip(),
            tone=str(brief.get("tone", "")).strip(),
            constraints=str(brief.get("constraints", "")).strip(),
            emphasis=str(brief.get("emphasis", "")).strip(),
            decisions_needed=str(brief.get("decisions_needed", "")).strip(),
            must_include=str(brief.get("must_include", "")).strip(),
        ),
        messages=messages,
        created_at=str(session.get("created_at", "")).strip(),
        updated_at=str(session.get("updated_at", "")).strip(),
    )


def _to_document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        title=document.title,
        doc_type=document.doc_type,
        target_audience=document.target_audience,
        language=document.language,
        objective=document.objective,
        tone=document.tone,
        constraints=document.constraints,
        status=document.status,
        current_content=document.current_content,
        last_opened_at=document.last_opened_at,
        finalized_at=document.finalized_at,
        archived_at=document.archived_at,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _to_attachment_response(attachment: Attachment) -> AttachmentResponse:
    return AttachmentResponse(
        id=attachment.id,
        document_id=attachment.document_id,
        filename=attachment.filename,
        mime_type=attachment.mime_type,
        size_bytes=attachment.size_bytes,
        storage_path=attachment.storage_path,
        extraction_status=attachment.extraction_status,
        extracted_text=attachment.extracted_text,
        extraction_error=attachment.extraction_error,
        created_at=attachment.created_at,
    )


def _safe_filename(filename: str) -> str:
    base = Path(filename).name.strip().replace(" ", "_")
    return base or "attachment.bin"


def _safe_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        return "query"
    return normalized[:64]


def _build_attachment_summary(attachments: list[Attachment]) -> str:
    max_total_chars = _resolve_summary_chars_env(
        key="MANAGER_DOCUMENT_AGENT_ATTACHMENT_SUMMARY_MAX_CHARS",
        default=90000,
        min_value=8000,
        max_value=300000,
    )
    attachments_with_text = [
        attachment
        for attachment in attachments
        if (attachment.extracted_text or "").strip()
    ]
    if not attachments_with_text:
        return ""

    per_attachment_cap = min(
        18000,
        max(2500, max_total_chars // max(1, len(attachments_with_text))),
    )
    lines: list[str] = []
    for attachment in attachments_with_text:
        extracted = (attachment.extracted_text or "").strip()
        lines.append(f"- {attachment.filename}")
        context_lines = _build_attachment_context_lines(
            extracted,
            per_attachment_cap=per_attachment_cap,
        )
        lines.extend([f"  - {line}" for line in context_lines])
        lines = _trim_summary_lines(lines, max_chars=max_total_chars)
        if _summary_char_len(lines) >= max_total_chars:
            break
    return "\n".join(lines).strip()


def _build_attachment_context_lines(text: str, *, per_attachment_cap: int) -> list[str]:
    if not text.strip():
        return []

    selected: list[str] = []
    selected.extend(_extract_high_signal_attachment_facts(text, limit=24))
    selected.extend(_extract_followup_list_lines(text, anchors=selected, limit=10))
    selected.extend(_extract_context_windows(text, per_attachment_cap=per_attachment_cap))

    normalized = [_normalize_attachment_line(line) for line in selected]
    unique = []
    seen: set[str] = set()
    for line in normalized:
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    if not unique:
        excerpt = _single_line_excerpt(text, max_chars=min(1600, per_attachment_cap))
        return [excerpt] if excerpt else []
    return _trim_context_lines(unique, max_chars=per_attachment_cap)


def _extract_high_signal_attachment_facts(text: str, *, limit: int) -> list[str]:
    raw_lines = [_normalize_attachment_line(line) for line in text.splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return []

    selected: list[str] = []
    seen: set[str] = set()

    def _push(line: str) -> None:
        if len(selected) >= limit:
            return
        if line in seen:
            return
        seen.add(line)
        selected.append(line)

    def _push_many(predicate, *, cap: int) -> None:
        pushed = 0
        for line in lines:
            if pushed >= cap or len(selected) >= limit:
                break
            if not predicate(line):
                continue
            before = len(selected)
            _push(line)
            if len(selected) > before:
                pushed += 1

    _push_many(_is_metric_line, cap=5)
    _push_many(_is_strategy_program_line, cap=4)
    _push_many(_is_goal_or_decision_line, cap=3)
    for line in lines:
        if len(selected) >= limit:
            break
        if _is_high_signal_line(line):
            _push(line)

    return selected[:limit]


def _extract_followup_list_lines(text: str, *, anchors: list[str], limit: int) -> list[str]:
    raw_lines = [_normalize_attachment_line(line) for line in text.splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return []
    selected: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        normalized_anchor = _normalize_attachment_line(anchor)
        if not normalized_anchor or not normalized_anchor.endswith(":"):
            continue
        try:
            start_index = lines.index(normalized_anchor)
        except ValueError:
            continue
        for candidate in lines[start_index + 1: start_index + 10]:
            if len(selected) >= limit:
                return selected
            if not _looks_like_list_or_program_line(candidate):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            selected.append(candidate)
    return selected


def _extract_context_windows(text: str, *, per_attachment_cap: int) -> list[str]:
    normalized = _normalize_whitespace_for_summary(text)
    if not normalized:
        return []
    if len(normalized) <= per_attachment_cap:
        return [normalized]

    window = max(600, min(2600, per_attachment_cap // 4))
    start = _single_line_excerpt(normalized[:window], max_chars=window)
    mid_start = max(0, (len(normalized) // 2) - (window // 2))
    middle = _single_line_excerpt(normalized[mid_start:mid_start + window], max_chars=window)
    end = _single_line_excerpt(normalized[-window:], max_chars=window)
    windows = []
    if start:
        windows.append(f"Context (start): {start}")
    if middle:
        windows.append(f"Context (middle): {middle}")
    if end:
        windows.append(f"Context (end): {end}")
    return windows


def _trim_context_lines(lines: list[str], *, max_chars: int) -> list[str]:
    output: list[str] = []
    running = 0
    for line in lines:
        if not line:
            continue
        proposed = running + len(line) + 1
        if proposed > max_chars:
            break
        output.append(line)
        running = proposed
    return output


def _trim_summary_lines(lines: list[str], *, max_chars: int) -> list[str]:
    output: list[str] = []
    running = 0
    for line in lines:
        proposed = running + len(line) + 1
        if proposed > max_chars:
            break
        output.append(line)
        running = proposed
    return output


def _summary_char_len(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def _resolve_summary_chars_env(
    *,
    key: str,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return min(max(parsed, min_value), max_value)


def _normalize_whitespace_for_summary(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_attachment_line(line: str) -> str:
    value = line.replace("\t", " | ").replace("•", "-").strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip("- ").strip()
    if len(value) < 18:
        return ""
    return _truncate_preserving_words(value, max_chars=900)


def _single_line_excerpt(text: str, *, max_chars: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return _truncate_preserving_words(collapsed, max_chars=max_chars)


def _truncate_preserving_words(text: str, *, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    truncated = value[:max_chars].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0].rstrip()
    return truncated


def _is_strategy_program_line(line: str) -> bool:
    if re.match(r"^(?:[IVX]{1,6}[.)-]\s+)", line):
        return True
    if re.search(r"\b(?:program|pillar|workstream|roadmap)\b", line, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(?:technical|authority|brand|geo|automation|marketplace)\b", line, flags=re.IGNORECASE):
        return True
    return False


def _is_metric_line(line: str) -> bool:
    if "%" in line:
        return True
    if not re.search(r"\d", line):
        return False
    metric_markers = (
        "gmv",
        "kpi",
        "crvisit",
        "q1",
        "q2",
        "q3",
        "q4",
    )
    lowered = line.lower()
    return any(marker in lowered for marker in metric_markers)


def _is_goal_or_decision_line(line: str) -> bool:
    return bool(
        re.search(
            r"\b(?:goal|target|objective|priority|risk|mitigation|decision|outcome)\b",
            line,
            flags=re.IGNORECASE,
        )
    )


def _is_high_signal_line(line: str) -> bool:
    if line.count(" | ") >= 2:
        return True
    return bool(
        re.search(
            r"\b(?:allegro|seo|geo|discovery|organic|ai|management)\b",
            line,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_list_or_program_line(line: str) -> bool:
    if re.match(r"^(?:[IVX]{1,6}[.)-]\s+)", line):
        return True
    if re.match(r"^\d{1,2}[.)-]\s+", line):
        return True
    if line.count(" | ") >= 2:
        return True
    return bool(
        re.search(
            r"\b(?:program|pillar|workstream|initiative|plan|owner|timeline|status)\b",
            line,
            flags=re.IGNORECASE,
        )
    )


def _to_export_record_response(record: ExportRecord) -> ExportRecordResponse:
    return ExportRecordResponse(
        id=record.id,
        document_id=record.document_id,
        export_type=record.export_type,
        status=record.status,
        external_url=record.external_url,
        file_path=record.file_path,
        error_message=record.error_message,
        created_at=record.created_at,
    )


def _to_job_response(job: JobRecord) -> JobResponse:
    return JobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        payload_json=job.payload_json,
        result_json=job.result_json,
        error_message=job.error_message,
        failure_class=job.failure_class,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        next_run_at=job.next_run_at,
        request_id=job.request_id,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _to_auth_user_response(user: AuthUser) -> AuthUserResponse:
    return AuthUserResponse(
        subject=user.subject,
        email=user.email,
        roles=user.roles,
        auth_mode=user.auth_mode,
    )


def _requires_auth(path: str) -> bool:
    public_paths = {
        "/",
        "/openapi.json",
        "/docs",
        "/redoc",
        "/auth/login/mock",
        "/auth/login/oauth",
        "/integrations/google-drive/quick-connect/callback",
    }
    if path in public_paths:
        return False
    return True


def _required_roles_for_request(*, method: str, path: str) -> list[str]:
    if path.startswith("/metrics") or path.startswith("/integrations"):
        return ["admin"]
    if path.startswith("/audit") or path.startswith("/jobs"):
        return ["admin"]
    if method.upper() == "GET":
        return ["author", "reviewer", "admin"]
    return ["author", "admin"]


def _build_google_drive_client_from_config(
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
        "Please reconnect via quick connect."
    )


def _resolve_google_oauth_client_credentials(
    client_id_override: str,
    client_secret_override: str,
) -> tuple[str, str, str]:
    client_id = client_id_override.strip()
    client_secret = client_secret_override.strip()
    if client_id and client_secret:
        return client_id, client_secret, "request"

    env_client_id = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_ID", "").strip()
    env_client_secret = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET", "").strip()
    if env_client_id and env_client_secret:
        return env_client_id, env_client_secret, "env"

    return "", "", "missing"

def _public_google_drive_config(config: dict) -> dict:
    mode = str(config.get("credential_mode", "")).strip().lower()
    if mode == "oauth_quick_connect":
        return {
            "credential_mode": "oauth_quick_connect",
            "folder_name": config.get("folder_name", "Manager Documents"),
            "folder_id": config.get("folder_id", ""),
            "connected": bool(config.get("token_json")),
        }
    return {
        "credential_mode": "legacy_file_paths_removed",
        "folder_name": config.get("folder_name", "Manager Documents"),
        "folder_id": config.get("folder_id", ""),
        "connected": False,
    }
