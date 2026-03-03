# Manager Document Agent - POC Architecture and Production Path (2026-03-02)

## 1. Goal

Build an AI-assisted document writing agent for managers, starting as a fast POC but with architecture that can be stabilized and deployed in cloud without redesign.

Primary business outcomes:
- Faster creation of manager-facing documents.
- Better consistency and quality of structure/narrative.
- Controlled collaboration between AI edits and manual edits.
- Traceable lifecycle: draft -> review -> finalized -> archived.

## 2. Scope from requirements

Supported user capabilities:
1. Create a new document with explicit purpose and audience.
2. Attach files and instructions during the whole drafting phase.
3. Generate document outline (skeleton) from metadata + attachments + prompt.
4. Edit draft in three modes:
   - AI full-document edit via prompt,
   - manual direct edit,
   - AI partial edit of selected fragment.
5. Track status (`IN_PROGRESS`, `FINALIZED`, `ARCHIVED`).
6. Open historical documents from list views.
7. Access a dedicated archive for documents older than 30 days.
8. Export to `.docx` or publish to Google Drive (if user configured API access).
9. Lock document after finalization (read-only).

Future capability (already accounted for in architecture):
- AuthN/AuthZ via OAuth (currently app can run without login in POC mode).

## 3. Recommended additional requirements

To avoid future rework, include these now (even if partially implemented in POC):
1. Versioning:
- Keep revision history for each AI/manual change.
- Allow rollback to previous revision.
2. Audit trail:
- Store who/what changed content (`manual`, `ai_full`, `ai_partial`).
- Keep timestamps for legal/process traceability.
3. Data provenance:
- Link each AI-generated section to source attachments/instructions used.
4. Guardrails:
- LLM output checks: empty sections, hallucination markers, invalid claims without source.
5. Background processing:
- Long operations (file parsing, AI generation, export, Drive publish) should run as jobs.
6. Retention policy:
- Configurable retention for attachments and intermediate AI prompts/responses.

## 4. File support

Requested formats:
- `.docx`, `.xlsx`, `.csv`, `.tsv`, `.txt`, `.pdf`.

Recommended additions:
- `.pptx` (management often provides context in slides).
- `.md` (technical teams frequently share notes in markdown).
- Optional later: image OCR (`.png`, `.jpg`) for scanned notes/screenshots.

POC parsing policy:
- Parse text/tables from all supported formats.
- Store raw file + extracted text representation.
- Mark extraction quality (`OK`, `PARTIAL`, `FAILED`).

## 5. UI information architecture

Tabs/views aligned to your requirement:
1. `Create New`
- Wizard: audience, document type, tone, language, objective, key constraints.
- Attachment uploader + instruction input.
- Action: `Generate Outline`.

2. `Documents`
- List of active and completed docs.
- Filters: status, type, owner, updated date.
- Quick-open and status badge.

3. `Last Opened`
- Fast resume of last active document.

4. `Archive`
- Documents older than 30 days (or forcibly archived).
- Read-only by default.

Editor view (inside opened document):
- Main rich-text editor.
- Prompt panel for AI edit requests.
- Selection-based "rewrite selected fragment" action.
- Attachments panel (add/remove while `IN_PROGRESS`).
- Revision timeline.
- Actions: `Save`, `Export DOCX`, `Publish to Drive`, `Finalize`.

## 6. Document lifecycle and rules

State machine:
- `IN_PROGRESS` -> editable, attachments can be added.
- `FINALIZED` -> read-only, no further content edits.
- `ARCHIVED` -> read-only + listed in archive section.

Business rules:
1. Finalization lock:
- `FINALIZED` disables manual and AI edits.
2. Auto-archive:
- If `FINALIZED` and `finalized_at + 30 days < now`, move to `ARCHIVED`.
3. Non-finalized old docs:
- Stay in `IN_PROGRESS` unless explicit policy says otherwise.
4. Export availability:
- Allowed in all states (read-only export for finalized/archived).

## 7. Proposed architecture (POC that scales)

### 7.1 Frontend
- React + TypeScript + Next.js (App Router) or Vite SPA.
- Rich-text editor with selection API (e.g., TipTap/ProseMirror).
- WebSocket or polling for async job progress.

### 7.2 Backend API
- Python FastAPI (fits current repo stack) or Node.js NestJS.
- Layered modules:
  - `document_service` (CRUD, state transitions, metadata),
  - `attachment_service` (storage, parsing dispatch),
  - `ai_service` (outline generation + edits),
  - `export_service` (`.docx`, Google Drive),
  - `archive_service` (retention and 30-day archival).

### 7.3 Workers / background jobs
- Queue: Celery/RQ (Python) or BullMQ (Node).
- Job types:
  - parse attachment,
  - generate outline,
  - ai rewrite (full/partial),
  - export docx,
  - publish drive,
  - nightly archival scan.

### 7.4 Storage
- PostgreSQL: metadata, document content snapshots, states, audit trail.
- Object storage (S3/GCS/MinIO): raw attachments + generated exports.
- Optional Redis: queue backend + short-lived caches.

### 7.5 Integrations
- LLM provider abstraction:
  - `generate_outline(context)`
  - `rewrite_full(document, prompt)`
  - `rewrite_fragment(fragment, prompt, neighbors)`
- Google Drive adapter:
  - OAuth credentials per user/workspace,
  - folder resolution,
  - upload and link persistence.

## 8. Data model (minimum)

### 8.1 `documents`
- `id`
- `title`
- `doc_type` (e.g., `TEST_SUMMARY`, `MANAGEMENT_BRIEF`, `TECH_DOC`)
- `target_audience`
- `language`
- `status` (`IN_PROGRESS`, `FINALIZED`, `ARCHIVED`)
- `current_content` (rich-text JSON or markdown)
- `last_opened_at`
- `finalized_at`
- `archived_at`
- `created_at`, `updated_at`

### 8.2 `document_revisions`
- `id`, `document_id`
- `revision_no`
- `change_type` (`MANUAL`, `AI_FULL`, `AI_PARTIAL`, `SYSTEM`)
- `prompt` (nullable)
- `content_snapshot`
- `created_at`

### 8.3 `attachments`
- `id`, `document_id`
- `filename`, `mime_type`, `size_bytes`
- `storage_path`
- `extraction_status` (`PENDING`, `OK`, `PARTIAL`, `FAILED`)
- `extracted_text`
- `created_at`

### 8.4 `exports`
- `id`, `document_id`
- `export_type` (`DOCX`, `GOOGLE_DRIVE`)
- `status`
- `external_url` (for Drive)
- `created_at`

## 9. API surface (starter)

- `POST /documents` create draft + metadata.
- `GET /documents` list + filters.
- `GET /documents/{id}` details.
- `PATCH /documents/{id}` metadata/content update (if `IN_PROGRESS`).
- `POST /documents/{id}/attachments` upload file.
- `POST /documents/{id}/outline` generate outline from current context.
- `POST /documents/{id}/rewrite` AI rewrite full document.
- `POST /documents/{id}/rewrite-selection` AI rewrite selected fragment.
- `POST /documents/{id}/finalize` lock document.
- `POST /documents/{id}/export/docx` generate/download `.docx`.
- `POST /documents/{id}/export/drive` publish to Google Drive.
- `GET /documents/last-opened` quick access.
- `GET /documents/archive` list archived docs.

## 10. POC -> production rollout plan

### Phase 1: POC (2-3 weeks)
- No login.
- Single workspace.
- Full basic flow from create -> outline -> edit -> export -> finalize.
- File parsing for required formats.
- Basic document list + archive tab.

Exit criteria:
- End-to-end happy path works reliably for at least 20 test documents.

### Phase 2: Stabilization (2-4 weeks)
- Add revision history and audit trail.
- Add async jobs + retries + observability.
- Improve parser robustness and error UX.
- Add integration tests and load smoke tests.

Exit criteria:
- Error budget and recovery behavior defined and tested.

### Phase 3: Cloud-ready productionization
- OAuth login and roles.
- Multi-tenant separation.
- Managed DB + object storage + queue.
- CI/CD, backups, monitoring, alerts, SLOs.
- Security hardening and secret management.

Exit criteria:
- Ready for controlled rollout to real manager teams.

## 11. Security baseline

POC minimum:
- Secrets in env/secret manager, never in repo.
- Input file scanning + MIME/size validation.
- Explicit allowed extension list.

Production target:
- OAuth2/OIDC login.
- RBAC (`author`, `reviewer`, `admin`).
- Encryption at rest + in transit.
- Full audit logs and retention policy.

## 12. Key risks and mitigations

1. Risk: low-quality extraction from complex PDFs/XLSX.
- Mitigation: extraction status + preview + manual correction workflow.

2. Risk: AI edits overwrite critical content.
- Mitigation: revision snapshots + diff preview + one-click rollback.

3. Risk: long generation latency.
- Mitigation: async jobs + partial progress + cancel/retry actions.

4. Risk: Drive integration failures.
- Mitigation: provider adapter with retries and explicit error codes in UI.

## 13. Suggested implementation order

1. Core document model + states + CRUD.
2. Editor with manual save.
3. AI outline generation.
4. Attachment ingestion/parsing.
5. AI rewrite full + selection rewrite.
6. Export DOCX.
7. Google Drive publish.
8. Finalize lock + auto archive worker.
9. Revision timeline and rollback.
10. OAuth + RBAC.

