# Manager Document Agent - Security Baseline

## Secret handling
- Never store API secrets or OAuth tokens in repository files.
- Keep credentials in environment variables or secret manager.
- Allowed persisted integration data stores only paths/references, not raw secret payloads.
- Existing ignore rules protect `.env`, `client_secret*.json`, `.google_drive_token.json`, and `secret.json`.

## File upload hardening
- Accept only allowlisted extensions: `.docx`, `.xlsx`, `.csv`, `.tsv`, `.txt`, `.pdf`.
- Enforce size limits and MIME validation.
- Enforce binary signature checks for PDF/Office files.
- Reject binary null bytes for text-based formats.

## API and runtime safety
- Every request receives `X-Request-ID` for traceability.
- Job queue tracks attempts, failure class, and retry/backoff behavior.
- Structured logs include request/job identifiers.
- Audit export endpoint provides trace records for revisions/attachments/exports/jobs.

## Operational controls
- Run nightly archival via `weekly-manager-document-agent-archive-nightly`.
- Monitor `/metrics/dashboard` for latency/failure/queue depth.
- Keep DB and output directories access-limited to service account.
