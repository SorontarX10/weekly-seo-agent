# Manager Document Agent - Cloud Release Checklist

## 1. Security and access
- [ ] OAuth/OIDC provider configured (issuer, client id, redirect URIs).
- [ ] Role model enabled (`author`, `reviewer`, `admin`) and validated on protected routes.
- [ ] Secrets moved to secret manager (no credentials in repo or image).
- [ ] Credential rotation and incident rollback procedure documented.

## 2. Data and storage
- [ ] Managed PostgreSQL (or approved DB) provisioned with backups.
- [ ] Object storage bucket configured for attachments/exports/audit logs.
- [ ] Retention policy applied for attachments, prompts, and audit artifacts.
- [ ] Disaster recovery restore test executed at least once.

## 3. Runtime and jobs
- [ ] API service deployed with health checks and autoscaling policy.
- [ ] Job worker deployed separately with queue depth alerts.
- [ ] Nightly archive job scheduled (`weekly-manager-document-agent-archive-nightly`).
- [ ] Retry/backoff policy validated against failure scenarios.

## 4. Observability
- [ ] Structured logs collected centrally (request_id + job_id preserved).
- [ ] Metrics dashboard wired (latency, failure rate, queue depth).
- [ ] Alerting configured for API 5xx, queue saturation, and job failures.
- [ ] Audit export path enabled and access-controlled.

## 5. Quality gates
- [ ] Unit/integration/e2e test suite green in CI.
- [ ] Smoke test run in staging: create -> edit -> finalize -> archive.
- [ ] Export verification passed for DOCX and Google Drive.
- [ ] Rollback procedure for failed deployment tested.

## 6. Go-live controls
- [ ] Stakeholder sign-off completed.
- [ ] Change window approved.
- [ ] Post-release monitoring owner assigned.
- [ ] 48h hypercare plan confirmed.
