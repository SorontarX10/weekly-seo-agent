# Manager Document Agent - Implementation Backlog

Rule of work:
- Keep this file as a live queue.
- After implementing a task, delete that task line from this file.
- Keep numbering as `NNN` to preserve stable task IDs while deleting lines.

Status:
- All currently defined tasks completed on 2026-03-02.
- 2026-03-03: Improved web research reliability and verification
  (DuckDuckGo HTML SERP fallback when Instant API has sparse results + explicit Playwright fetch stats in summary).
- 2026-03-03: Refreshed UI and session flow
  (planning chat highlighted as default start with auto-session init + persistent Black Mode toggle).
- 2026-03-03: Added LLM response pagination + merge for long generations
  (automatic continuation when model hits length limit; overlap-aware stitching to reduce cut content).
- 2026-03-03: Google Drive switched to OAuth-only integration mode
  (legacy credentials-file flow removed from API/UI path; no manual authorization alert in UI).
- 2026-03-02: Google Drive integrated into "Create New" flow as alternative attachment source
  (OAuth quick-connect fields + file URL/ID import path).
- 2026-03-02: Fixed Google OAuth quick-connect scope mismatch
  (`include_granted_scopes` removed from authorization request).
- 2026-03-02: Improved outline quality from attachments
  (high-signal fact extraction + source-facts section + stricter anti-generic outline prompt).
- 2026-03-02: Added internet research integration
  (DuckDuckGo Instant Answer API + Playwright stealth fetch + UI actions + attachment import).
- 2026-03-02: Added Google Drive file picker UI
  (browse/search Drive files from connected account, then add by click instead of manual URL/ID).
- 2026-03-02: Improved Google Drive export UX
  (explicit success signal in UI + readable/stable Drive document naming).
- 2026-03-02: Added Executive Playbook grounding for LLM document writing
  (outline/full/selection prompts now include playbook rules, configurable via env path/limit).
- 2026-03-02: Improved create-document validation UX
  (client-side required field checks + readable 422 validation messages with field names).
- 2026-03-02: Improved long-content handling and markdown cleanup
  (higher Manager Agent LLM/context limits, removal of inline `**...**` artifacts, cleaner DOCX export rendering, no dangling `...` facts).
- 2026-03-02: Added conversational planning gate before document creation
  (chat-driven requirement gathering, suggested points approval, and create-from-approved-plan flow).
- 2026-03-02: Added visible loading/progress state for "Create Document"
  (button lock + phase-by-phase status for draft creation and attachment import steps).
- 2026-03-02: Fixed SQLite concurrency issue in document open flow
  (thread-safe serialized DB access in DocumentService to prevent intermittent `sqlite3.InterfaceError` 500s).
- 2026-03-02: Improved context auto-loading for long source materials
  (multi-pack prompt context: start/middle/end + larger attachment budget + follow-up extraction for list sections after `...:` lines).
- 2026-03-02: Added Archive/Delete controls on Documents list
  (new API endpoints + per-row actions in UI with confirmations and live list refresh).

## A) Product baseline


## B) Backend core

## C) Attachments and parsing


## D) AI editing workflow


## E) Frontend UX


## F) Export and integrations


## G) Automation, quality, observability


## H) Security and productionization

## I) Testing and release
