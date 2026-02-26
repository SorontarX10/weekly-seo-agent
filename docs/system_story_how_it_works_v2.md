# System Story: How It Works (V2)

## Scope and Purpose
This document explains the current production workflow of the Weekly SEO Intelligence system.
It is written as an operational guide: after reading it, you should understand what runs, when it runs, which integrations are used, and how decisions are made before publishing.

## What This System Produces
The system generates weekly SEO intelligence reports per market (PL, CZ, SK, HU).
Each report combines:
- Search Console performance changes (WoW, MoM, YoY context)
- External and business context (campaigns, trends, weather, events, status sheets)
- LLM-based reasoning and managerial narrative
- Quality evaluation before publication

Primary output format:
- Local DOCX report per market
- Optional publication to Google Drive as a Google Doc

## Core Runtime Components
The current implementation is organized into these runtime modules:

### Orchestrator
File: `weekly_seo_agent/weekly_reporting_agent/main.py`
Responsibilities:
- loads runtime config from `.env`
- executes startup preflight checks (GSC, Drive, sheets, tabs)
- builds per-country configuration
- runs single-country or multi-country execution
- applies quality gate rules
- triggers Google Drive publication
- writes observability telemetry

### Workflow Engine
File: `weekly_seo_agent/weekly_reporting_agent/workflow.py`
Responsibilities:
- computes date windows
- fetches and normalizes data from all sources
- runs analysis layers and hypothesis generation
- builds enriched report context
- produces final text for rendering

### Report Renderer
File: `weekly_seo_agent/weekly_reporting_agent/reporting.py`
Responsibilities:
- builds manager-facing narrative sections
- formats KPI and driver sections
- deduplicates and canonicalizes timeline items
- renders final DOCX

### Integrations Layer
Main clients:
- `clients/gsc_client.py` (Google Search Console)
- `clients/google_drive_client.py` (upload DOCX -> Google Doc)
- `clients/continuity_client.py` (Google Drive/Sheets context: status, trends, trade plan, continuity)
- `clients/external_signals.py` (weather, holidays, news, status/blog feeds)

## End-to-End Flow (Current Version)
1. Preflight starts before each run.
- verifies explicit GSC mapping for selected countries
- probes GSC data access
- verifies Drive publish target
- checks required Sheets integrations (status/trends/trade plan)

2. Country run starts.
- computes analysis windows (current, previous, YoY-aligned)
- fetches GSC totals and segmented datasets (query/page/device/search appearance/date)

3. Context collectors run.
- external signals: weather/news/holidays/platform signals
- additional business context: status logs, product trend sheets, trade plan, market events, updates timelines

4. Ingestion robustness layer is applied.
- source freshness registry per feed
- GSC daily completeness and missing-day masks
- explicit metric window availability map (current/previous/YoY per source)
- standardized ingestion schema for downstream logic
- country normalization context (timezone/date/number conventions)
- missing-data fallback policy by source
- sanitized ingestion snapshot with retention policy

5. Reasoning and narrative generation.
- data signals are transformed into hypotheses with confidence
- contradictions and supporting evidence are reconciled in narrative sections
- final report text is assembled for non-technical business reading

6. Evaluation and gate.
- quality metrics are calculated (readability, structure, evidence linkage, duplication, jargon load)
- score and pass/fail are persisted in telemetry
- depending on gate policy, publication can be blocked

7. Output and publication.
- report is written locally as DOCX
- if enabled and allowed by gate policy, DOCX is uploaded and converted to Google Doc in target Drive folder

## Data Sources Used in Current System
Performance core:
- Google Search Console (primary KPI source)

Business/context sources:
- Google Sheets status log
- Google Sheets product trend trackers
- Google Sheets trade plan (including YoY sheet when available)
- market-event API context
- Google status/blog and SEO/GEO publication timelines
- weather context (history + forecast)

Publication/storage:
- local file system (DOCX + telemetry + ingestion snapshots)
- Google Drive (Google Docs publication)

## Quality and Governance Rules
The current system includes:
- startup preflight matrix with blocker/warning severity
- quality scoring before publication
- optional strict LLM profile in weekly reporter runtime
- source reliability segmentation in context handling
- fallback behavior for degraded sources
- observability logs per run

## Operational Controls
Useful runtime controls include:
- run date override (`--run-date`)
- preflight-only mode (`--preflight-only`)
- strict LLM profile toggle (`--strict-llm-profile` / `--no-strict-llm-profile`)
- source enable/disable overrides (`--enable-source`, `--disable-source`)

Country scope is controlled by env for each run, e.g.:
- `REPORT_COUNTRIES=PL`
- `REPORT_COUNTRY_CODE=PL`

## Telemetry and Debugging
Current run artifacts include:
- DOCX report files in output folder
- observability JSONL (`_telemetry/*_weekly_reporting_observability.jsonl`)
- sanitized ingestion snapshots (`_telemetry/raw_sources/*_ingestion_snapshot.json`)

This allows debugging:
- what data was available
- what was stale or missing
- why report quality passed or failed

## Current Known Constraints
- quality gate can block publication even when report file is generated locally
- Drive publication depends on valid OAuth/service-account access to target folder
- source coverage can vary by market and week; fallback logic reduces but does not remove this variability

## Version Note
This is Version 2 of the system story document and reflects the currently deployed workflow behavior, integrations, and guardrails.
