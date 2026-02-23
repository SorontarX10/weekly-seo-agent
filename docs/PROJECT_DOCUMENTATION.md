# Project Documentation: Weekly SEO Intelligence Agent

## 1. Purpose

This project is a production-oriented analytical agent that builds weekly SEO intelligence reports for multiple Allegro markets (`PL`, `CZ`, `SK`, `HU`).

Main goals:

1. Detect what changed in organic performance (`WoW` and aligned `52W`).
2. Explain likely causes using internal + external evidence.
3. Separate demand shifts from technical SEO signals.
4. Produce manager-readable narrative and analyst-ready appendix.
5. Publish reports locally (`.docx`) and optionally to Google Drive (Google Docs).

The same architecture can be reused to build a general traffic analytics agent (not only SEO).

Companion technical reference:

- `docs/SOURCES_AND_INTEGRATIONS.md` - full list of data sources, API links, auth methods, libraries, and integration pattern.

## 2. High-Level Architecture

Codebase roots:

- `weekly_seo_agent/main.py`: batch entrypoint, per-country parallel execution.
- `weekly_seo_agent/workflow.py`: LangGraph orchestration and LLM pipeline.
- `weekly_seo_agent/reporting.py`: markdown-to-docx rendering and formatting rules.
- `weekly_seo_agent/additional_context.py`: external context aggregation.
- `weekly_seo_agent/clients/*.py`: API clients (GSC, Senuto, GA4, Drive, Allegro Trends, etc.).
- `weekly_seo_agent/config.py`: all runtime configuration from environment.

Execution model:

1. Parent process reads config and resolves countries.
2. Each country runs in a separate process (`ProcessPoolExecutor`) in parallel.
3. Per-country workflow fetches data, analyzes deltas, enriches context, generates narrative, validates output, and returns final markdown.
4. Markdown is rendered to `.docx`; optional upload creates a Google Doc with the same name.

## 3. End-to-End Flow (per country)

Pipeline implemented in `weekly_seo_agent/workflow.py`:

1. Build date windows (weekly baseline + previous week + aligned 52W).
2. Pull GSC data for configured dimension scopes (query/page/etc.).
3. Filter irrelevant queries/noise.
4. Compute KPIs, movers, findings (`analysis.py`).
5. Pull Senuto + optional Allegro Trends + continuity/status/trends context (`additional_context.py` and clients).
6. Pull external signals (news, weather, holidays/ferie, Google status/blog, optional CrUX, market events).
7. Build structured markdown report (`reporting.py`).
8. Run multi-step LLM generation:
   - key-data extraction packets,
   - split prompts on smaller packets,
   - merge synthesis for `Executive summary` and `What is happening and why`.
9. Run LLM validator rounds (up to configured max) to catch inconsistencies.
10. Produce final markdown and render `.docx`.

## 4. Data Sources

Core:

- Google Search Console API (organic baseline).
- Senuto API (visibility + competitor intelligence modules).
- Google Analytics 4 Data API (context; currently controlled by policy switches in report logic).

External:

- Open-Meteo / weather sources (historical + forecast context).
- OpenHolidays API (holidays/ferie context by country).
- Google Search Status and Search Central Blog.
- Market/news feeds (RSS and selected HTML sources).
- GDELT market-event feed (country event calendar).
- Google Trends RSS.
- NBP FX and IMGW warnings (PL-specific where configured).
- Allegro Trends API (optional demand signals from marketplace search).

Collaboration/context:

- Google Drive presentations folder (SEO team decks).
- Google Sheets status log.
- Previous generated reports (continuity checks).

## 5. Reporting Model

Output is intentionally split into two layers:

1. Decision layer (for managers):
   - `Leadership snapshot`
   - `Executive summary`
   - `What is happening and why`
2. Evidence layer (for specialists/analysts):
   - structured appendix tables, movers, source-level context.

Design intent:

- Keep top sections concise and interpretation-focused.
- Keep raw diagnostics in appendix.
- Avoid repeating the same signal in multiple sections.

## 6. Multi-Country Strategy

In `main.py`, each country gets a derived config via `_build_country_config(...)`:

- `gsc_country_filter`: mapped (`PL->pol`, `CZ->cze`, `SK->svk`, `HU->hun`).
- `senuto_country_id`: resolved from `SENUTO_COUNTRY_ID_MAP`.
- `ga4_property_id`: resolved from `GA4_PROPERTY_ID_MAP`.
- localized weather coordinates/labels.
- localized holidays language/country mapping.
- localized trends/news settings where applicable.

Reports are generated in parallel processes to reduce total runtime.

## 7. Configuration Contract (`.env`)

The full field list is in `.env.example`. Operationally critical groups:

1. LLM:
   - `GAIA_ENDPOINT`, `GAIA_API_VERSION`, `GAIA_MODEL`, `GAIA_API_KEY`
2. GSC:
   - OAuth client secret + refresh token, site URL, country filter defaults
3. Senuto:
   - token or email/password flow, base URL, per-country IDs
4. GA4:
   - service account JSON path + property map
5. Google Drive/Docs:
   - OAuth/service credentials, token path, destination folder
6. External context:
   - weather, holidays, market events, trends/news endpoints
7. Execution:
   - `REPORT_COUNTRIES`, output directory, thresholds, feature toggles

Configuration parsing and normalization is centralized in `weekly_seo_agent/config.py`.

## 8. Local Runbook

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Single run (all configured countries):

```bash
source .venv/bin/activate
set -a; source .env; set +a
weekly-seo-agent
```

Single date run:

```bash
weekly-seo-agent --run-date 2026-02-12
```

Wrapper script:

```bash
./scripts/run_report.sh
```

Tests:

```bash
pytest
```

## 9. Scheduling (Cron)

Recommended weekly cron:

```cron
CRON_TZ=Europe/Warsaw
0 10 * * 2 cd /path/to/repo && /path/to/repo/.venv/bin/bash -lc 'set -a; source .env; set +a; weekly-seo-agent'
```

Behavior note:

- same-day report files for matching country/date stems are replaced,
- historical reports from other dates remain intact.

## 10. Security and Secrets

Keep credentials outside code:

- use `.env` + JSON credential files not committed to VCS,
- rotate API secrets regularly,
- use least-privilege service accounts for GA4/Drive,
- share only required folders/sheets with service principals.

Sensitive files typically used:

- `secret.json` (service account),
- `client_secret_*.json` (OAuth),
- `.google_drive_token.json` (OAuth token cache),
- `.env` (runtime secrets).

## 11. Quality Controls

Current safety/quality patterns:

1. Query cleaning for irrelevant/noisy strings.
2. Brand/noise guards for Allegro Trends interpretation.
3. Multi-pass LLM generation (smaller context chunks).
4. LLM validator rounds with feedback loop.
5. Structured appendix retained as evidence trail.
6. Unit tests for key transformations/clients/time windows.

## 12. Known Constraints

1. External APIs can be rate-limited or partially unavailable.
2. Some contexts are probabilistic and require confidence labeling.
3. GA4 vs GSC comparability may require strict policy gating.
4. News/event detection depends on source coverage quality.

## 13. Extending to a General Traffic Analytics Agent

To reuse this project beyond SEO:

## 14. Free Public Source Hub (20 sources)

The workflow now includes `free_public_source_hub` in `additional_context` and report appendix.

Purpose:

- track availability and payload quality of free/public APIs + RSS,
- enrich weekly reasoning with external context without paid subscriptions,
- keep one inventory table with status (`ok` / `empty` / `error` / `integrated` / `skipped`).

Main env toggles:

- `FREE_PUBLIC_SOURCES_ENABLED=true`
- `FREE_PUBLIC_SOURCES_TOP_ROWS=3`
- `NAGER_HOLIDAYS_COUNTRY_CODE=PL`
- `EIA_API_KEY=` (optional; only if you want EIA data)

Credentials needed (and how to get them):

1. **No credential required** (public endpoints):
   - Google News RSS, Google Search Central RSS, Search Engine Journal RSS, Search Engine Land RSS,
   - Search Engine Roundtable RSS, Reuters RSS, Eurostat RSS/API endpoint checks, OECD RSS/API endpoint checks,
   - Nager.Date API, Frankfurter API, Open-Meteo API, Wikimedia Pageviews API, Wikidata API, OpenSky API.

2. **Optional API key**:
   - `EIA_API_KEY` for U.S. Energy Information Administration.
   - Get key at: `https://www.eia.gov/opendata/register.php`
   - Put it in `.env`: `EIA_API_KEY=...`

3. **Existing project credentials still required for non-free internal sources**:
   - Google Drive / Docs OAuth (`GOOGLE_DRIVE_*`),
   - GSC OAuth / service account (`GSC_*`),
   - GA4 service account (`GA4_*`),
   - Senuto (`SENUTO_*`) if used.

1. Keep `workflow.py` orchestration pattern (collect -> analyze -> synthesize -> validate -> publish).
2. Replace/extend source clients with:
   - paid channels,
   - CRM conversion data,
   - app analytics,
   - product inventory/pricing signals.
3. Keep decision/evidence split in report structure.
4. Preserve country-level process parallelization.
5. Preserve validator stage for consistency checks.

Minimal migration plan:

1. Define new `MetricSummary`/scope schemas for non-SEO channels.
2. Add new clients under `weekly_seo_agent/clients/`.
3. Add context packet builder(s) in `workflow.py`.
4. Update prompts to include channel-specific causal reasoning.
5. Add tests for each new data-contract and narrative rule.

## 14. Recommended Next Improvements

1. Introduce explicit data lineage metadata per metric (source + timestamp + transformation).
2. Add confidence calibration based on source-quality and data freshness.
3. Add regression tests for prompt outputs (golden snapshots).
4. Add run telemetry (duration, API errors, per-country success rate).
5. Add alerting for empty critical sections (e.g., no campaign signals for prolonged periods).

## 15. File Index (Most Important)

- `README.md`: operational quickstart and env setup.
- `docs/PROJECT_DOCUMENTATION.md`: this full architecture + reproduction guide.
- `weekly_seo_agent/main.py`: multi-country batch runner.
- `weekly_seo_agent/workflow.py`: LangGraph pipeline.
- `weekly_seo_agent/reporting.py`: report composition and docx rendering.
- `weekly_seo_agent/additional_context.py`: external market/news/weather enrichment.
- `scripts/run_report.sh`: simple run wrapper.
- `scripts/export_langgraph_graph.py`: graph export utility.
