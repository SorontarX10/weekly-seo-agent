# Weekly Reporter - Section A Baseline and Guardrails

This file closes implementation package **Section A (001-008)** and defines the frozen baseline plus editorial/quality guardrails.

## A001-A002: Frozen baseline snapshot

Baseline folder:
- `logs/baselines/2026_02_25_theme_A`

Frozen artifacts:
- `logs/baselines/2026_02_25_theme_A/2026_02_25_pl_seo_weekly_report.docx`
- `logs/baselines/2026_02_25_theme_A/pl_pipeline_markdown_report.md`
- `logs/baselines/2026_02_25_theme_A/pl_final_report.md`
- `logs/baselines/2026_02_25_theme_A/pl_workflow_quality_eval.json`
- `logs/baselines/2026_02_25_theme_A/quality_baseline_by_country.json`
- `logs/baselines/2026_02_25_theme_A/BASELINE_MANIFEST.json`

Quality baseline by country for `2026_02_25` snapshot:
- `PL`: score `100`, passed `true`
- `CZ`: score `56`, passed `false`
- `SK`: score `56`, passed `false`
- `HU`: score `56`, passed `false`

## A003: Target audience contract

Primary audience:
- business managers and cross-functional stakeholders with partial SEO knowledge.

Writing contract:
- first explain business impact, then explain SEO mechanism.
- each key point must answer: what changed, why it matters, what we do next.
- do not require technical SEO knowledge to follow the core narrative.

## A004: Target report length

Hard targets:
- Executive summary: `120-170` words.
- Main narrative ("What is happening and why"): `350-550` words.
- Confirmed vs hypothesis: `120-220` words.
- Driver scoreboard + actions: `120-220` words.
- Appendix (technical context): `max 450` words.

Global target:
- total report length: `900-1300` words, hard cap `1450`.

## A005: Mandatory and optional sections

Mandatory sections:
- `# Weekly SEO Intelligence Report (...)`
- `## Executive summary`
- `## What is happening and why`
- `## Confirmed vs hypothesis`
- `## Driver Scoreboard`
- `## Reasoning ledger (facts -> hypotheses -> validation)`
- `## Evidence ledger`
- `## Governance and provenance`

Optional sections (only when data exists):
- daily anomaly block (GSC day-level shift + likely drivers).
- trade-plan YoY block.
- SERP appearance WoW/MoM/YoY block.
- weather impact block.
- external timeline/context block.

Optional-section rule:
- if data is missing, show one short "not available" sentence instead of synthetic filler.

## A006: Language rules and banned jargon

Language rules:
- English only.
- short active sentences.
- one metric sentence = one interpretation sentence.
- no profanity, no colloquial slang.

Banned or discouraged terms in main narrative (allowed in appendix only):
- `indexation`, `canonical`, `SERP volatility`, `P52W`, `query cluster`.

Preferred business wording:
- "visibility mix across result types" instead of "SERP feature volatility".
- "demand rotation" instead of "query-cluster shift".
- "measurement window aligned YoY" instead of "P52W alignment".

## A007: Canonical metric naming dictionary

Use exactly these labels in the report:
- `Organic clicks` (GSC clicks)
- `Organic impressions` (GSC impressions)
- `Organic CTR` (GSC CTR)
- `Average position` (GSC avg position)
- `Brand clicks` (GSC branded clicks)
- `Brand impressions` (GSC branded impressions)
- `Non-brand clicks` (GSC non-brand clicks)
- `Weather delta (temperature)`
- `Weather delta (precipitation)`
- `Campaign overlap`
- `Trade-plan overlap`
- `Timeline signal density`
- `Feature mix shift` (for GSC search appearance types)

Comparison labels:
- `WoW` = week over previous week
- `MoM` = trailing-28-day period over previous trailing-28-day period
- `YoY` = aligned year-over-year window

## A008: Confidence scale rubric

Confidence interpretation:
- `0-39`: weak signal. Mention only as low-priority context.
- `40-59`: directional signal. Use as supporting hypothesis, not decision driver.
- `60-79`: moderate confidence. Actionable with validation checkpoint next run.
- `80-89`: strong confidence. Can drive primary weekly decision if no contradictions.
- `90-100`: very strong confidence. Consistent multi-source evidence and no material conflicts.

Confidence assignment rules:
- increase confidence when multiple independent sources agree.
- decrease confidence when freshness is poor, data is sparse, or signals conflict.
- never use confidence alone; always pair with concrete evidence references.
