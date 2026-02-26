# Weekly Reporter Rollout J - Business Changelog (2026-02-26)

## What Improved
- The weekly workflow now runs in enforced LLM mode (no manual/non-LLM fallback path).
- Report publication is protected by a quality gate and a per-country delivery status matrix.
- Every run now generates a machine-readable run manifest with model/runtime/source metadata.
- Google Drive publication is now verified after upload (document exists, title matches, folder matches, timestamp captured).
- Overwrite policy is idempotent for both local DOCX and Drive Google Docs (same date + country key).
- Post-run summary now shows clear delivery KPIs (success/fail/uploaded/quality min-max-average).

## Why It Matters
- Lower delivery risk: broken runs are isolated per country and no longer hidden in long logs.
- Better trust in outputs: uploaded docs are verified, not just "attempted".
- Better operational visibility: one manifest file contains exactly what happened and why.
- Predictable quality governance: low-quality outputs can be stopped before business distribution.

## Rollout J Results (071-076)
### PL Dry-Run Baseline vs Current
- Baseline (pre-rollout snapshot): **72/100**, gate failed.
- Current PL runs after fixes: **88/100** and **91/100**, gate passed.
- Net quality delta vs baseline: **+16 to +19 points**.

### Country Rollout Stability
- CZ: **91/100**, gate passed.
- SK: **91/100**, gate passed.
- HU: **88/100**, gate passed.
- PL: **91/100**, gate passed.

### Full 4-Country Batch with Drive Publish
- Run date: **2026-02-25**.
- Countries requested: **PL, CZ, SK, HU**.
- Final delivery: **4/4 success**, **4/4 uploaded**, **0 gate failures**.
- Batch quality summary: **avg 90.25**, **min 88**, **max 91**.

## Known Context Warnings (Non-Blocking)
- External market-event source (GDELT) occasionally returns `429`/timeout under parallel load.
- This is currently treated as supporting context degradation (warning), not a blocker.

## Next Recommended Step
- Add adaptive retry + backoff for GDELT and cache-first read strategy during parallel country runs to reduce warning noise.
