from __future__ import annotations

import re
from typing import Any


REQUIRED_SECTIONS = (
    "Executive summary",
    "What is happening and why",
)

HYPOTHESIS_MARKERS = (
    "falsifier",
    "validation metric",
    "validation date",
)

JARGON_TOKENS = (
    "serp",
    "indexation",
    "canonical",
    "volatility",
    "merchant_listings",
    "product_snippets",
    "p52w",
    "query cluster",
    "algorithm context",
)

MANAGER_MARKERS = (
    "in plain language",
    "business implication",
    "decision this week",
    "data reliability",
)


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _duplicate_line_ratio(text: str) -> float:
    lines = [line.lower() for line in _non_empty_lines(text)]
    if not lines:
        return 0.0
    unique_count = len(set(lines))
    return max(0.0, 1.0 - (unique_count / len(lines)))


def _directional_conflicts(text: str) -> list[str]:
    conflicts: list[str] = []
    for line in _non_empty_lines(text):
        lower = line.lower()
        has_up = bool(re.search(r"\b(is|are)?\s*up\b|\bup wow\b|\bup yoy\b", lower))
        has_down = bool(re.search(r"\b(is|are)?\s*down\b|\bdown wow\b|\bdown yoy\b", lower))
        has_negative = bool(re.search(r"-\d+(?:[.,]\d+)?\s*(?:%|pp)\b", lower))
        has_positive = bool(re.search(r"\+\d+(?:[.,]\d+)?\s*(?:%|pp)\b", lower))
        if has_up and has_negative and not has_positive:
            conflicts.append(line)
        if has_down and has_positive and not has_negative:
            conflicts.append(line)
    return conflicts[:5]


def evaluate_report_text(report_text: str) -> dict[str, Any]:
    text = str(report_text or "")
    text_lower = text.lower()
    issues: list[str] = []
    score = 100

    if not text.strip():
        return {
            "score": 0,
            "passed": False,
            "issues": ["Empty report body."],
            "metrics": {},
        }

    def _has_section(section_title: str) -> bool:
        title = section_title.strip().lower()
        if f"## {title}" in text_lower:
            return True
        for line in text_lower.splitlines():
            if line.strip() == title:
                return True
        return False

    for section in REQUIRED_SECTIONS:
        if _has_section(section):
            continue
        if section == "What is happening and why" and _has_section("Narrative Analysis"):
            continue
        score -= 20
        issues.append(f"Missing required section: {section}")

    evidence_refs = re.findall(r"\[E\d+\]", text)
    if len(evidence_refs) < 3:
        score -= 15
        issues.append("Too few evidence references ([E#]) in narrative.")

    has_ledger = (
        "## evidence ledger" in text_lower
        or "## evidence anchors" in text_lower
        or any(
            line.strip() in {"evidence ledger", "evidence anchors"}
            for line in text_lower.splitlines()
        )
    )
    if not has_ledger:
        score -= 20
        issues.append("Missing `## Evidence ledger` or `## Evidence Anchors` section.")

    has_governance = "## governance and provenance" in text_lower or any(
        line.strip() == "governance and provenance" for line in text_lower.splitlines()
    )
    if not has_governance:
        score -= 15
        issues.append("Missing `## Governance and provenance` section.")

    marker_hits = sum(1 for marker in HYPOTHESIS_MARKERS if marker in text.lower())
    if marker_hits < 2:
        score -= 10
        issues.append("Hypothesis discipline markers are incomplete (falsifier/validation metric/date).")

    if not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text):
        score -= 10
        issues.append("No explicit ISO dates found.")

    word_count = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    if word_count > 1800:
        score -= 16
        issues.append(f"Report is too long for decision brief readability ({word_count} words).")
    elif word_count > 1400:
        score -= 8
        issues.append(f"Report is longer than target manager format ({word_count} words).")
    elif word_count < 300:
        score -= 6
        issues.append(f"Report may be too short to support decisions ({word_count} words).")

    duplicate_ratio = _duplicate_line_ratio(text)
    if duplicate_ratio > 0.20:
        score -= 12
        issues.append(f"High line duplication ratio detected ({duplicate_ratio:.2f}).")
    elif duplicate_ratio > 0.14:
        score -= 6
        issues.append(f"Moderate line duplication ratio detected ({duplicate_ratio:.2f}).")

    jargon_hits = sum(text_lower.count(token) for token in JARGON_TOKENS)
    if jargon_hits > 18:
        score -= 8
        issues.append("Too many technical terms for non-SEO audience readability.")

    confidence_mentions = len(re.findall(r"\b\d{1,3}/100\b", text))
    if confidence_mentions > 10:
        score -= 8
        issues.append(
            f"Too many confidence-score mentions ({confidence_mentions}); prefer fewer confidence callouts."
        )

    manager_marker_hits = sum(1 for marker in MANAGER_MARKERS if marker in text_lower)
    if manager_marker_hits < 3:
        score -= 6
        issues.append("Manager-oriented framing is incomplete (plain language / decision / reliability markers).")

    direction_conflicts = _directional_conflicts(text)
    if direction_conflicts:
        score -= 12
        issues.append(
            "Potential up/down wording conflict with signed metrics: "
            + "; ".join(direction_conflicts[:2])
        )

    score = max(0, min(100, score))
    return {
        "score": score,
        "passed": score >= 75 and not any("Missing required section" in item for item in issues),
        "issues": issues,
        "metrics": {
            "evidence_reference_count": len(evidence_refs),
            "has_evidence_ledger": has_ledger,
            "has_governance_section": has_governance,
            "hypothesis_marker_hits": marker_hits,
            "word_count": word_count,
            "duplicate_line_ratio": round(duplicate_ratio, 3),
            "jargon_hits": jargon_hits,
            "confidence_mentions": confidence_mentions,
            "manager_marker_hits": manager_marker_hits,
            "directional_conflicts": len(direction_conflicts),
        },
    }
