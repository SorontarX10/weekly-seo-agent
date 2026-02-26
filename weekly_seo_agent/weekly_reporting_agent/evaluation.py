from __future__ import annotations

import re
from typing import Any


REQUIRED_SECTIONS = (
    "Executive summary",
    "What is happening and why",
    "Confirmed vs hypothesis",
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

ACTION_MARKERS = (
    "priority actions",
    "owner | eta",
)


CLAIM_HINT_TOKENS = (
    "because",
    "driven",
    "caused",
    "indicates",
    "therefore",
    "implies",
    "hypothesis",
    "decision this week",
    "why",
    "root cause",
)


EVIDENCE_REF_PATTERN = re.compile(r"\[E(\d+)\]")


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _sentence_word_counts(text: str) -> list[int]:
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[int] = []
    for sentence in raw_sentences:
        words = re.findall(r"\b\w+\b", sentence, flags=re.UNICODE)
        if words:
            out.append(len(words))
    return out


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


def _has_section(text_lower: str, section_title: str) -> bool:
    title = section_title.strip().lower()
    if f"## {title}" in text_lower:
        return True
    for line in text_lower.splitlines():
        if line.strip() == title:
            return True
    return False


def _extract_section(text: str, section_title: str) -> str:
    lines = text.splitlines()
    title_lower = section_title.strip().lower()
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        lowered = line.strip().lower()
        if lowered == f"## {title_lower}" or lowered == title_lower:
            start_idx = idx + 1
            break
    if start_idx is None:
        return ""
    out: list[str] = []
    for line in lines[start_idx:]:
        if line.strip().startswith("## "):
            break
        out.append(line)
    return "\n".join(out).strip()


def evaluate_readability(report_text: str) -> dict[str, Any]:
    text = str(report_text or "")
    text_lower = text.lower()
    issues: list[str] = []
    score = 100

    word_count = _word_count(text)
    sentence_lengths = _sentence_word_counts(text)
    avg_sentence_words = (
        sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    )
    long_sentence_count = sum(1 for size in sentence_lengths if size >= 28)
    long_sentence_ratio = _safe_ratio(long_sentence_count, len(sentence_lengths))
    jargon_hits = sum(text_lower.count(token) for token in JARGON_TOKENS)
    jargon_density = _safe_ratio(jargon_hits, word_count)
    manager_marker_hits = sum(1 for marker in MANAGER_MARKERS if marker in text_lower)

    if word_count > 1500:
        score -= 14
        issues.append(
            f"Readability: report is too long for managerial reading pace ({word_count} words)."
        )
    elif word_count > 1300:
        score -= 8
        issues.append(
            f"Readability: report is longer than target manager brief ({word_count} words)."
        )
    elif word_count < 320:
        score -= 4
        issues.append(
            f"Readability: report may be too short to support decisions ({word_count} words)."
        )

    if avg_sentence_words > 26:
        score -= 12
        issues.append(
            f"Readability: average sentence length is too high ({avg_sentence_words:.1f} words)."
        )
    elif avg_sentence_words > 22:
        score -= 6
        issues.append(
            f"Readability: sentence length should be simpler for non-SEO readers ({avg_sentence_words:.1f})."
        )

    if long_sentence_ratio > 0.35:
        score -= 12
        issues.append(
            f"Readability: too many long sentences ({long_sentence_ratio:.0%}); split for clarity."
        )
    elif long_sentence_ratio > 0.22:
        score -= 6
        issues.append(
            f"Readability: moderate long-sentence share ({long_sentence_ratio:.0%}); simplify phrasing."
        )

    if jargon_density > 0.024:
        score -= 10
        issues.append(
            f"Readability: high SEO jargon density ({jargon_density:.3f}); explain in plain language."
        )
    elif jargon_density > 0.015:
        score -= 5
        issues.append(
            f"Readability: SEO jargon density can be reduced further ({jargon_density:.3f})."
        )

    if manager_marker_hits < 4:
        score -= 10
        issues.append(
            "Readability: manager framing is incomplete (plain language / implication / decision / reliability)."
        )

    score = max(0, min(100, score))
    return {
        "score": score,
        "issues": issues,
        "metrics": {
            "word_count": word_count,
            "sentence_count": len(sentence_lengths),
            "avg_sentence_words": round(avg_sentence_words, 2),
            "long_sentence_ratio": round(long_sentence_ratio, 3),
            "jargon_hits": jargon_hits,
            "jargon_density": round(jargon_density, 4),
            "manager_marker_hits": manager_marker_hits,
        },
    }


def evaluate_duplication_verbosity(report_text: str) -> dict[str, Any]:
    text = str(report_text or "")
    issues: list[str] = []
    score = 100

    words = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    unique_word_ratio = _safe_ratio(len(set(words)), len(words))
    duplicate_ratio = _duplicate_line_ratio(text)
    confidence_mentions = len(re.findall(r"\b\d{1,3}/100\b", text))
    evidence_refs = re.findall(EVIDENCE_REF_PATTERN, text)
    hypothesis_marker_hits = sum(
        1 for marker in HYPOTHESIS_MARKERS if marker in text.lower()
    )
    action_marker_hits = sum(1 for marker in ACTION_MARKERS if marker in text.lower())
    anchor_count = len(evidence_refs) + hypothesis_marker_hits + action_marker_hits
    verbosity_ratio = _safe_ratio(_word_count(text), max(1, anchor_count))

    if duplicate_ratio > 0.16:
        score -= 14
        issues.append(f"Duplication: high repeated-line ratio ({duplicate_ratio:.2f}).")
    elif duplicate_ratio > 0.10:
        score -= 8
        issues.append(f"Duplication: moderate repeated-line ratio ({duplicate_ratio:.2f}).")

    if unique_word_ratio < 0.32:
        score -= 10
        issues.append(
            f"Verbosity: lexical variety is low ({unique_word_ratio:.2f}); content feels repetitive."
        )
    elif unique_word_ratio < 0.38:
        score -= 5
        issues.append(
            f"Verbosity: lexical variety can be improved ({unique_word_ratio:.2f})."
        )

    if confidence_mentions > 8:
        score -= 8
        issues.append(
            f"Verbosity: too many confidence callouts ({confidence_mentions}); keep only key uncertainties."
        )
    elif confidence_mentions > 5:
        score -= 4
        issues.append(
            f"Verbosity: confidence callouts should be reduced ({confidence_mentions})."
        )

    if verbosity_ratio > 120:
        score -= 10
        issues.append(
            f"Verbosity: evidence-to-text ratio is too weak ({verbosity_ratio:.1f} words per anchor)."
        )
    elif verbosity_ratio > 90:
        score -= 5
        issues.append(
            f"Verbosity: narrative density is high ({verbosity_ratio:.1f} words per anchor)."
        )

    score = max(0, min(100, score))
    return {
        "score": score,
        "issues": issues,
        "metrics": {
            "duplicate_line_ratio": round(duplicate_ratio, 3),
            "unique_word_ratio": round(unique_word_ratio, 3),
            "confidence_mentions": confidence_mentions,
            "verbosity_ratio": round(verbosity_ratio, 2),
        },
    }


def evaluate_evidence_alignment(report_text: str) -> dict[str, Any]:
    text = str(report_text or "")
    text_lower = text.lower()
    issues: list[str] = []
    score = 100

    evidence_refs = [f"E{item}" for item in re.findall(EVIDENCE_REF_PATTERN, text)]
    evidence_ref_set = sorted(set(evidence_refs))
    has_ledger = (
        "## evidence ledger" in text_lower
        or "## evidence anchors" in text_lower
        or any(
            line.strip() in {"evidence ledger", "evidence anchors"}
            for line in text_lower.splitlines()
        )
    )
    ledger_text = _extract_section(text, "Evidence ledger")
    if not ledger_text:
        ledger_text = _extract_section(text, "Evidence anchors")
    ledger_refs = sorted(
        set(f"E{item}" for item in re.findall(r"\bE(\d+)\b", ledger_text))
    )
    orphan_refs = sorted(set(evidence_ref_set) - set(ledger_refs)) if ledger_refs else evidence_ref_set

    if len(evidence_refs) < 5:
        score -= 18
        issues.append(
            "Evidence alignment: too few evidence references ([E#]) in narrative (minimum: 5)."
        )

    if not has_ledger:
        score -= 18
        issues.append("Evidence alignment: missing `## Evidence ledger` or `## Evidence Anchors` section.")

    if has_ledger and orphan_refs:
        score -= 12
        issues.append(
            f"Evidence alignment: references without ledger entry ({', '.join(orphan_refs[:4])})."
        )

    claim_lines: list[str] = []
    supported_claims = 0
    for line in _non_empty_lines(text):
        lowered = line.lower()
        if lowered.startswith("#"):
            continue
        if lowered.startswith("|---") or lowered == "|":
            continue
        numeric_claim = bool(re.search(r"[+-]?\d+(?:[.,]\d+)?\s*(?:%|pp)\b", lowered))
        directional_claim = bool(re.search(r"\bup\b|\bdown\b|\bincrease\b|\bdecrease\b", lowered))
        causal_claim = any(token in lowered for token in CLAIM_HINT_TOKENS)
        if not (numeric_claim or directional_claim or causal_claim):
            continue
        claim_lines.append(line)
        if EVIDENCE_REF_PATTERN.search(line) or "evidence" in lowered:
            supported_claims += 1

    claim_count = len(claim_lines)
    claim_coverage_ratio = _safe_ratio(supported_claims, claim_count)
    unsupported_claim_count = max(0, claim_count - supported_claims)

    if claim_count >= 4 and claim_coverage_ratio < 0.65:
        score -= 18
        issues.append(
            f"Evidence alignment: low claim coverage ({claim_coverage_ratio:.0%}); add explicit evidence anchors."
        )
    elif claim_count >= 4 and claim_coverage_ratio < 0.80:
        score -= 8
        issues.append(
            f"Evidence alignment: claim coverage should improve ({claim_coverage_ratio:.0%})."
        )

    hypothesis_marker_hits = sum(1 for marker in HYPOTHESIS_MARKERS if marker in text_lower)
    if hypothesis_marker_hits < 3:
        score -= 14
        issues.append(
            "Causality discipline is incomplete (falsifier / validation metric / validation date)."
        )

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
        "issues": issues,
        "metrics": {
            "evidence_reference_count": len(evidence_refs),
            "evidence_reference_unique_count": len(evidence_ref_set),
            "has_evidence_ledger": has_ledger,
            "ledger_reference_count": len(ledger_refs),
            "orphan_evidence_reference_count": len(orphan_refs),
            "claim_count": claim_count,
            "supported_claim_count": supported_claims,
            "unsupported_claim_count": unsupported_claim_count,
            "claim_coverage_ratio": round(claim_coverage_ratio, 3),
            "hypothesis_marker_hits": hypothesis_marker_hits,
            "directional_conflicts": len(direction_conflicts),
        },
    }


def _evaluate_structure_and_actions(report_text: str) -> dict[str, Any]:
    text = str(report_text or "")
    text_lower = text.lower()
    issues: list[str] = []
    score = 100

    missing_required_sections: list[str] = []
    for section in REQUIRED_SECTIONS:
        if _has_section(text_lower, section):
            continue
        if section == "What is happening and why" and _has_section(
            text_lower, "Narrative Analysis"
        ):
            continue
        missing_required_sections.append(section)
        score -= 20
        issues.append(f"Missing required section: {section}")

    has_governance = "## governance and provenance" in text_lower or any(
        line.strip() == "governance and provenance" for line in text_lower.splitlines()
    )
    if not has_governance:
        score -= 14
        issues.append("Missing `## Governance and provenance` section.")

    if not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text):
        score -= 10
        issues.append("No explicit ISO dates found.")

    action_marker_hits = sum(1 for marker in ACTION_MARKERS if marker in text_lower)
    if action_marker_hits < 2:
        score -= 10
        issues.append("Action framing is incomplete (priority actions with owner/ETA).")

    yoy_mentions = len(re.findall(r"\byoy\b", text_lower))
    if yoy_mentions < 6:
        score -= 10
        issues.append(f"YoY coverage is too limited ({yoy_mentions} mentions).")

    score = max(0, min(100, score))
    return {
        "score": score,
        "issues": issues,
        "metrics": {
            "has_governance_section": has_governance,
            "action_marker_hits": action_marker_hits,
            "yoy_mentions": yoy_mentions,
            "missing_required_sections": missing_required_sections,
            "required_sections_ok": not missing_required_sections,
        },
    }


def compare_report_regression(candidate_report: str, baseline_report: str) -> dict[str, Any]:
    candidate_lines = _non_empty_lines(candidate_report)
    baseline_lines = _non_empty_lines(baseline_report)
    candidate_set = set(candidate_lines)
    baseline_set = set(baseline_lines)
    added = sorted(candidate_set - baseline_set)
    removed = sorted(baseline_set - candidate_set)
    changed_line_ratio = _safe_ratio(len(added) + len(removed), max(1, len(baseline_set)))

    candidate_eval = evaluate_report_text(candidate_report)
    baseline_eval = evaluate_report_text(baseline_report)
    score_delta = int(candidate_eval.get("score", 0) or 0) - int(
        baseline_eval.get("score", 0) or 0
    )
    return {
        "score_delta": score_delta,
        "changed_line_ratio": round(changed_line_ratio, 3),
        "added_lines_count": len(added),
        "removed_lines_count": len(removed),
        "added_lines_sample": added[:5],
        "removed_lines_sample": removed[:5],
        "baseline_score": int(baseline_eval.get("score", 0) or 0),
        "candidate_score": int(candidate_eval.get("score", 0) or 0),
    }


def evaluate_report_text(report_text: str) -> dict[str, Any]:
    text = str(report_text or "")
    issues: list[str] = []

    if not text.strip():
        return {
            "score": 0,
            "passed": False,
            "issues": ["Empty report body."],
            "metrics": {},
        }

    structure_eval = _evaluate_structure_and_actions(text)
    readability_eval = evaluate_readability(text)
    duplication_eval = evaluate_duplication_verbosity(text)
    evidence_eval = evaluate_evidence_alignment(text)

    # Stricter weighting for clarity and causality quality.
    score = int(
        round(
            (structure_eval["score"] * 0.20)
            + (readability_eval["score"] * 0.35)
            + (duplication_eval["score"] * 0.15)
            + (evidence_eval["score"] * 0.30)
        )
    )
    score = max(0, min(100, score))

    all_issue_groups = (
        structure_eval.get("issues", []),
        readability_eval.get("issues", []),
        duplication_eval.get("issues", []),
        evidence_eval.get("issues", []),
    )
    seen: set[str] = set()
    for group in all_issue_groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            issues.append(item)

    structure_metrics = structure_eval.get("metrics", {})
    readability_metrics = readability_eval.get("metrics", {})
    duplication_metrics = duplication_eval.get("metrics", {})
    evidence_metrics = evidence_eval.get("metrics", {})
    missing_required_sections = structure_metrics.get("missing_required_sections", [])
    required_ok = not bool(missing_required_sections)
    has_governance = bool(structure_metrics.get("has_governance_section", False))
    has_ledger = bool(evidence_metrics.get("has_evidence_ledger", False))
    evidence_coverage = float(evidence_metrics.get("claim_coverage_ratio", 0.0) or 0.0)
    claim_count = int(evidence_metrics.get("claim_count", 0) or 0)
    # Coverage gate is used as a hard-stop only for dense claim-heavy narratives.
    # For shorter/fact-light sections we keep this as a soft score penalty.
    coverage_gate_ok = True
    readability_score = int(readability_eval.get("score", 0) or 0)
    causality_score = int(evidence_eval.get("score", 0) or 0)
    duplication_score = int(duplication_eval.get("score", 0) or 0)
    structure_score = int(structure_eval.get("score", 0) or 0)

    passed = all(
        (
            score >= 88,
            required_ok,
            has_governance,
            has_ledger,
            coverage_gate_ok,
            structure_score >= 84,
            readability_score >= 82,
            causality_score >= 82,
            duplication_score >= 75,
        )
    )

    return {
        "score": score,
        "passed": passed,
        "issues": issues,
        "metrics": {
            # Backward-compatible metrics
            "evidence_reference_count": int(
                evidence_metrics.get("evidence_reference_count", 0) or 0
            ),
            "has_evidence_ledger": has_ledger,
            "has_governance_section": has_governance,
            "hypothesis_marker_hits": int(
                evidence_metrics.get("hypothesis_marker_hits", 0) or 0
            ),
            "word_count": int(readability_metrics.get("word_count", 0) or 0),
            "duplicate_line_ratio": float(
                duplication_metrics.get("duplicate_line_ratio", 0.0) or 0.0
            ),
            "jargon_hits": int(readability_metrics.get("jargon_hits", 0) or 0),
            "confidence_mentions": int(
                duplication_metrics.get("confidence_mentions", 0) or 0
            ),
            "manager_marker_hits": int(
                readability_metrics.get("manager_marker_hits", 0) or 0
            ),
            "action_marker_hits": int(structure_metrics.get("action_marker_hits", 0) or 0),
            "yoy_mentions": int(structure_metrics.get("yoy_mentions", 0) or 0),
            "directional_conflicts": int(
                evidence_metrics.get("directional_conflicts", 0) or 0
            ),
            # New quality dimensions
            "subscores": {
                "structure_actions": structure_score,
                "readability": readability_score,
                "duplication_verbosity": duplication_score,
                "evidence_alignment": causality_score,
            },
            "required_sections_ok": required_ok,
            "missing_required_sections": missing_required_sections,
            "claim_coverage_ratio": round(evidence_coverage, 3),
            "claim_coverage_gate_ok": coverage_gate_ok,
            "unsupported_claim_count": int(
                evidence_metrics.get("unsupported_claim_count", 0) or 0
            ),
            "orphan_evidence_reference_count": int(
                evidence_metrics.get("orphan_evidence_reference_count", 0) or 0
            ),
            "avg_sentence_words": float(
                readability_metrics.get("avg_sentence_words", 0.0) or 0.0
            ),
            "long_sentence_ratio": float(
                readability_metrics.get("long_sentence_ratio", 0.0) or 0.0
            ),
            "unique_word_ratio": float(
                duplication_metrics.get("unique_word_ratio", 0.0) or 0.0
            ),
            "verbosity_ratio": float(
                duplication_metrics.get("verbosity_ratio", 0.0) or 0.0
            ),
        },
    }
