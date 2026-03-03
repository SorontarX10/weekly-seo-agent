from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .models import Document


try:
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass

DEFAULT_PLAYBOOK_FILENAME = "Executive_Playbook_Tech_Company.md"
DEFAULT_PLAYBOOK_MAX_CHARS = 6500
DEFAULT_MANAGER_ATTACHMENT_CONTEXT_CHARS = 64000
DEFAULT_MANAGER_DOCUMENT_CONTEXT_CHARS = 40000
DEFAULT_MANAGER_LLM_MAX_TOKENS = 2400
DEFAULT_MANAGER_LLM_MAX_PAGES = 4
DEFAULT_MANAGER_LLM_CONTINUATION_TAIL_CHARS = 5000


class AIContentError(Exception):
    """Raised when generated AI output fails guardrails."""


@dataclass(slots=True)
class OutlineContext:
    instructions: str
    attachments_summary: str


class AIService:
    def __init__(self, *, playbook_path: str | Path | None = None) -> None:
        self._llm_max_tokens = _resolve_manager_llm_max_tokens()
        self._llm = self._build_llm_from_env()
        self._playbook_grounding = _load_playbook_grounding(playbook_path=playbook_path)
        self._max_llm_pages = _resolve_int_env(
            "MANAGER_DOCUMENT_AGENT_LLM_MAX_PAGES",
            default=DEFAULT_MANAGER_LLM_MAX_PAGES,
            min_value=1,
            max_value=12,
        )
        self._continuation_tail_chars = _resolve_int_env(
            "MANAGER_DOCUMENT_AGENT_LLM_CONTINUATION_TAIL_CHARS",
            default=DEFAULT_MANAGER_LLM_CONTINUATION_TAIL_CHARS,
            min_value=1000,
            max_value=30000,
        )
        self._max_attachment_context_chars = _resolve_int_env(
            "MANAGER_DOCUMENT_AGENT_MAX_ATTACHMENT_CONTEXT_CHARS",
            default=DEFAULT_MANAGER_ATTACHMENT_CONTEXT_CHARS,
            min_value=2000,
            max_value=120000,
        )
        self._max_document_context_chars = _resolve_int_env(
            "MANAGER_DOCUMENT_AGENT_MAX_DOCUMENT_CONTEXT_CHARS",
            default=DEFAULT_MANAGER_DOCUMENT_CONTEXT_CHARS,
            min_value=4000,
            max_value=200000,
        )

    def generate_outline(self, document: Document, context: OutlineContext) -> str:
        if self._llm is not None:
            draft = self._llm_outline(document, context)
            try:
                validated = guardrail_full_document(
                    _sanitize_inline_markdown_emphasis(draft),
                )
                return _append_source_facts_section_if_missing(
                    outline=validated,
                    attachment_summary=context.attachments_summary,
                )
            except AIContentError:
                pass

        fallback = self._fallback_outline(document, context)
        validated_fallback = guardrail_full_document(
            _sanitize_inline_markdown_emphasis(fallback),
        )
        return _append_source_facts_section_if_missing(
            outline=validated_fallback,
            attachment_summary=context.attachments_summary,
        )

    def rewrite_full(self, document: Document, prompt: str) -> str:
        instruction = _normalize_prompt(prompt)
        if self._llm is not None:
            draft = self._llm_full_rewrite(document, instruction)
            try:
                return guardrail_full_document(_sanitize_inline_markdown_emphasis(draft))
            except AIContentError:
                pass

        fallback = self._fallback_full_rewrite(document, instruction)
        return guardrail_full_document(_sanitize_inline_markdown_emphasis(fallback))

    def rewrite_selection(
        self,
        *,
        selected_text: str,
        prompt: str,
        left_context: str,
        right_context: str,
    ) -> str:
        instruction = _normalize_prompt(prompt)
        source = selected_text.strip()
        if not source:
            raise AIContentError("Selected fragment is empty")

        if self._llm is not None:
            draft = self._llm_selection_rewrite(
                selected_text=source,
                prompt=instruction,
                left_context=left_context,
                right_context=right_context,
            )
            try:
                return guardrail_fragment(_sanitize_inline_markdown_emphasis(draft))
            except AIContentError:
                pass

        fallback = self._fallback_selection_rewrite(source, instruction)
        return guardrail_fragment(_sanitize_inline_markdown_emphasis(fallback))

    def _llm_outline(self, document: Document, context: OutlineContext) -> str:
        playbook_grounding = _playbook_prompt_block(
            self._playbook_grounding,
            max_chars=5500,
        )
        system = (
            "You are a senior strategy writer preparing management-ready documents. "
            "Write in concise, concrete business language. Avoid placeholders and generic filler. "
            "Follow executive-writing standards from the provided playbook grounding. "
            "Return markdown only."
        )
        human = f"""
Prepare a management-ready outline.

Document metadata:
- title: {document.title}
- type: {document.doc_type}
- audience: {document.target_audience}
- language: {document.language}
- objective: {document.objective}
- tone: {document.tone}
- constraints: {document.constraints}

Additional instructions:
{context.instructions or '(none)'}

Context from attachments:
{_select_context_packs((context.attachments_summary or '(none)'), max_chars=self._max_attachment_context_chars)}

Executive Playbook grounding:
{playbook_grounding}

Grounding application rules:
- Start from recommendation first (Pyramid Principle), then evidence.
- Keep top-level argumentation MECE and decision-oriented.
- Make the narrative explicit with SCQA-like flow where useful.
- Keep governance signals explicit: KPI, owner/accountability, next steps, required decision.
- Avoid anti-patterns: no decision, generic filler, or weak/no ownership.

Language policy:
- Output language must match this value: {document.language}
- If language starts with "pl", write in Polish.

Evidence requirements:
- Use concrete facts from attachments (numbers, program names, KPIs, quarters, owners).
- Include at least 6 source-specific facts across the whole document.
- Ensure each major section includes at least one source-specific detail.
- Avoid generic statements that could apply to any company.

Required structure:
# <title>
## Executive Summary
## Strategic Context
## Priorities for 2026
## Operating Model and Dependencies
## Risks and Mitigations
## KPI Framework
## 90-Day Plan
## Decisions Needed from Management

Rules:
- Include concrete bullets under each section.
- Do not include phrases like "Key point 1" or "Step 1".
- Keep it board-ready and specific to provided context.
- Avoid inline markdown emphasis markers like **bold** or __bold__ in output text.
""".strip()
        return self._invoke(system, human)

    def _llm_full_rewrite(self, document: Document, prompt: str) -> str:
        playbook_grounding = _playbook_prompt_block(
            self._playbook_grounding,
            max_chars=4200,
        )
        system = (
            "You are a staff-level strategy editor. Rewrite documents to management-ready quality. "
            "Preserve factual meaning, remove noise, and produce clear structure. "
            "Follow executive-writing standards from the provided playbook grounding. "
            "Return markdown only."
        )
        human = f"""
Rewrite the document to be management-ready.

Rewrite request:
{prompt}

Metadata:
- title: {document.title}
- audience: {document.target_audience}
- language: {document.language}
- tone: {document.tone}
- constraints: {document.constraints}

Current document:
{_select_context_packs(document.current_content, max_chars=self._max_document_context_chars)}

Executive Playbook grounding:
{playbook_grounding}

Rules:
- Keep only high-signal content.
- Add clear sections and concise bullets where useful.
- Remove duplication and weak statements.
- Do not append meta-commentary like "AI revision".
- Keep recommendation-first flow and explicit decision framing.
- Keep KPI + ownership + risks clearly visible when context allows.
- Avoid inline markdown emphasis markers like **bold** or __bold__ in output text.
""".strip()
        return self._invoke(system, human)

    def _llm_selection_rewrite(
        self,
        *,
        selected_text: str,
        prompt: str,
        left_context: str,
        right_context: str,
    ) -> str:
        playbook_grounding = _playbook_prompt_block(
            self._playbook_grounding,
            max_chars=1800,
        )
        system = (
            "You rewrite only the selected fragment. "
            "Keep local coherence with neighboring text. "
            "Respect executive-writing standards from provided grounding. "
            "Return only rewritten fragment text, no quotes, no markdown fences."
        )
        human = f"""
Rewrite selected fragment.

Instruction:
{prompt}

Left context (for coherence):
{left_context[-1000:]}

Selected fragment:
{selected_text}

Right context (for coherence):
{right_context[:1000]}

Executive Playbook grounding:
{playbook_grounding}

Rules:
- Keep same language as selected fragment.
- Improve clarity and business tone.
- Keep it concise, recommendation-oriented, and concrete.
- Do not add inline markdown emphasis markers like **bold** or __bold__.
""".strip()
        return self._invoke(system, human)

    def _fallback_outline(self, document: Document, context: OutlineContext) -> str:
        attachment_context = _trim_text(context.attachments_summary, max_chars=1800)
        instructions = context.instructions.strip()
        source_facts = _extract_facts_from_summary(attachment_context, limit=8)

        sections = [
            f"# {document.title.strip() or 'Document'}",
            "",
            "## Executive Summary",
            f"- Objective: {document.objective.strip() or 'Define the document objective clearly.'}",
            f"- Audience: {document.target_audience.strip() or 'Management'}",
            "- Expected outcome: decision-ready direction for 2026 priorities.",
            "",
            "## Strategic Context",
            "- Market and discovery behavior are shifting toward AI-assisted and zero-click journeys.",
            "- Discovery quality now directly impacts commercial performance, not only traffic volume.",
            "",
            "## Priorities for 2026",
            "- Prioritize high-impact discovery initiatives with measurable business outcomes.",
            "- Align SEO/GEO roadmap with product, content, and analytics operating cadence.",
            "- Establish repeatable experimentation and reporting routines for leadership visibility.",
            "",
            "## Operating Model and Dependencies",
            "- Define ownership per initiative and cross-team dependency map.",
            "- Clarify resource assumptions and execution constraints.",
            "",
            "## Risks and Mitigations",
            "- Risk: execution spread too thin across channels.",
            "- Mitigation: sequence roadmap by business impact and delivery capacity.",
            "- Risk: weak measurement discipline.",
            "- Mitigation: enforce KPI review cadence and data quality checks.",
            "",
            "## KPI Framework",
            "- Business: GMV contribution and conversion from organic/AI-driven journeys.",
            "- Discovery: visibility quality and answer-surface presence for strategic intents.",
            "- Delivery: initiative throughput and cycle time.",
            "",
            "## 90-Day Plan",
            "- Finalize initiative backlog and ownership matrix.",
            "- Launch first decision-grade KPI baseline and weekly governance rhythm.",
            "- Deliver first management checkpoint with risks, progress, and decisions needed.",
            "",
            "## Decisions Needed from Management",
            "- Confirm priority order for 2026 initiatives.",
            "- Confirm capacity allocation and cross-functional owners.",
            "- Confirm KPI ownership and review cadence.",
        ]

        if source_facts:
            sections.extend(["", "## Source Facts From Attachments"])
            sections.extend([f"- {fact}" for fact in source_facts])
        if instructions:
            sections.extend(["", "## Additional Instructions", instructions])
        if attachment_context:
            sections.extend(["", "## Context From Attachments", attachment_context])
        return "\n".join(sections)

    def _fallback_full_rewrite(self, document: Document, prompt: str) -> str:
        current = _trim_text(document.current_content, max_chars=9000)
        return (
            f"# {document.title}\n\n"
            f"## Executive Summary\n"
            f"This document is revised for management readability with clear business framing.\n\n"
            f"## Management Narrative\n"
            f"{current or 'Content prepared for management review.'}\n\n"
            f"## Requested Focus Applied\n"
            f"- {prompt}\n\n"
            f"## Decisions Needed\n"
            f"- Confirm priorities, ownership, and timeline.\n"
        )

    def _fallback_selection_rewrite(self, selected_text: str, prompt: str) -> str:
        trimmed = _trim_text(selected_text, max_chars=1800)
        return f"{trimmed}\n(edited for management tone: {prompt})"

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        if self._llm is None:
            raise AIContentError("LLM is not configured")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        merged = ""
        for page_index in range(self._max_llm_pages):
            response = self._llm.invoke(messages)
            raw_chunk = _normalize_whitespace(_coerce_text(response.content))
            chunk = _strip_already_generated_prefix(merged, raw_chunk)
            before_len = len(merged)
            merged = _merge_generation_chunks(merged, chunk)
            after_len = len(merged)
            if not merged.strip():
                raise AIContentError("LLM returned empty output")

            finish_reason = _extract_finish_reason(response)
            if not _should_request_continuation(
                finish_reason=finish_reason,
                chunk=chunk,
                llm_max_tokens=self._llm_max_tokens,
                page_index=page_index,
                max_pages=self._max_llm_pages,
            ):
                break
            if after_len <= before_len + 12:
                break

            messages.append(AIMessage(content=raw_chunk))
            messages.append(
                HumanMessage(
                    content=_build_continuation_prompt(
                        merged_text=merged,
                        tail_chars=self._continuation_tail_chars,
                    )
                )
            )
        return _normalize_whitespace(merged)

    @staticmethod
    def _build_llm_from_env() -> Any | None:
        enabled_raw = os.getenv("MANAGER_DOCUMENT_AGENT_USE_LLM", "")
        enabled = enabled_raw.strip().lower() in {"1", "true", "yes", "on"}
        llm_max_tokens = _resolve_manager_llm_max_tokens()

        gaia_endpoint = os.getenv("GAIA_ENDPOINT", "").strip()
        gaia_api_version = os.getenv("GAIA_API_VERSION", "").strip()
        gaia_model = os.getenv("GAIA_MODEL", "").strip()
        gaia_api_key = (
            os.getenv("GAIA_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )

        # Auto-enable when full GAIA config is present.
        if gaia_endpoint and gaia_api_version and gaia_model and gaia_api_key:
            enabled = True

        if enabled and gaia_endpoint and gaia_api_version and gaia_model and gaia_api_key:
            return AzureChatOpenAI(
                azure_endpoint=gaia_endpoint,
                api_key=gaia_api_key,
                openai_api_version=gaia_api_version,
                azure_deployment=gaia_model,
                temperature=0.2,
                max_tokens=llm_max_tokens,
                timeout=max(30, int(os.getenv("GAIA_TIMEOUT_SEC", "120"))),
                max_retries=max(0, int(os.getenv("GAIA_MAX_RETRIES", "1"))),
            )

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        if enabled and openai_api_key:
            return ChatOpenAI(
                api_key=openai_api_key,
                model=openai_model,
                temperature=0.2,
                max_tokens=llm_max_tokens,
                timeout=120,
                max_retries=1,
            )

        return None


def guardrail_full_document(text: str) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) < 120:
        raise AIContentError("AI output is too short")
    if not re.search(r"^#{1,3}\s+", normalized, flags=re.MULTILINE):
        raise AIContentError("AI output must include markdown headings")

    placeholder_patterns = [
        r"\bKey point \d+\b",
        r"\bStep \d+\b",
        r"\bTBD\b",
        r"\bLorem ipsum\b",
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            raise AIContentError("AI output contains placeholder content")
    return normalized


def guardrail_fragment(text: str) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) < 8:
        raise AIContentError("AI fragment output is too short")
    if re.search(r"\bKey point \d+\b", normalized, flags=re.IGNORECASE):
        raise AIContentError("AI fragment output contains placeholder content")
    return normalized


def _normalize_prompt(prompt: str) -> str:
    normalized = _normalize_whitespace(prompt)
    if not normalized:
        raise AIContentError("Prompt cannot be empty")
    return normalized


def _normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _sanitize_inline_markdown_emphasis(text: str) -> str:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""
    sanitized = re.sub(r"\*\*(.+?)\*\*", r"\1", normalized)
    sanitized = re.sub(r"__(.+?)__", r"\1", sanitized)
    sanitized = sanitized.replace("**", "").replace("__", "")
    return sanitized


def _select_context_packs(text: str, *, max_chars: int) -> str:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return "(none)"
    if len(normalized) <= max_chars:
        return normalized
    # Load multiple regions from long context: beginning, middle, and end.
    marker = "\n\n[... additional context packs loaded ...]\n\n"
    budget = max(1200, max_chars - (2 * len(marker)))
    head_budget = int(budget * 0.45)
    mid_budget = int(budget * 0.2)
    tail_budget = budget - head_budget - mid_budget

    head = normalized[:head_budget].rstrip()
    mid_start = max(0, (len(normalized) // 2) - (mid_budget // 2))
    middle = normalized[mid_start:mid_start + mid_budget].strip()
    tail = normalized[-tail_budget:].lstrip()

    chunks = []
    if head:
        chunks.append(f"Context pack (start):\n{head}")
    if middle:
        chunks.append(f"Context pack (middle):\n{middle}")
    if tail:
        chunks.append(f"Context pack (end):\n{tail}")
    if not chunks:
        return _trim_text(normalized, max_chars=max_chars)
    combined = marker.join(chunks)
    return _trim_text(combined, max_chars=max_chars)


def _load_playbook_grounding(*, playbook_path: str | Path | None = None) -> str:
    resolved_path = _resolve_playbook_path(playbook_path=playbook_path)
    if resolved_path is None:
        return ""
    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    normalized = _normalize_whitespace(raw_text)
    if not normalized:
        return ""
    return _trim_text(normalized, max_chars=_resolve_playbook_max_chars())


def _resolve_playbook_path(*, playbook_path: str | Path | None = None) -> Path | None:
    if playbook_path is not None:
        candidate = Path(playbook_path).expanduser()
        if candidate.is_file():
            return candidate
        return None

    configured_path = os.getenv("MANAGER_DOCUMENT_AGENT_PLAYBOOK_PATH", "").strip()
    if configured_path:
        configured_candidate = Path(configured_path).expanduser()
        if configured_candidate.is_file():
            return configured_candidate
        relative_candidate = Path.cwd() / configured_candidate
        if relative_candidate.is_file():
            return relative_candidate
        return None

    default_path = Path(__file__).with_name(DEFAULT_PLAYBOOK_FILENAME)
    if default_path.is_file():
        return default_path
    return None


def _resolve_playbook_max_chars() -> int:
    raw_value = os.getenv(
        "MANAGER_DOCUMENT_AGENT_PLAYBOOK_MAX_CHARS",
        str(DEFAULT_PLAYBOOK_MAX_CHARS),
    ).strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_PLAYBOOK_MAX_CHARS
    return min(max(parsed, 1200), 50000)


def _resolve_int_env(
    key: str,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    raw_value = os.getenv(key, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return min(max(parsed, min_value), max_value)


def _resolve_manager_llm_max_tokens() -> int:
    explicit = _resolve_int_env(
        "MANAGER_DOCUMENT_AGENT_LLM_MAX_TOKENS",
        default=0,
        min_value=600,
        max_value=6000,
    )
    if explicit:
        return explicit

    legacy = _resolve_int_env(
        "GAIA_MAX_OUTPUT_TOKENS",
        default=0,
        min_value=600,
        max_value=6000,
    )
    if legacy:
        return max(DEFAULT_MANAGER_LLM_MAX_TOKENS, legacy)
    return DEFAULT_MANAGER_LLM_MAX_TOKENS


def _playbook_prompt_block(playbook_grounding: str, *, max_chars: int) -> str:
    trimmed = _trim_text(playbook_grounding, max_chars=max_chars).strip()
    if trimmed:
        return trimmed
    return "(none)"


def _trim_text(text: str, *, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    chunks.append(text)
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _extract_finish_reason(response: Any) -> str:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        finish_reason = metadata.get("finish_reason")
        if isinstance(finish_reason, str):
            return finish_reason.strip().lower()
        if isinstance(finish_reason, list) and finish_reason:
            first = finish_reason[0]
            if isinstance(first, str):
                return first.strip().lower()
        nested = metadata.get("model_extra")
        if isinstance(nested, dict):
            nested_reason = nested.get("finish_reason")
            if isinstance(nested_reason, str):
                return nested_reason.strip().lower()

    additional_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        finish_reason = additional_kwargs.get("finish_reason")
        if isinstance(finish_reason, str):
            return finish_reason.strip().lower()
    return ""


def _should_request_continuation(
    *,
    finish_reason: str,
    chunk: str,
    llm_max_tokens: int,
    page_index: int,
    max_pages: int,
) -> bool:
    if page_index + 1 >= max_pages:
        return False

    normalized_reason = (finish_reason or "").strip().lower()
    if normalized_reason in {"length", "max_tokens"}:
        return True
    if normalized_reason in {"stop", "end_turn"}:
        return False

    approx_char_limit = max(1200, int(llm_max_tokens * 2.6))
    if len(chunk) >= int(approx_char_limit * 0.9):
        return True
    return _looks_truncated_tail(chunk)


def _looks_truncated_tail(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return False
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return False
    tail = lines[-1]
    if tail.endswith((":", ",", ";", "-", "(", "/", "|")):
        return True
    if re.match(r"^(?:[IVX]{1,6}[.)-]?)$", tail):
        return True
    return False


def _merge_generation_chunks(current: str, chunk: str) -> str:
    incoming = (chunk or "").strip()
    if not incoming:
        return current
    if not current:
        return incoming
    if incoming in current:
        return current
    if not current.endswith("\n") and not incoming.startswith("\n"):
        return f"{current}\n{incoming}".strip()
    return f"{current}{incoming}".strip()


def _strip_already_generated_prefix(current: str, chunk: str) -> str:
    existing = (current or "").strip()
    incoming = (chunk or "").strip()
    if not existing or not incoming:
        return incoming

    max_overlap = min(len(existing), len(incoming), 1800)
    for overlap in range(max_overlap, 40, -1):
        if existing.endswith(incoming[:overlap]):
            return incoming[overlap:].lstrip()

    existing_lines = [line.strip() for line in existing.splitlines() if line.strip()]
    incoming_lines = [line.strip() for line in incoming.splitlines() if line.strip()]
    if existing_lines and incoming_lines:
        last_line = _normalize_overlap_line(existing_lines[-1])
        first_line = _normalize_overlap_line(incoming_lines[0])
        if first_line and (first_line == last_line or first_line in last_line):
            incoming_lines = incoming_lines[1:]
            incoming = "\n".join(incoming_lines).strip()
    return incoming


def _build_continuation_prompt(*, merged_text: str, tail_chars: int) -> str:
    tail = _tail_text(merged_text, max_chars=tail_chars)
    return (
        "Continue from exactly where you stopped.\n"
        "Return only the missing continuation in the same language and format.\n"
        "Do not repeat headings or bullets already written.\n"
        "Do not restart from the document title.\n\n"
        "Already generated tail (for alignment):\n"
        f"{tail}"
    ).strip()


def _tail_text(text: str, *, max_chars: int) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:].lstrip()


def _normalize_overlap_line(line: str) -> str:
    normalized = re.sub(r"^[-*]\s+", "", (line or "").strip())
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _extract_facts_from_summary(summary: str, *, limit: int) -> list[str]:
    if not summary.strip():
        return []
    candidates: list[str] = []
    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        cleaned = line.lstrip("-").strip()
        if re.search(r"\.(docx|xlsx|csv|tsv|txt|pdf)$", cleaned, flags=re.IGNORECASE):
            continue
        if cleaned.endswith("..."):
            continue
        if not cleaned or "." not in cleaned and ":" not in cleaned and not any(ch.isdigit() for ch in cleaned):
            continue
        candidates.append(cleaned)

    if not candidates:
        return []

    facts: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        if len(facts) >= limit:
            return
        if value in seen:
            return
        seen.add(value)
        facts.append(value)

    def _push_many(predicate, *, cap: int) -> None:
        pushed = 0
        for value in candidates:
            if pushed >= cap or len(facts) >= limit:
                break
            if not predicate(value):
                continue
            before = len(facts)
            _push(value)
            if len(facts) > before:
                pushed += 1

    _push_many(_is_summary_metric_fact, cap=4)
    _push_many(_is_summary_program_fact, cap=2)
    for value in candidates:
        _push(value)
    return facts


def _append_source_facts_section_if_missing(
    *,
    outline: str,
    attachment_summary: str,
) -> str:
    normalized = outline.rstrip()
    if not attachment_summary.strip():
        return normalized
    if re.search(
        r"^##\s+Source Facts From Attachments\b",
        normalized,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        return normalized
    facts = _extract_facts_from_summary(attachment_summary, limit=6)
    if not facts:
        return normalized
    lines = [normalized, "", "## Source Facts From Attachments"]
    lines.extend([f"- {fact}" for fact in facts])
    return "\n".join(lines).strip()


def _is_summary_metric_fact(value: str) -> bool:
    lowered = value.lower()
    if "%" in value:
        return True
    return bool(
        re.search(r"\b(?:gmv|kpi|crvisit|q1|q2|q3|q4|2026|2025)\b", lowered)
    )


def _is_summary_program_fact(value: str) -> bool:
    if re.match(r"^(?:[IVX]{1,6}[.)-]\s+)", value):
        return True
    return bool(re.search(r"\b(?:program|roadmap|priority|goal)\b", value, flags=re.IGNORECASE))
