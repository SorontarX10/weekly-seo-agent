from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
import uuid

PLANNING_STATUS_COLLECTING = "COLLECTING"
PLANNING_STATUS_READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
PLANNING_STATUS_APPROVED = "APPROVED"
PLANNING_STATUS_DOCUMENT_CREATED = "DOCUMENT_CREATED"

PLANNING_STEPS: list[tuple[str, str]] = [
    (
        "objective",
        "1/6: Co dokładnie dokument ma dostarczyć? Opisz cel i kontekst biznesowy w 2-4 zdaniach.",
    ),
    (
        "emphasis",
        "2/6: Co chcemy najmocniej podkreślić (np. wyniki, ryzyka, plan 90 dni, decyzje)?",
    ),
    (
        "decisions_needed",
        "3/6: Jakie decyzje lub akceptacje są potrzebne od odbiorców dokumentu?",
    ),
    (
        "audience_doc_type",
        "4/6: Dla kogo dokument i jakiego typu? (np. Management + MANAGEMENT_BRIEF, Zarząd + STRATEGY_PAPER).",
    ),
    (
        "language",
        "5/6: W jakim języku ma być dokument? (np. polski/pl lub english/en).",
    ),
    (
        "tone_constraints",
        "6/6: Jaki ton i ograniczenia? (np. formalny, max 2 strony, bez danych poufnych).",
    ),
]


def new_planning_session(*, initial_brief: dict[str, str]) -> dict:
    now = _now_iso()
    session = {
        "id": str(uuid.uuid4()),
        "status": PLANNING_STATUS_COLLECTING,
        "step_index": 0,
        "suggested_points": [],
        "approved_points": [],
        "created_document_id": "",
        "created_at": now,
        "updated_at": now,
        "brief": _build_initial_brief(initial_brief),
        "messages": [],
    }
    _append_message(
        session,
        role="assistant",
        content=(
            "Zanim utworzymy dokument, doprecyzujmy plan. "
            "Po zebraniu informacji zaproponuję punkty do zatwierdzenia."
        ),
    )
    _append_message(session, role="assistant", content=PLANNING_STEPS[0][1])
    return session


def planning_session_user_turn(*, session: dict, user_message: str) -> dict:
    message = _normalize_text(user_message)
    if not message:
        raise ValueError("Message cannot be empty")

    _append_message(session, role="user", content=message)
    status = str(session.get("status", ""))

    if status == PLANNING_STATUS_DOCUMENT_CREATED:
        _append_message(
            session,
            role="assistant",
            content="Dokument został już utworzony z tego planu. Rozpocznij nową sesję, jeśli chcesz inny dokument.",
        )
        return session

    if status == PLANNING_STATUS_APPROVED:
        _append_message(
            session,
            role="assistant",
            content=(
                "Plan jest już zatwierdzony. Możesz utworzyć dokument albo dopisać zmianę, "
                "a ja wrócę do etapu zatwierdzenia."
            ),
        )
        session["status"] = PLANNING_STATUS_READY_FOR_APPROVAL
        _apply_refinement(session, message)
        return session

    if status == PLANNING_STATUS_COLLECTING:
        _apply_step_answer(session, message)
        _advance_step(session)
        if int(session.get("step_index", 0)) >= len(PLANNING_STEPS):
            _prepare_approval(session)
            return session
        next_question = PLANNING_STEPS[int(session["step_index"])][1]
        _append_message(session, role="assistant", content=next_question)
        return session

    # READY_FOR_APPROVAL
    _apply_refinement(session, message)
    return session


def planning_session_approve(*, session: dict, approved_points: list[str]) -> dict:
    if str(session.get("status")) == PLANNING_STATUS_DOCUMENT_CREATED:
        raise ValueError("Document was already created for this session")
    if str(session.get("status")) != PLANNING_STATUS_READY_FOR_APPROVAL:
        raise ValueError("Session is not ready for approval yet")

    suggestions = [str(item).strip() for item in session.get("suggested_points", []) if str(item).strip()]
    if not suggestions:
        raise ValueError("No suggested points to approve")

    selected = [str(item).strip() for item in approved_points if str(item).strip()]
    if not selected:
        selected = suggestions
    selected = _dedupe_keep_order(selected)

    session["approved_points"] = selected
    session["status"] = PLANNING_STATUS_APPROVED
    _append_message(
        session,
        role="assistant",
        content=(
            "Punkty zostały zatwierdzone. Możesz teraz utworzyć dokument. "
            "Jeśli chcesz, dopisz jeszcze jedną korektę przed utworzeniem."
        ),
    )
    return session


def planning_session_build_document_payload(
    *,
    session: dict,
    include_chat_summary: bool,
) -> dict[str, str]:
    status = str(session.get("status", ""))
    if status not in {PLANNING_STATUS_APPROVED, PLANNING_STATUS_DOCUMENT_CREATED}:
        raise ValueError("Session is not approved yet")

    brief = dict(session.get("brief", {}))
    title = _normalize_text(brief.get("title", ""))
    if not title:
        title = _generate_title(brief)

    doc_type = _normalize_text(brief.get("doc_type", "")) or "MANAGEMENT_BRIEF"
    target = _normalize_text(brief.get("target_audience", "")) or "Management"
    language = _normalize_text(brief.get("language", "")) or "pl"
    objective = _normalize_text(brief.get("objective", "")) or "Document objective to be defined."
    tone = _normalize_text(brief.get("tone", "")) or "formal"
    constraints = _normalize_text(brief.get("constraints", ""))

    content = _build_planned_document_content(
        title=title,
        brief=brief,
        approved_points=[str(item) for item in session.get("approved_points", [])],
        include_chat_summary=include_chat_summary,
    )
    return {
        "title": title,
        "doc_type": doc_type,
        "target_audience": target,
        "language": language,
        "objective": objective,
        "tone": tone,
        "constraints": constraints,
        "current_content": content,
    }


def planning_session_mark_document_created(*, session: dict, document_id: str) -> dict:
    session["status"] = PLANNING_STATUS_DOCUMENT_CREATED
    session["created_document_id"] = str(document_id).strip()
    _append_message(
        session,
        role="assistant",
        content=f"Dokument został utworzony: {document_id}",
    )
    return session


def prune_planning_sessions(
    *,
    sessions: dict[str, dict],
    max_age_hours: int = 24,
    max_sessions: int = 200,
) -> None:
    if not sessions:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    stale_ids: list[str] = []
    for session_id, session in sessions.items():
        updated_at = _parse_iso(str(session.get("updated_at", "")))
        if updated_at is None or updated_at < cutoff:
            stale_ids.append(session_id)
    for session_id in stale_ids:
        sessions.pop(session_id, None)

    if len(sessions) <= max_sessions:
        return
    ordered = sorted(
        sessions.items(),
        key=lambda item: str(item[1].get("updated_at", "")),
    )
    overflow = len(sessions) - max_sessions
    for session_id, _ in ordered[:overflow]:
        sessions.pop(session_id, None)


def _build_initial_brief(initial_brief: dict[str, str]) -> dict[str, str]:
    raw = {str(k): str(v).strip() for k, v in initial_brief.items()}
    return {
        "title": raw.get("title", ""),
        "doc_type": raw.get("doc_type", "") or "MANAGEMENT_BRIEF",
        "target_audience": raw.get("target_audience", "") or "Management",
        "language": raw.get("language", "") or "pl",
        "objective": raw.get("objective", ""),
        "tone": raw.get("tone", "") or "formal",
        "constraints": raw.get("constraints", ""),
        "emphasis": "",
        "decisions_needed": "",
        "must_include": "",
    }


def _apply_step_answer(session: dict, message: str) -> None:
    step_index = int(session.get("step_index", 0))
    if step_index < 0 or step_index >= len(PLANNING_STEPS):
        return
    key = PLANNING_STEPS[step_index][0]
    brief = session["brief"]

    if key == "objective":
        brief["objective"] = message
        return
    if key == "emphasis":
        brief["emphasis"] = message
        return
    if key == "decisions_needed":
        brief["decisions_needed"] = message
        return
    if key == "audience_doc_type":
        _apply_audience_doc_type(brief, message)
        return
    if key == "language":
        _apply_language_preference(brief, message)
        return
    if key == "tone_constraints":
        _apply_tone_constraints(brief, message)
        return


def _advance_step(session: dict) -> None:
    current = int(session.get("step_index", 0)) + 1
    session["step_index"] = current


def _prepare_approval(session: dict) -> None:
    suggestions = _build_suggested_points(session["brief"])
    session["suggested_points"] = suggestions
    session["approved_points"] = []
    session["status"] = PLANNING_STATUS_READY_FOR_APPROVAL

    lines = ["Proponowane punkty do zatwierdzenia:"]
    lines.extend([f"- {item}" for item in suggestions])
    lines.append("")
    lines.append("Jeśli OK, kliknij Approve. Jeśli chcesz poprawki, odpisz w czacie.")
    _append_message(session, role="assistant", content="\n".join(lines))


def _apply_refinement(session: dict, message: str) -> None:
    brief = session["brief"]
    lower = message.lower()
    if any(token in lower for token in ("podkre", "emphas", "highlight")):
        brief["emphasis"] = message
    elif any(token in lower for token in ("decyz", "approve", "akcept")):
        brief["decisions_needed"] = message
    elif any(token in lower for token in ("język", "jezyk", "language", "polish", "english", "polski")):
        _apply_language_preference(brief, message)
    elif any(token in lower for token in ("audience", "management", "zarz", "board")):
        _apply_audience_doc_type(brief, message)
    elif any(token in lower for token in ("tone", "formal", "styl", "ograniczen", "constraint")):
        _apply_tone_constraints(brief, message)
    else:
        joined = " ".join(part for part in [brief.get("must_include", ""), message] if part).strip()
        brief["must_include"] = _truncate(joined, max_chars=1200)

    session["status"] = PLANNING_STATUS_READY_FOR_APPROVAL
    session["approved_points"] = []
    session["suggested_points"] = _build_suggested_points(brief)
    _append_message(
        session,
        role="assistant",
        content=(
            "Zaktualizowałem plan i punkty do zatwierdzenia. "
            "Sprawdź listę i kliknij Approve."
        ),
    )


def _build_suggested_points(brief: dict[str, str]) -> list[str]:
    title = _normalize_text(brief.get("title", ""))
    if not title:
        title = _generate_title(brief)
    suggestions = [
        f"Tytuł dokumentu: {title}",
        f"Cel: {_normalize_text(brief.get('objective', '')) or 'Do doprecyzowania'}",
        f"Priorytet do podkreślenia: {_normalize_text(brief.get('emphasis', '')) or 'Do doprecyzowania'}",
        f"Odbiorca/typ: {brief.get('target_audience', 'Management')} / {brief.get('doc_type', 'MANAGEMENT_BRIEF')}",
        f"Język dokumentu: {brief.get('language', 'pl')}",
        f"Decyzje do zatwierdzenia: {_normalize_text(brief.get('decisions_needed', '')) or 'Do doprecyzowania'}",
        f"Ton i ograniczenia: {brief.get('tone', 'formal')} | {_normalize_text(brief.get('constraints', '')) or 'brak dodatkowych'}",
    ]
    must_include = _normalize_text(brief.get("must_include", ""))
    if must_include:
        suggestions.append(f"Dodatkowo uwzględnić: {must_include}")
    return _dedupe_keep_order([_truncate(item, max_chars=420) for item in suggestions if item.strip()])


def _apply_audience_doc_type(brief: dict[str, str], message: str) -> None:
    lowered = message.lower()
    if any(token in lowered for token in ("zarząd", "zarzad", "board")):
        brief["target_audience"] = "Board"
    elif "leadership" in lowered:
        brief["target_audience"] = "Leadership Team"
    elif "management" in lowered or "manager" in lowered:
        brief["target_audience"] = "Management"

    doc_type_map = [
        ("qbr", "QBR"),
        ("test", "TEST_SUMMARY"),
        ("strategy", "STRATEGY_PAPER"),
        ("strateg", "STRATEGY_PAPER"),
        ("documentation", "DOCUMENTATION"),
        ("dokumentac", "DOCUMENTATION"),
        ("charter", "PROJECT_CHARTER"),
        ("brief", "MANAGEMENT_BRIEF"),
    ]
    for token, doc_type in doc_type_map:
        if token in lowered:
            brief["doc_type"] = doc_type
            break

    if re.search(r"\b(english|en)\b", lowered):
        brief["language"] = "en"
    if re.search(r"\b(polish|polski|pl)\b", lowered):
        brief["language"] = "pl"


def _apply_tone_constraints(brief: dict[str, str], message: str) -> None:
    lowered = message.lower()
    if "formal" in lowered or "formalny" in lowered:
        brief["tone"] = "formal"
    elif "technical" in lowered or "technicz" in lowered:
        brief["tone"] = "technical"
    elif "executive" in lowered:
        brief["tone"] = "executive"
    elif "informal" in lowered or "luź" in lowered or "luz" in lowered:
        brief["tone"] = "informal"
    brief["constraints"] = _truncate(message, max_chars=1200)


def _apply_language_preference(brief: dict[str, str], message: str) -> None:
    lowered = message.lower()
    if re.search(r"\b(english|en)\b", lowered):
        brief["language"] = "en"
        return
    if re.search(r"\b(polish|polski|pl)\b", lowered):
        brief["language"] = "pl"
        return


def _build_planned_document_content(
    *,
    title: str,
    brief: dict[str, str],
    approved_points: list[str],
    include_chat_summary: bool,
) -> str:
    objective = _normalize_text(brief.get("objective", ""))
    emphasis = _normalize_text(brief.get("emphasis", ""))
    decisions = _normalize_text(brief.get("decisions_needed", ""))
    must_include = _normalize_text(brief.get("must_include", ""))
    constraints = _normalize_text(brief.get("constraints", ""))

    lines = [
        f"# {title}",
        "",
        "## Purpose",
        objective or "Define document objective and expected business outcome.",
        "",
        "## What To Emphasize",
        f"- {emphasis or 'Key outcomes, risks, and decisions for leadership.'}",
        "",
        "## Decisions Needed",
        f"- {decisions or 'Confirm priorities, ownership, and timeline.'}",
        "",
        "## Constraints",
        f"- {constraints or 'Keep concise, decision-ready, and specific.'}",
    ]
    if must_include:
        lines.extend(["", "## Must Include", f"- {must_include}"])
    if approved_points:
        lines.extend(["", "## Approved Planning Points"])
        lines.extend([f"- {point}" for point in approved_points])

    if include_chat_summary:
        lines.extend(["", "## Planning Chat Summary"])
        lines.extend(_render_chat_summary_lines(brief))

    return "\n".join(lines).strip()


def _render_chat_summary_lines(brief: dict[str, str]) -> list[str]:
    summary = [
        f"- Audience: {brief.get('target_audience', 'Management')}",
        f"- Type: {brief.get('doc_type', 'MANAGEMENT_BRIEF')}",
        f"- Language: {brief.get('language', 'pl')}",
        f"- Tone: {brief.get('tone', 'formal')}",
    ]
    return summary


def _generate_title(brief: dict[str, str]) -> str:
    objective = _normalize_text(brief.get("objective", "")).lower()
    emphasis = _normalize_text(brief.get("emphasis", "")).lower()
    merged = f"{objective} {emphasis}".strip()
    language = _normalize_text(brief.get("language", "")).lower() or "pl"
    audience = _normalize_text(brief.get("target_audience", "")) or "Management"

    has_seo = "seo" in merged
    has_geo = "geo" in merged
    year_match = re.search(r"\b(20\d{2})\b", merged)
    year = year_match.group(1) if year_match else ""

    if language.startswith("pl"):
        if has_seo and has_geo:
            base = "Plan SEO i GEO"
        elif has_seo:
            base = "Plan SEO"
        elif has_geo:
            base = "Plan GEO"
        else:
            base = "Plan strategiczny"
        if year:
            base = f"{base} na {year}"
        return f"{base} dla {audience}".strip()

    if has_seo and has_geo:
        base = "SEO and GEO Plan"
    elif has_seo:
        base = "SEO Plan"
    elif has_geo:
        base = "GEO Plan"
    else:
        base = "Strategic Plan"
    if year:
        base = f"{base} {year}"
    return f"{base} for {audience}".strip()


def _append_message(session: dict, *, role: str, content: str) -> None:
    now = _now_iso()
    session.setdefault("messages", []).append(
        {
            "role": role,
            "content": _truncate(_normalize_text(content), max_chars=8000),
            "created_at": now,
        }
    )
    session["updated_at"] = now


def _normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\r\n?", "\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _truncate(value: str, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
