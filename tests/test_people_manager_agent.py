from weekly_seo_agent.people_manager_agent import (
    NoteDoc,
    _build_heuristic_assessment,
    _coerce_bullet_only_report,
    _enforce_yearly_review_first_topic,
    _person_match_score,
    _render_html_for_google_doc,
    _detect_yearly_review_triggers,
    _row_mentions_person,
)


def test_person_match_score_prefers_full_match() -> None:
    strong = _person_match_score("Jan", "2026-02-10 1on1 Jan")
    weak = _person_match_score("Jan", "2026-02-10 1on1 Janina")
    none = _person_match_score("Jan", "2026-02-10 1on1 Anna Nowak")

    assert strong > weak
    assert weak > none
    assert none == 0


def test_heuristic_assessment_contains_manager_sections() -> None:
    notes = [
        NoteDoc(
            id="1",
            name="1on1 - Jan Kowalski",
            modified_time="2026-02-10T08:00:00Z",
            web_view_link="https://docs.google.com/document/d/1/edit",
            text=(
                "Delivered cross-team initiative with clear impact on KPI. "
                "Strong ownership and proactive stakeholder communication. "
                "Blocker: dependency from analytics team delayed execution by 1 week. "
                "Started mentoring junior specialist and led planning session."
            ),
        )
    ]

    report = _build_heuristic_assessment("Jan Kowalski", notes)

    assert "## 2. Zacznij od otwartych tematów z poprzedniego 1:1" in report
    assert "## 3. Agenda 1:1 (prowadzenie rozmowy)" in report
    assert "## 4. Punkty do performance review (ściągawka managera)" in report
    assert "## 6. Decyzja o gotowości do awansu" in report
    assert "Status gotowości do awansu" in report
    assert "## Źródła" in report


def test_row_mentions_person_word_boundary() -> None:
    assert _row_mentions_person(["Owner", "Jan Kowalski", "Blocker in analytics"], "Jan")
    assert not _row_mentions_person(["Owner", "Janina Kowalska", "Blocked"], "Jan")


def test_heuristic_assessment_includes_status_topics() -> None:
    notes = [
        NoteDoc(
            id="1",
            name="1on1 - Jan",
            modified_time="2026-02-10T08:00:00Z",
            web_view_link="https://docs.google.com/document/d/1/edit",
            text="Delivered initiative with impact.",
        )
    ]
    status_topics = ["[Status/2026] Task: Canonical fix | Status: blocked by analytics"]

    report = _build_heuristic_assessment(
        "Jan",
        notes,
        status_topics=status_topics,
        status_sheet_reference="https://docs.google.com/spreadsheets/d/fake/edit",
    )

    assert "Tematy ze statusu:" in report
    assert "Status-log:" in report
    assert "Status sheet:" in report


def test_render_html_for_google_doc_has_headings_and_bold() -> None:
    src = (
        "# Manager Support Report: Asia\n\n"
        "## Performance Snapshot\n"
        "- Readiness sygnal (promocja): **Wysoka**\n"
        "- Liczba przeanalizowanych notatek: 1\n"
    )
    html = _render_html_for_google_doc(src)
    assert "<h1>Manager Support Report: Asia</h1>" in html
    assert "<h2>Performance Snapshot</h2>" in html
    assert "<li><strong>Readiness sygnal (promocja):</strong> <strong>Wysoka</strong></li>" in html


def test_yearly_review_trigger_forces_first_1on1_topic() -> None:
    notes = [
        NoteDoc(
            id="1",
            name="Asia",
            modified_time="2026-02-10T08:00:00Z",
            web_view_link="https://docs.google.com/document/d/1/edit",
            text="To jest yearly review i roczne podsumowanie.",
        )
    ]
    status_topics: list[str] = []
    report = _build_heuristic_assessment("Asia", notes, status_topics=status_topics)
    triggers = _detect_yearly_review_triggers(notes, status_topics)
    report = _enforce_yearly_review_first_topic(report, triggers)

    section_start = report.find("## 3. Agenda 1:1 (prowadzenie rozmowy)")
    assert section_start >= 0
    snippet = report[section_start : section_start + 250]
    assert "Ocena roczna/półroczna: zacznij spotkanie od omówienia oceny rocznej" in snippet


def test_half_year_phrase_without_review_does_not_trigger_yearly_mode() -> None:
    notes = [
        NoteDoc(
            id="1",
            name="Asia",
            modified_time="2026-02-10T08:00:00Z",
            web_view_link="https://docs.google.com/document/d/1/edit",
            text="Rozmowa na temat promocji w połowie roku i etapów realizacji.",
        )
    ]
    triggers = _detect_yearly_review_triggers(notes, [])
    assert triggers == []


def test_bullet_only_coercion() -> None:
    raw = "Naglowek\nAkapit 1\n1. Punkt\n- Inny punkt\n"
    coerced = _coerce_bullet_only_report(raw)
    assert "- Naglowek" in coerced
    assert "- Akapit 1" in coerced
    assert "- Punkt" in coerced
