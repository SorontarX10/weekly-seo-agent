from weekly_seo_agent.people_manager_agent import (
    NoteDoc,
    _build_heuristic_assessment,
    _person_match_score,
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

    assert "## Tematy na 1:1" in report
    assert "## Tematy do Performance Review" in report
    assert "## Gotowosc do awansu" in report
    assert "Readiness sygnal" in report
    assert "## Zrodla" in report


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

    assert "## Tematy ze statusu" in report
    assert "Status-log:" in report
    assert "status sheet:" in report
