from weekly_seo_agent.people_team_agent import _collect_docs_by_person, _infer_person_from_name


def test_infer_person_from_name() -> None:
    assert _infer_person_from_name("Adam") == "Adam"
    assert _infer_person_from_name("Asia - 1on1") == "Asia"
    assert _infer_person_from_name("  Przemek_notes ") == "Przemek"


def test_collect_docs_groups_by_person() -> None:
    rows = [
        {"name": "Adam", "modified_time": "2026-02-10T08:00:00Z"},
        {"name": "Adam - old", "modified_time": "2026-02-01T08:00:00Z"},
        {"name": "Asia", "modified_time": "2026-02-09T08:00:00Z"},
    ]
    grouped = _collect_docs_by_person(rows)
    assert set(grouped.keys()) == {"Adam", "Asia"}
    assert len(grouped["Adam"]) == 2
    assert grouped["Adam"][0]["modified_time"] > grouped["Adam"][1]["modified_time"]
