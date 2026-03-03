from weekly_seo_agent.reporting import _enforce_section_line_limits


def test_section_compaction_does_not_append_technical_marker() -> None:
    lines = [
        "## Executive summary",
        "- line 1",
        "- line 2",
        "- line 3",
        "- line 4",
        "- line 5",
        "- line 6",
        "- line 7",
        "",
        "## What is happening and why",
        "- line 1",
    ]

    compacted = _enforce_section_line_limits(lines)
    joined = "\n".join(compacted)

    assert "Auto-compressed for readability" not in joined
