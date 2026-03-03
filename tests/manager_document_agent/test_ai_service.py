from __future__ import annotations

from datetime import datetime, timezone

from weekly_seo_agent.manager_document_agent.ai import (
    AIService,
    OutlineContext,
    _extract_facts_from_summary,
    _load_playbook_grounding,
    _select_context_packs,
    _sanitize_inline_markdown_emphasis,
)
from weekly_seo_agent.manager_document_agent.models import Document, DocumentStatus


_VALID_MANAGEMENT_DOC = """
# SEO & GEO Outlook

## Executive Summary
- Recommendation: prioritize AI-led discovery initiatives with KPI ownership.
- Why now: organic journeys are shifting to answer-driven and zero-click surfaces.
- Impact: protect GMV and improve conversion across strategic intents.

## Decisions Needed from Management
- Confirm ownership and timeline for top initiatives.
- Approve KPI governance and monthly decision cadence.
""".strip()


def _build_document(*, current_content: str = "Current content") -> Document:
    now = datetime.now(timezone.utc)
    return Document(
        id="doc-1",
        title="SEO & GEO Outlook",
        doc_type="MANAGEMENT_BRIEF",
        target_audience="Management",
        language="pl",
        objective="Prepare 2026 direction",
        tone="formal",
        constraints="Keep concise",
        status=DocumentStatus.IN_PROGRESS,
        current_content=current_content,
        last_opened_at=now,
        finalized_at=None,
        archived_at=None,
        created_at=now,
        updated_at=now,
    )


class _CapturingAIService(AIService):
    @staticmethod
    def _build_llm_from_env():
        return object()

    def __init__(self, *, playbook_path: str):
        self.invocations: list[tuple[str, str]] = []
        super().__init__(playbook_path=playbook_path)

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        self.invocations.append((system_prompt, user_prompt))
        if "Rewrite selected fragment." in user_prompt:
            return "Poprawiony fragment dla managementu z mocniejszym framingiem decyzji."
        return _VALID_MANAGEMENT_DOC


def test_load_playbook_grounding_returns_empty_for_missing_file(tmp_path):
    missing = tmp_path / "missing_playbook.md"
    assert _load_playbook_grounding(playbook_path=missing) == ""


def test_outline_prompt_includes_executive_playbook_grounding(tmp_path):
    playbook = tmp_path / "Executive_Playbook_Tech_Company.md"
    playbook.write_text(
        (
            "# Executive Playbook\n"
            "Pyramid Principle: start from recommendation.\n"
            "MECE: keep argument structure mutually exclusive.\n"
        ),
        encoding="utf-8",
    )
    service = _CapturingAIService(playbook_path=str(playbook))
    document = _build_document(current_content="")

    output = service.generate_outline(
        document,
        OutlineContext(
            instructions="Management-ready version",
            attachments_summary="- February 2026 AI Organic Traffic delivered 4.5% of Organic GMV.",
        ),
    )

    assert output.startswith("# SEO & GEO Outlook")
    assert service.invocations
    prompt = service.invocations[-1][1]
    assert "Executive Playbook grounding:" in prompt
    assert "Pyramid Principle" in prompt
    assert "MECE" in prompt


def test_rewrite_prompts_include_executive_playbook_grounding(tmp_path):
    playbook = tmp_path / "Executive_Playbook_Tech_Company.md"
    playbook.write_text(
        (
            "# Executive Playbook\n"
            "Decision memo should contain recommendation, risks and owner.\n"
        ),
        encoding="utf-8",
    )
    service = _CapturingAIService(playbook_path=str(playbook))
    document = _build_document(current_content="# Draft\n\nSome text to rewrite.")

    full = service.rewrite_full(document, "Prepare management-ready structure")
    assert "Executive Summary" in full
    assert service.invocations
    full_prompt = service.invocations[-1][1]
    assert "Executive Playbook grounding:" in full_prompt
    assert "recommendation, risks and owner" in full_prompt

    fragment = service.rewrite_selection(
        selected_text="Some text to rewrite.",
        prompt="Make this more decision-oriented",
        left_context="# Draft\n\n",
        right_context="\n\n## Next steps",
    )
    assert "managementu" in fragment
    selection_prompt = service.invocations[-1][1]
    assert "Executive Playbook grounding:" in selection_prompt
    assert "recommendation, risks and owner" in selection_prompt


def test_sanitize_inline_markdown_emphasis_removes_bold_markers():
    source = (
        "## Strategic Context\n"
        "**Market Shift**: AI answers are growing.\n"
        "- __CPC Inflation__: +20.5%\n"
    )
    sanitized = _sanitize_inline_markdown_emphasis(source)
    assert "**" not in sanitized
    assert "__" not in sanitized
    assert "Market Shift" in sanitized
    assert "CPC Inflation" in sanitized


def test_select_context_packs_includes_beginning_middle_and_end():
    source = (
        "BEGIN " + ("alpha " * 500) +
        "MIDDLE " + ("beta " * 500) +
        "END " + ("omega " * 500)
    )
    selected = _select_context_packs(source, max_chars=2200)
    assert "Context pack (start)" in selected
    assert "Context pack (middle)" in selected
    assert "Context pack (end)" in selected
    assert "additional context packs loaded" in selected


def test_invoke_paginates_when_llm_hits_length_limit(monkeypatch):
    class FakeResponse:
        def __init__(self, content: str, finish_reason: str):
            self.content = content
            self.response_metadata = {"finish_reason": finish_reason}
            self.additional_kwargs = {}

    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list[list] = []

        def invoke(self, messages):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                return FakeResponse(
                    (
                        "# SEO & GEO Outlook 2026\n\n"
                        "## Source Facts From Attachments\n"
                        "- This roadmap is built around five strategic programs:"
                    ),
                    "length",
                )
            return FakeResponse(
                (
                    "five strategic programs:\n"
                    "- I. Technical & Index Foundations\n"
                    "- II. Authority & Brand Leadership"
                ),
                "stop",
            )

    fake_llm = FakeLLM()
    monkeypatch.setattr(
        AIService,
        "_build_llm_from_env",
        staticmethod(lambda: fake_llm),
    )

    service = AIService(playbook_path=None)
    output = service._invoke("system", "user")

    assert len(fake_llm.calls) == 2
    assert output.count("five strategic programs:") == 1
    assert "I. Technical & Index Foundations" in output
    assert "II. Authority & Brand Leadership" in output

    continuation_prompt = fake_llm.calls[1][-1].content
    assert "Continue from exactly where you stopped." in continuation_prompt


def test_extract_facts_from_summary_skips_web_research_noise_and_dangling_lines():
    summary = """
- # Web Research: GEO 2026
- Provider: DuckDuckGo Instant Answer API
- Retrieved results: 0
- Playwright pages attempted: 0
- 2026 Strategic Goal: Protect and grow organic GMV
- This roadmap is built around five strategic programs:
- I. Technical & Index Foundations (Protect the Core)
""".strip()
    facts = _extract_facts_from_summary(summary, limit=8)
    assert "2026 Strategic Goal: Protect and grow organic GMV" in facts
    assert any("Technical & Index Foundations" in fact for fact in facts)
    assert all("Retrieved results" not in fact for fact in facts)
    assert all("Playwright pages attempted" not in fact for fact in facts)
    assert all(not fact.endswith(":") for fact in facts)
