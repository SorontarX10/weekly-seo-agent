from __future__ import annotations

import weekly_seo_agent.manager_document_agent.web_research as web_research


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_duckduckgo_search_parses_abstract_and_related(monkeypatch):
    payload = {
        "Heading": "SEO",
        "AbstractText": "Abstract snippet",
        "AbstractURL": "https://example.com/abstract",
        "RelatedTopics": [
            {"Text": "Related one", "FirstURL": "https://example.com/r1"},
            {"Topics": [{"Text": "Nested topic", "FirstURL": "https://example.com/r2"}]},
        ],
    }

    monkeypatch.setattr(
        web_research.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(payload),
    )

    rows = web_research.duckduckgo_search(
        query="seo outlook",
        region="us-en",
        max_results=5,
        timeout_sec=15,
    )
    assert len(rows) == 3
    assert rows[0]["url"] == "https://example.com/abstract"
    assert rows[1]["url"] == "https://example.com/r1"
    assert rows[2]["url"] == "https://example.com/r2"


def test_run_web_research_attaches_playwright_excerpts(monkeypatch):
    payload = {
        "Heading": "SEO",
        "AbstractText": "Abstract snippet",
        "AbstractURL": "https://example.com/abstract",
    }
    class _FakeHTMLResponse:
        text = ""

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url, *args, **kwargs):
        if "api.duckduckgo.com" in url:
            return _FakeResponse(payload)
        return _FakeHTMLResponse()

    monkeypatch.setattr(web_research.requests, "get", _fake_get)

    class _FakeFetcher:
        def fetch_page_text(self, url: str, *, max_chars: int = 1800) -> str:
            return f"Fetched {url} ({max_chars})"

    result = web_research.run_web_research(
        query="seo geo roadmap",
        region="us-en",
        max_results=5,
        fetch_pages=True,
        max_pages=1,
        page_char_limit=900,
        page_fetcher_factory=lambda: _FakeFetcher(),
    )
    assert result["provider"] == "duckduckgo_instant_answer"
    assert result["items"][0]["page_excerpt"].startswith("Fetched https://example.com/abstract")
    assert result["summary_text"].startswith("# Web Research: seo geo roadmap")
    assert "Playwright pages attempted: 1" in result["summary_text"]


def test_run_web_research_uses_html_fallback_and_fetches_pages(monkeypatch):
    instant_payload = {
        "Heading": "",
        "AbstractText": "",
        "AbstractURL": "",
        "Results": [],
        "RelatedTopics": [],
    }
    html_payload = """
    <html><body>
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle-1">Article One</a>
      <a class="result__a" href="https://example.com/article-2">Article Two</a>
    </body></html>
    """

    def _fake_get(url, *args, **kwargs):
        if "api.duckduckgo.com" in url:
            return _FakeResponse(instant_payload)
        return _FakeHTMLResponse(html_payload)

    class _FakeHTMLResponse:
        def __init__(self, payload: str):
            self.text = payload

        def raise_for_status(self) -> None:
            return None

    fetched_urls: list[str] = []

    class _FakeFetcher:
        def fetch_page_text(self, url: str, *, max_chars: int = 1800) -> str:
            fetched_urls.append(url)
            return f"Fetched {url} ({max_chars})"

    monkeypatch.setattr(web_research.requests, "get", _fake_get)

    result = web_research.run_web_research(
        query="seo geo outlook",
        region="us-en",
        max_results=5,
        fetch_pages=True,
        max_pages=2,
        page_char_limit=1200,
        page_fetcher_factory=lambda: _FakeFetcher(),
    )

    assert len(result["items"]) == 2
    assert result["items"][0]["source"] == "duckduckgo_html"
    assert result["items"][0]["page_excerpt"].startswith("Fetched https://example.com/article-1")
    assert "Fallback: DuckDuckGo HTML SERP" in result["summary_text"]
    assert "Playwright pages attempted: 2" in result["summary_text"]
    assert fetched_urls == ["https://example.com/article-1", "https://example.com/article-2"]
