from __future__ import annotations

import argparse
import re
from datetime import date

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from weekly_seo_agent.weekly_reporting_agent.clients.gmail_client import GmailClient
from weekly_seo_agent.weekly_reporting_agent.config import AgentConfig
from weekly_seo_agent.weekly_reporting_agent.llm import build_gaia_llm
from weekly_seo_agent.weekly_reporting_agent.time_windows import compute_windows
from weekly_seo_agent.weekly_reporting_agent.weekly_news import NewsItem, collect_weekly_news


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly SEO + GEO news summary agent")
    parser.add_argument(
        "--run-date",
        dest="run_date",
        help="Execution date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send email, just print the summary.",
    )
    return parser.parse_args()


def _parse_run_date(raw: str | None) -> date:
    if not raw:
        return date.today()
    return date.fromisoformat(raw)


def _format_item(item: NewsItem) -> str:
    date_label = item.published.isoformat() if item.published else "unknown date"
    source_label = item.source or item.domain or "unknown source"
    summary = re.sub(r"\s+", " ", (item.summary or "")).strip()
    lines = [
        f"- {date_label} | {item.title}",
        f"  Source: {source_label}",
        f"  Link: {item.url}",
        f"  Why it matters: {summary}",
    ]
    return "\n".join(lines)


def _fallback_summary(seo_items: list[NewsItem], geo_items: list[NewsItem]) -> str:
    def section(title: str, items: list[NewsItem], max_display: int) -> list[str]:
        lines: list[str] = [f"{title}"]
        if not items:
            lines.append("- Brak istotnych publikacji w tym tygodniu.")
            return lines

        display = items[:max_display]
        for item in display:
            lines.append(_format_item(item))
        hidden = max(0, len(items) - len(display))
        if hidden:
            lines.append(f"- (+{hidden} więcej pozycji pominiętych dla czytelności)")
        return lines

    lines: list[str] = []
    lines.extend(section("SEO", seo_items, max_display=8))
    lines.append("")
    lines.extend(section("GEO", geo_items, max_display=6))
    return "\n".join(lines).strip()


def _llm_summary(config: AgentConfig, seo_items: list[NewsItem], geo_items: list[NewsItem]) -> str:
    llm = build_gaia_llm(config)

    def item_block(items: list[NewsItem]) -> str:
        if not items:
            return "No items."
        return "\n".join(
            f"{item.published.isoformat() if item.published else 'unknown'} | "
            f"{item.title} | {item.domain} | {item.url} | {item.summary}"
            for item in items
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Jestes starszym analitykiem SEO. "
                    "Podsumuj kluczowe newsy z ostatniego pelnego tygodnia. "
                    "Zasady: nie wymyslaj faktow, nie dopisuj danych spoza listy. "
                    "Uzyj jezyka polskiego. "
                    "Skup sie na tym, co ma wplyw na SEO lub GEO, i dodaj 1 zdanie "
                    "o potencjalnym znaczeniu dla strategii. "
                    "Uzyj CZYSTEGO TEKSTU. Zachowaj bardzo czytelny i krotki format. "
                    "Maks: 8 pozycji SEO i 6 pozycji GEO. "
                    "Format dokladnie:\n"
                    "SEO\n"
                    "- <data> | <tytul>\n"
                    "  Source: <nazwa zrodla>\n"
                    "  Link: <URL>\n"
                    "  Why it matters: <1 zdanie>\n"
                    "\n"
                    "GEO\n"
                    "- <data> | <tytul>\n"
                    "  Source: <nazwa zrodla>\n"
                    "  Link: <URL>\n"
                    "  Why it matters: <1 zdanie>"
                ),
            ),
            (
                "user",
                (
                    "SEO items:\n"
                    f"{item_block(seo_items)}\n\n"
                    "GEO items:\n"
                    f"{item_block(geo_items)}"
                ),
            ),
        ]
    )
    return (prompt | llm | StrOutputParser()).invoke({})


def _sanitize_output(text: str) -> str:
    sanitized = text.replace("**", "").replace("__", "").replace("`", "")
    sanitized = re.sub(r"^#+\s*", "", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"^\s*\*\s+", "- ", sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    sanitized = re.sub(r"^\s*No items\.\s*$", "- Brak istotnych publikacji w tym tygodniu.", sanitized, flags=re.MULTILINE)
    return sanitized.strip()


def build_summary(config: AgentConfig, seo_items: list[NewsItem], geo_items: list[NewsItem]) -> str:
    if config.use_llm_analysis and config.gaia_llm_enabled:
        try:
            return _sanitize_output(_llm_summary(config, seo_items, geo_items))
        except Exception:
            return _fallback_summary(seo_items, geo_items)
    return _fallback_summary(seo_items, geo_items)


def main() -> None:
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    args = _parse_args()
    run_date = _parse_run_date(args.run_date)
    config = AgentConfig.from_env()
    windows = compute_windows(run_date)
    window = windows["current_28d"]

    seo_items, geo_items = collect_weekly_news(
        window=window,
        seo_urls=config.weekly_news_rss_urls_seo,
        geo_urls=config.weekly_news_rss_urls_geo,
        seo_allowlist=config.weekly_news_domains_seo,
        geo_allowlist=config.weekly_news_domains_geo,
        seo_keywords=config.weekly_news_keywords_seo,
        geo_keywords=config.weekly_news_keywords_geo,
        max_items=config.weekly_news_max_items,
    )

    summary = build_summary(config, seo_items, geo_items)
    subject = f"Tygodniowe podsumowanie SEO i GEO: {window.start}–{window.end}"
    body = (
        f"Tygodniowe podsumowanie SEO i GEO (zakres: {window.start}–{window.end}).\n\n"
        f"{summary}\n"
    )

    if args.dry_run or not config.gmail_enabled:
        print(subject)
        print("=")
        print(body)
        if not config.gmail_enabled and not args.dry_run:
            print("\nGMAIL_ENABLED=false, email not sent.")
        return

    client = GmailClient(
        service_account_path=config.gmail_service_account_path,
        delegated_user=config.gmail_delegate_user,
        sender=config.gmail_sender,
        auth_mode=config.gmail_auth_mode,
        oauth_client_secret_path=config.gmail_oauth_client_secret_path,
        oauth_refresh_token=config.gmail_oauth_refresh_token,
        oauth_token_uri=config.gmail_oauth_token_uri,
    )
    client.send_message(
        to_email=config.gmail_recipient,
        subject=subject,
        body=body,
    )
    print(f"Email sent to {config.gmail_recipient}.")


if __name__ == "__main__":
    main()
