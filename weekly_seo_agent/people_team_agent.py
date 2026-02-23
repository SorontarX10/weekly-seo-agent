from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from weekly_seo_agent.people_manager_agent import (
    NoteDoc,
    PeopleNotesClient,
    _build_heuristic_assessment,
    _coerce_bullet_only_report,
    _detect_yearly_review_triggers,
    _enforce_yearly_review_first_topic,
    _llm_assessment,
    _normalize_text,
    _parse_iso_datetime,
    _resolve_credentials_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Team manager agent: generate per-person 1:1/review plans and upload "
            "Google Docs to SEO Team Data/<today>."
        )
    )
    parser.add_argument(
        "--notes-folder-reference",
        default=os.environ.get("PEOPLE_NOTES_FOLDER_REFERENCE", ""),
        help="Google Drive folder URL/ID containing person docs.",
    )
    parser.add_argument(
        "--status-sheet-reference",
        default=(
            os.environ.get("PEOPLE_MANAGER_STATUS_SHEET_REFERENCE", "").strip()
            or os.environ.get("STATUS_FILE_REFERENCE", "").strip()
        ),
        help="Google Sheets URL/ID with status rows.",
    )
    parser.add_argument(
        "--exclude",
        default=os.environ.get("PEOPLE_TEAM_EXCLUDE", "Roksana"),
        help="Comma-separated first names to exclude.",
    )
    parser.add_argument(
        "--max-docs-per-person",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_MAX_DOCS", "12") or "12"),
        help="Max docs analyzed per person.",
    )
    parser.add_argument(
        "--status-max-topics",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_STATUS_MAX_TOPICS", "10") or "10"),
        help="Max status topics merged per person.",
    )
    parser.add_argument(
        "--status-max-sheets",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_STATUS_MAX_SHEETS", "8") or "8"),
        help="Max tabs read from status sheet.",
    )
    parser.add_argument(
        "--status-max-rows-per-sheet",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_STATUS_MAX_ROWS_PER_SHEET", "400") or "400"),
        help="Max rows read per status tab.",
    )
    parser.add_argument(
        "--status-llm-name-mapping",
        action="store_true",
        default=os.environ.get("PEOPLE_MANAGER_STATUS_LLM_NAME_MAPPING", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Use LLM for matching first name to full-name status rows.",
    )
    parser.add_argument(
        "--no-status-llm-name-mapping",
        action="store_true",
        help="Disable LLM name mapping for status rows.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable narrative LLM refinement (heuristic only).",
    )
    parser.add_argument(
        "--drive-root-folder",
        default=os.environ.get("PEOPLE_TEAM_OUTPUT_ROOT_FOLDER", "SEO Team Data"),
        help="Root Drive folder where date subfolder is created.",
    )
    parser.add_argument(
        "--run-date",
        default="",
        help="YYYY-MM-DD date label for output subfolder (default: today).",
    )
    parser.add_argument(
        "--local-output-dir",
        default="outputs/people_manager/team_plans",
        help="Local fallback/report copy directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate local markdown only, do not upload to Drive.",
    )
    return parser.parse_args()


def _parse_run_date(value: str) -> date:
    if not value.strip():
        return date.today()
    return date.fromisoformat(value.strip())


def _infer_person_from_name(doc_name: str) -> str:
    normalized = re.sub(r"\s+", " ", doc_name.strip())
    if not normalized:
        return ""
    token = re.split(r"[\s\-_/]+", normalized, maxsplit=1)[0].strip()
    return token


def _collect_docs_by_person(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        name = str(row.get("name", "")).strip()
        person = _infer_person_from_name(name)
        if not person:
            continue
        grouped[person].append(row)

    for person in list(grouped.keys()):
        grouped[person].sort(
            key=lambda r: _parse_iso_datetime(r.get("modified_time", "")) or datetime.min,
            reverse=True,
        )
    return grouped


def main() -> None:
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    args = _parse_args()
    run_day = _parse_run_date(args.run_date)
    date_label = run_day.isoformat()

    credentials_path = _resolve_credentials_path()
    if not credentials_path:
        raise SystemExit(
            "Google credentials missing. Set PEOPLE_MANAGER_GOOGLE_CREDENTIALS_PATH "
            "or GOOGLE_DRIVE_CLIENT_SECRET_PATH."
        )

    token_path = os.environ.get("GOOGLE_DRIVE_TOKEN_PATH", ".google_drive_token.json").strip()
    client = PeopleNotesClient(credentials_path=credentials_path, token_path=token_path)

    if not args.notes_folder_reference.strip():
        raise SystemExit("Provide --notes-folder-reference or PEOPLE_NOTES_FOLDER_REFERENCE.")

    all_docs = client.list_docs_in_folder(args.notes_folder_reference, max_docs=500)
    grouped = _collect_docs_by_person(all_docs)

    excluded = {
        _normalize_text(item)
        for item in [chunk.strip() for chunk in args.exclude.split(",") if chunk.strip()]
    }
    people = [p for p in grouped.keys() if _normalize_text(p) not in excluded]
    people.sort(key=lambda x: _normalize_text(x))

    if not people:
        raise SystemExit("No people found after applying excludes.")

    local_dir = Path(args.local_output_dir) / date_label.replace("-", "_")
    local_dir.mkdir(parents=True, exist_ok=True)

    output_folder_id = ""
    if not args.dry_run:
        output_folder_id = client.ensure_output_folder(args.drive_root_folder, date_label)

    uploaded = 0
    generated = 0

    for person in people:
        selected_rows = grouped.get(person, [])[: max(1, args.max_docs_per_person)]
        notes: list[NoteDoc] = []
        for row in selected_rows:
            doc_id = row.get("id", "")
            if not doc_id:
                continue
            try:
                text = client.extract_doc_text(doc_id, max_chars=12000)
            except Exception:
                continue
            if not text:
                continue
            notes.append(
                NoteDoc(
                    id=doc_id,
                    name=row.get("name", ""),
                    modified_time=row.get("modified_time", ""),
                    web_view_link=row.get("web_view_link", ""),
                    text=text,
                )
            )

        status_topics: list[str] = []
        if args.status_sheet_reference.strip():
            try:
                status_topics = client.collect_status_topics(
                    status_sheet_reference=args.status_sheet_reference,
                    person_name=person,
                    max_sheets=max(1, args.status_max_sheets),
                    max_rows_per_sheet=max(1, args.status_max_rows_per_sheet),
                    max_topics=max(1, args.status_max_topics),
                    use_llm_name_mapping=bool(args.status_llm_name_mapping)
                    and not bool(args.no_status_llm_name_mapping),
                )
            except Exception:
                status_topics = []

        if not notes and not status_topics:
            print(f"Skip {person}: no notes/status rows.")
            continue

        report = _build_heuristic_assessment(
            person,
            notes,
            status_topics=status_topics,
            status_sheet_reference=args.status_sheet_reference,
        )
        yearly_review_triggers = _detect_yearly_review_triggers(notes, status_topics)
        report = _enforce_yearly_review_first_topic(report, yearly_review_triggers)

        if not args.no_llm:
            try:
                report = _llm_assessment(
                    person,
                    notes,
                    report,
                    status_topics=status_topics,
                    yearly_review_triggers=yearly_review_triggers,
                )
                report = _enforce_yearly_review_first_topic(report, yearly_review_triggers)
                report = _coerce_bullet_only_report(report)
            except Exception:
                pass

        local_path = local_dir / f"{person}.md"
        local_path.write_text(report, encoding="utf-8")
        generated += 1

        if not args.dry_run:
            uploaded_meta = client.create_google_doc_in_folder(
                folder_id=output_folder_id,
                title=person,
                content=report,
                replace_existing=True,
            )
            uploaded += 1
            print(f"Uploaded: {person} | {uploaded_meta.get('webViewLink', '')}")
        else:
            print(f"Generated (dry-run): {person} | {local_path}")

    print(f"Done. Generated={generated}, Uploaded={uploaded}, DateFolder={date_label}")


if __name__ == "__main__":
    main()
