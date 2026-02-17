from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import find_dotenv, load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from weekly_seo_agent.config import AgentConfig
from weekly_seo_agent.llm import build_gaia_llm


@dataclass
class NoteDoc:
    id: str
    name: str
    modified_time: str
    web_view_link: str
    text: str


class PeopleNotesClient:
    SCOPES = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/documents.readonly",
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    ]
    DOC_MIME = "application/vnd.google-apps.document"

    def __init__(self, credentials_path: str, token_path: str) -> None:
        self.credentials_path = credentials_path.strip()
        self.token_path = token_path.strip() or ".google_drive_token.json"
        self._drive = None
        self._docs = None
        self._sheets = None

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _extract_drive_id(reference: str) -> str:
        raw = reference.strip()
        if not raw:
            return ""
        if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", raw):
            return raw

        parsed = urlparse(raw)
        if not parsed.scheme and not parsed.netloc:
            return ""

        patterns = (
            r"/folders/([a-zA-Z0-9_-]{20,})",
            r"/d/([a-zA-Z0-9_-]{20,})",
        )
        for pattern in patterns:
            match = re.search(pattern, parsed.path)
            if match:
                return match.group(1)

        query = parse_qs(parsed.query)
        if "id" in query and query["id"]:
            candidate = query["id"][0].strip()
            if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", candidate):
                return candidate
        return ""

    def _load_credentials(self) -> Credentials:
        secret_path = Path(self.credentials_path)
        if not secret_path.exists():
            raise RuntimeError(
                "Google credentials file not found: " f"{self.credentials_path}"
            )

        payload = json.loads(secret_path.read_text(encoding="utf-8"))
        if payload.get("type") == "service_account":
            return service_account.Credentials.from_service_account_file(
                str(secret_path),
                scopes=self.SCOPES,
            )

        creds: Credentials | None = None
        token_file = Path(self.token_path)
        if token_file.exists():
            creds = Credentials.from_authorized_user_file(str(token_file), self.SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), self.SCOPES)
            creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

        token_file.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def _drive_service(self):
        if self._drive is None:
            self._drive = build(
                "drive",
                "v3",
                credentials=self._load_credentials(),
                cache_discovery=False,
            )
        return self._drive

    def _docs_service(self):
        if self._docs is None:
            self._docs = build(
                "docs",
                "v1",
                credentials=self._load_credentials(),
                cache_discovery=False,
            )
        return self._docs

    def _sheets_service(self):
        if self._sheets is None:
            self._sheets = build(
                "sheets",
                "v4",
                credentials=self._load_credentials(),
                cache_discovery=False,
            )
        return self._sheets

    def list_docs_in_folder(self, folder_reference: str, max_docs: int) -> list[dict[str, str]]:
        folder_id = self._extract_drive_id(folder_reference)
        if not folder_id:
            raise RuntimeError("Could not parse Google Drive folder ID.")

        drive = self._drive_service()
        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and mimeType='{self.DOC_MIME}'"
        )
        response = drive.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name,mimeType,modifiedTime,webViewLink)",
            pageSize=min(max(1, max_docs * 3), 500),
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            orderBy="modifiedTime desc",
        ).execute()

        rows: list[dict[str, str]] = []
        for row in response.get("files", []):
            doc_id = str(row.get("id", "")).strip()
            if not doc_id:
                continue
            rows.append(
                {
                    "id": doc_id,
                    "name": str(row.get("name", "")).strip(),
                    "modified_time": str(row.get("modifiedTime", "")).strip(),
                    "web_view_link": str(row.get("webViewLink", "")).strip(),
                }
            )
        return rows[:max_docs]

    def get_doc_meta(self, doc_reference: str) -> dict[str, str]:
        doc_id = self._extract_drive_id(doc_reference)
        if not doc_id:
            raise RuntimeError(f"Could not parse Google Doc ID from reference: {doc_reference}")

        drive = self._drive_service()
        row = drive.files().get(
            fileId=doc_id,
            fields="id,name,mimeType,modifiedTime,webViewLink",
            supportsAllDrives=True,
        ).execute()

        mime = str(row.get("mimeType", "")).strip()
        if mime and mime != self.DOC_MIME:
            raise RuntimeError(f"Reference is not a Google Doc: {doc_reference}")

        return {
            "id": str(row.get("id", "")).strip(),
            "name": str(row.get("name", "")).strip(),
            "modified_time": str(row.get("modifiedTime", "")).strip(),
            "web_view_link": str(row.get("webViewLink", "")).strip(),
        }

    def _collect_text_from_elements(
        self,
        elements: list[dict[str, Any]],
        chunks: list[str],
        max_chars: int,
    ) -> None:
        if sum(len(part) for part in chunks) >= max_chars:
            return

        for element in elements:
            if sum(len(part) for part in chunks) >= max_chars:
                return

            paragraph = element.get("paragraph")
            if isinstance(paragraph, dict):
                line_parts: list[str] = []
                for item in paragraph.get("elements", []):
                    if not isinstance(item, dict):
                        continue
                    text_run = item.get("textRun")
                    if not isinstance(text_run, dict):
                        continue
                    line_parts.append(str(text_run.get("content", "")))
                line = re.sub(r"\s+", " ", "".join(line_parts)).strip()
                if line:
                    chunks.append(line)

            table = element.get("table")
            if isinstance(table, dict):
                for row in table.get("tableRows", []):
                    if not isinstance(row, dict):
                        continue
                    for cell in row.get("tableCells", []):
                        if not isinstance(cell, dict):
                            continue
                        nested = cell.get("content")
                        if isinstance(nested, list):
                            self._collect_text_from_elements(nested, chunks, max_chars)

            toc = element.get("tableOfContents")
            if isinstance(toc, dict):
                nested = toc.get("content")
                if isinstance(nested, list):
                    self._collect_text_from_elements(nested, chunks, max_chars)

    def extract_doc_text(self, document_id: str, max_chars: int = 10000) -> str:
        docs = self._docs_service()
        payload = docs.documents().get(documentId=document_id).execute()
        body = payload.get("body", {})
        content = body.get("content", []) if isinstance(body, dict) else []

        chunks: list[str] = []
        if isinstance(content, list):
            self._collect_text_from_elements(content, chunks, max_chars)

        merged = "\n".join(chunks)
        merged = re.sub(r"\n{3,}", "\n\n", merged)
        merged = merged.strip()
        return merged[:max_chars]

    def collect_status_topics(
        self,
        status_sheet_reference: str,
        person_name: str,
        max_sheets: int = 8,
        max_rows_per_sheet: int = 400,
        max_topics: int = 10,
        use_llm_name_mapping: bool = True,
    ) -> list[str]:
        spreadsheet_id = self._extract_drive_id(status_sheet_reference)
        if not spreadsheet_id:
            return []

        sheets = self._sheets_service()
        meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_rows = meta.get("sheets", []) if isinstance(meta, dict) else []

        titles: list[str] = []
        for row in sheet_rows:
            if not isinstance(row, dict):
                continue
            props = row.get("properties")
            if not isinstance(props, dict):
                continue
            title = str(props.get("title", "")).strip()
            if title:
                titles.append(title)
        titles = titles[: max(1, max_sheets)]

        topics: list[str] = []
        llm_rows: list[dict[str, str]] = []
        seen: set[str] = set()
        row_idx = 0
        for title in titles:
            safe_title = title.replace("'", "''")
            range_name = f"'{safe_title}'!A1:ZZ"
            try:
                payload = sheets.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    majorDimension="ROWS",
                ).execute()
            except Exception:
                continue

            values = payload.get("values", []) if isinstance(payload, dict) else []
            if not isinstance(values, list) or not values:
                continue

            headers = [str(cell).strip() for cell in values[0]]
            for row_cells in values[1 : max_rows_per_sheet + 1]:
                if not isinstance(row_cells, list):
                    continue
                row_idx += 1
                topic = _format_status_row(title, headers, row_cells)
                if _row_mentions_person(row_cells, person_name):
                    topic_key = _normalize_text(topic)
                    if not topic_key or topic_key in seen:
                        continue
                    seen.add(topic_key)
                    topics.append(topic)
                    if len(topics) >= max_topics:
                        return topics
                if use_llm_name_mapping:
                    llm_rows.append({"idx": str(row_idx), "text": topic})

        if topics or not use_llm_name_mapping:
            return topics

        try:
            selected_ids = set(_llm_pick_status_rows(person_name, llm_rows, max_topics))
        except Exception:
            return []

        for row in llm_rows:
            try:
                idx_value = int(row.get("idx", "0"))
            except Exception:
                continue
            if idx_value not in selected_ids:
                continue
            topic = row.get("text", "").strip()
            topic_key = _normalize_text(topic)
            if not topic_key or topic_key in seen:
                continue
            seen.add(topic_key)
            topics.append(topic)
            if len(topics) >= max_topics:
                break
        return topics


def _normalize_text(value: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"\s+", " ", ascii_text).strip()


def _parse_iso_datetime(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _person_match_score(person_name: str, doc_name: str) -> int:
    person = _normalize_text(person_name)
    doc = _normalize_text(doc_name)
    if not person or not doc:
        return 0

    tokens = [token for token in person.split(" ") if len(token) >= 2]
    if not tokens:
        return 0

    padded_doc = f" {doc} "
    score = 0
    for token in tokens:
        if re.search(rf"\b{re.escape(token)}\b", padded_doc):
            score += 2 if len(token) >= 4 else 1

    compact_person = person.replace(" ", "")
    compact_doc = doc.replace(" ", "")
    if compact_person and compact_person in compact_doc:
        score += 3

    return score


def _split_candidate_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        cleaned = re.sub(r"\s+", " ", raw).strip("-*\t ")
        if len(cleaned) < 25:
            continue
        lines.append(cleaned)

    if not lines:
        for raw in re.split(r"(?<=[.!?])\s+", text):
            cleaned = re.sub(r"\s+", " ", raw).strip()
            if len(cleaned) < 25:
                continue
            lines.append(cleaned)

    dedup: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = _normalize_text(line)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(line)
    return dedup


def _row_mentions_person(cells: list[object], person_name: str) -> bool:
    person = _normalize_text(person_name)
    if not person:
        return False
    tokens = [token for token in person.split(" ") if len(token) >= 2]
    if not tokens:
        return False
    blob = " ".join(_normalize_text(str(cell)) for cell in cells if str(cell).strip())
    if not blob:
        return False
    for token in tokens:
        if re.search(rf"\b{re.escape(token)}\b", blob):
            return True
    return False


def _format_status_row(sheet_title: str, headers: list[str], row_cells: list[object]) -> str:
    pairs: list[str] = []
    max_cols = min(max(len(headers), len(row_cells)), 8)
    for idx in range(max_cols):
        header = headers[idx].strip() if idx < len(headers) else f"col_{idx+1}"
        value = str(row_cells[idx]).strip() if idx < len(row_cells) else ""
        if not value:
            continue
        header_norm = _normalize_text(header)
        if header_norm in {"imie", "imie i nazwisko", "owner", "osoba"}:
            continue
        pairs.append(f"{header}: {value}")
    summary = " | ".join(pairs) if pairs else " | ".join(str(cell).strip() for cell in row_cells if str(cell).strip())
    summary = re.sub(r"\s+", " ", summary).strip()
    return f"[Status/{sheet_title}] {summary}"


def _llm_pick_status_rows(person_name: str, rows: list[dict[str, str]], max_topics: int) -> list[int]:
    if not rows:
        return []
    config = AgentConfig.from_env()
    llm = build_gaia_llm(config)
    packed_rows = "\n".join(f"{row['idx']}. {row['text'][:420]}" for row in rows[:120])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Otrzymasz liste wierszy statusowych. "
                    "Wskaz tylko te, ktore dotycza wskazanej osoby. "
                    "Dopuszczalne dopasowania: imie, imie+nazwisko, skroty, literowki. "
                    "Jesli brak pewnosci, pomijaj. "
                    "Zwroc TYLKO JSON: {\"row_ids\":[1,2,...]}."
                ),
            ),
            (
                "user",
                (
                    f"Osoba docelowa: {person_name}\n"
                    f"Maksymalnie wybierz: {max(1, max_topics)}\n\n"
                    "Wiersze:\n"
                    f"{packed_rows}"
                ),
            ),
        ]
    )
    raw = (prompt | llm | StrOutputParser()).invoke({})
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return []
    row_ids = payload.get("row_ids")
    if not isinstance(row_ids, list):
        return []
    out: list[int] = []
    for item in row_ids:
        try:
            value = int(item)
        except Exception:
            continue
        if value > 0:
            out.append(value)
    return out[: max(1, max_topics)]


def _pick_top_lines(lines: list[str], keywords: tuple[str, ...], limit: int) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        lowered = _normalize_text(line)
        score = sum(1 for token in keywords if token in lowered)
        if score <= 0:
            continue
        scored.append((score, -idx, line))

    scored.sort(reverse=True)
    picked: list[str] = []
    seen: set[str] = set()
    for _, _, line in scored:
        key = _normalize_text(line)
        if key in seen:
            continue
        seen.add(key)
        picked.append(line)
        if len(picked) >= limit:
            break
    return picked


def _build_heuristic_assessment(
    person_name: str,
    notes: list[NoteDoc],
    status_topics: list[str] | None = None,
    status_sheet_reference: str = "",
) -> str:
    status_topics = status_topics or []
    positive_tokens = (
        "delivered",
        "dowioz",
        "ownership",
        "initiative",
        "improved",
        "impact",
        "result",
        "quality",
        "collaboration",
        "stakeholder",
        "samodziel",
        "proactive",
        "lead",
        "mentoring",
    )
    risk_tokens = (
        "blocker",
        "risk",
        "delay",
        "opozn",
        "problem",
        "dependency",
        "stuck",
        "missed",
        "escalat",
    )
    promotion_tokens = (
        "lead",
        "mentoring",
        "cross-team",
        "stakeholder",
        "ownership",
        "strategy",
        "strateg",
        "decision",
        "scope",
        "autonomy",
    )
    growth_gap_tokens = (
        "needs support",
        "do poprawy",
        "missing",
        "unclear",
        "follow-up",
        "consistency",
        "quality gap",
        "communication",
    )

    all_lines: list[str] = []
    for doc in notes:
        all_lines.extend(_split_candidate_lines(doc.text))
    all_lines.extend(status_topics)

    strengths = _pick_top_lines(all_lines, positive_tokens, limit=6)
    concerns = _pick_top_lines(all_lines, risk_tokens, limit=5)
    promotion_evidence = _pick_top_lines(all_lines, promotion_tokens, limit=5)
    promotion_gaps = _pick_top_lines(all_lines, growth_gap_tokens + risk_tokens, limit=4)

    one_on_one_topics: list[str] = []
    review_topics: list[str] = []

    for line in strengths[:4]:
        one_on_one_topics.append(f"Podkresl i utrwal: {line}")
        review_topics.append(f"Mocna strona z okresu: {line}")

    for line in concerns[:4]:
        one_on_one_topics.append(f"Odblokowanie / wsparcie: {line}")
        review_topics.append(f"Ryzyko do omowienia: {line}")
    for line in status_topics[:4]:
        one_on_one_topics.append(f"Status-log: {line}")
        review_topics.append(f"Status-log (evidence): {line}")

    one_on_one_topics = one_on_one_topics[:8]
    review_topics = review_topics[:8]

    promotion_score = len(promotion_evidence) * 1.4 + len(strengths) * 0.7 - len(concerns) * 1.1
    if promotion_score >= 6:
        readiness = "Wysoka"
    elif promotion_score >= 3:
        readiness = "Srednia"
    else:
        readiness = "Niska"

    confidence_points = len(notes) + len(strengths) + len(concerns)
    if confidence_points >= 14:
        confidence = "Wysoka"
    elif confidence_points >= 8:
        confidence = "Srednia"
    else:
        confidence = "Niska"

    source_list = []
    for doc in notes:
        modified = _parse_iso_datetime(doc.modified_time)
        date_label = modified.date().isoformat() if modified else "unknown"
        source_list.append(f"- {date_label} | {doc.name} | {doc.web_view_link}")

    if not strengths:
        strengths = ["Brak jednoznacznych sygnalow pozytywnego impactu w notatkach."]
    if not concerns:
        concerns = ["Brak wyraznych blockerow w notatkach (warto potwierdzic na 1:1)."]
    if not promotion_evidence:
        promotion_evidence = ["Brak silnych sygnalow poziomu wyzej niz obecna rola."]
    if not promotion_gaps:
        promotion_gaps = ["Doprecyzowac oczekiwania roli docelowej i plan rozwojowy."]

    lines: list[str] = [
        f"# Manager Support Report: {person_name}",
        "",
        "## Performance Snapshot",
        f"- Readiness sygnal (promocja): **{readiness}**",
        f"- Pewnosc oceny (na bazie notatek): **{confidence}**",
        f"- Liczba przeanalizowanych notatek: **{len(notes)}**",
        f"- Tematy z pliku statusowego: **{len(status_topics)}**",
        "",
        "## Tematy na 1:1",
    ]
    lines.extend(f"- {item}" for item in one_on_one_topics or ["Brak wystarczajacych danych."])

    lines.append("")
    lines.append("## Tematy do Performance Review")
    lines.extend(f"- {item}" for item in review_topics or ["Brak wystarczajacych danych."])

    lines.append("")
    lines.append("## Tematy ze statusu")
    lines.extend(f"- {item}" for item in status_topics or ["Brak wpisow statusowych dla tej osoby."])

    lines.append("")
    lines.append("## Gotowosc do awansu")
    lines.append("- Sygnaly za:")
    lines.extend(f"- {item}" for item in promotion_evidence)
    lines.append("- Luki do domkniecia:")
    lines.extend(f"- {item}" for item in promotion_gaps)

    lines.append("")
    lines.append("## Mocne strony (evidence)")
    lines.extend(f"- {item}" for item in strengths)

    lines.append("")
    lines.append("## Ryzyka / blokery (evidence)")
    lines.extend(f"- {item}" for item in concerns)

    lines.append("")
    lines.append("## Zrodla")
    lines.extend(source_list)
    if status_sheet_reference.strip():
        lines.append(f"- status sheet: {status_sheet_reference.strip()}")

    return "\n".join(lines).strip() + "\n"


def _llm_assessment(
    person_name: str,
    notes: list[NoteDoc],
    baseline_report: str,
    status_topics: list[str] | None = None,
) -> str:
    config = AgentConfig.from_env()
    llm = build_gaia_llm(config)
    status_topics = status_topics or []

    notes_blob_parts: list[str] = []
    for idx, doc in enumerate(notes, start=1):
        excerpt = re.sub(r"\s+", " ", doc.text).strip()[:3500]
        notes_blob_parts.append(
            f"[{idx}] {doc.name} | {doc.modified_time} | {doc.web_view_link}\n{excerpt}"
        )

    notes_blob = "\n\n".join(notes_blob_parts)
    status_blob = "\n".join(f"- {row}" for row in status_topics) if status_topics else "No status topics."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Jestes manager-coachem. Analizujesz notatki o pracowniku i tworzysz material dla managera. "
                    "Nie wymyslaj faktow, uzywaj tylko danych ze zrodel. "
                    "Pisz po polsku. "
                    "Uzyj markdown. W kazdym punkcie odwolyj sie do evidence z notatek. "
                    "Sekcje wymagane: Performance Snapshot, Tematy na 1:1, Tematy do Performance Review, "
                    "Gotowosc do awansu, Mocne strony (evidence), Ryzyka/blokery (evidence), Rekomendowane cele rozwojowe na 30-60 dni."
                ),
            ),
            (
                "user",
                (
                    f"Osoba: {person_name}\n\n"
                    "Baseline heuristic report:\n"
                    f"{baseline_report}\n\n"
                    "Status topics from spreadsheet:\n"
                    f"{status_blob}\n\n"
                    "Source notes:\n"
                    f"{notes_blob}"
                ),
            ),
        ]
    )
    raw = (prompt | llm | StrOutputParser()).invoke({})
    return raw.strip() + "\n"


def _slugify(value: str) -> str:
    normalized = _normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "person"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manager support agent from Google Docs notes (1:1 + review + promotion readiness)."
    )
    parser.add_argument(
        "--person",
        required=True,
        help="Person first name used for file-name matching.",
    )
    parser.add_argument(
        "--notes-folder-reference",
        default=os.environ.get("PEOPLE_NOTES_FOLDER_REFERENCE", ""),
        help="Google Drive folder URL/ID containing notes.",
    )
    parser.add_argument(
        "--doc-reference",
        action="append",
        default=[],
        help="Direct Google Doc URL/ID (can be provided multiple times).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_MAX_DOCS", "12") or "12"),
        help="Maximum number of docs to analyze.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output markdown path (default: outputs/people_manager/<date>_<person>_manager_report.md).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM refinement and use heuristic report only.",
    )
    parser.add_argument(
        "--status-sheet-reference",
        default=(
            os.environ.get("PEOPLE_MANAGER_STATUS_SHEET_REFERENCE", "").strip()
            or os.environ.get("STATUS_FILE_REFERENCE", "").strip()
        ),
        help="Google Sheets URL/ID with status entries to merge into report.",
    )
    parser.add_argument(
        "--status-max-topics",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_STATUS_MAX_TOPICS", "10") or "10"),
        help="Maximum number of status topics to merge.",
    )
    parser.add_argument(
        "--status-max-sheets",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_STATUS_MAX_SHEETS", "8") or "8"),
        help="Maximum number of tabs read from status sheet.",
    )
    parser.add_argument(
        "--status-max-rows-per-sheet",
        type=int,
        default=int(os.environ.get("PEOPLE_MANAGER_STATUS_MAX_ROWS_PER_SHEET", "400") or "400"),
        help="Maximum number of rows read per status tab.",
    )
    parser.add_argument(
        "--status-llm-name-mapping",
        action="store_true",
        default=os.environ.get("PEOPLE_MANAGER_STATUS_LLM_NAME_MAPPING", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Use LLM to map first names to full names in status rows when direct matching is weak.",
    )
    parser.add_argument(
        "--no-status-llm-name-mapping",
        action="store_true",
        help="Disable LLM-based person mapping for status rows.",
    )
    return parser.parse_args()


def _resolve_credentials_path() -> str:
    candidates = (
        os.environ.get("PEOPLE_MANAGER_GOOGLE_CREDENTIALS_PATH", "").strip(),
        os.environ.get("GOOGLE_DRIVE_CLIENT_SECRET_PATH", "").strip(),
        os.environ.get("GSC_OAUTH_CLIENT_SECRET_PATH", "").strip(),
    )
    for path in candidates:
        if path:
            return path
    return ""


def main() -> None:
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    args = _parse_args()
    credentials_path = _resolve_credentials_path()
    token_path = os.environ.get("GOOGLE_DRIVE_TOKEN_PATH", ".google_drive_token.json").strip()

    if not credentials_path:
        raise SystemExit(
            "Google credentials missing. Set PEOPLE_MANAGER_GOOGLE_CREDENTIALS_PATH "
            "or GOOGLE_DRIVE_CLIENT_SECRET_PATH."
        )

    status_ref = args.status_sheet_reference.strip()
    if not args.doc_reference and not args.notes_folder_reference and not status_ref:
        raise SystemExit(
            "Provide --notes-folder-reference, --doc-reference, or --status-sheet-reference."
        )

    client = PeopleNotesClient(credentials_path=credentials_path, token_path=token_path)

    source_docs: list[dict[str, str]] = []
    if args.doc_reference:
        for ref in args.doc_reference:
            source_docs.append(client.get_doc_meta(ref))

    if args.notes_folder_reference:
        source_docs.extend(
            client.list_docs_in_folder(
                folder_reference=args.notes_folder_reference,
                max_docs=max(1, args.max_docs),
            )
        )

    dedup_by_id: dict[str, dict[str, str]] = {}
    for row in source_docs:
        doc_id = row.get("id", "")
        if doc_id:
            dedup_by_id[doc_id] = row

    docs = list(dedup_by_id.values())
    docs.sort(
        key=lambda row: (
            _person_match_score(args.person, row.get("name", "")),
            _parse_iso_datetime(row.get("modified_time", "")) or datetime.min,
        ),
        reverse=True,
    )

    person_filtered = [
        row for row in docs if _person_match_score(args.person, row.get("name", "")) > 0
    ]
    selected = person_filtered if person_filtered else docs
    selected = selected[: max(1, args.max_docs)]

    note_docs: list[NoteDoc] = []
    for row in selected:
        doc_id = row.get("id", "")
        if not doc_id:
            continue
        text = client.extract_doc_text(doc_id, max_chars=12000)
        if not text:
            continue
        note_docs.append(
            NoteDoc(
                id=doc_id,
                name=row.get("name", ""),
                modified_time=row.get("modified_time", ""),
                web_view_link=row.get("web_view_link", ""),
                text=text,
            )
        )

    status_topics: list[str] = []
    if status_ref:
        try:
            use_llm_name_mapping = bool(args.status_llm_name_mapping) and not bool(
                args.no_status_llm_name_mapping
            )
            status_topics = client.collect_status_topics(
                status_sheet_reference=status_ref,
                person_name=args.person,
                max_sheets=max(1, args.status_max_sheets),
                max_rows_per_sheet=max(1, args.status_max_rows_per_sheet),
                max_topics=max(1, args.status_max_topics),
                use_llm_name_mapping=use_llm_name_mapping,
            )
        except Exception:
            status_topics = []

    if not note_docs and not status_topics:
        raise SystemExit("No note content or status topics extracted for selected person/docs.")

    heuristic_report = _build_heuristic_assessment(
        args.person,
        note_docs,
        status_topics=status_topics,
        status_sheet_reference=status_ref,
    )
    final_report = heuristic_report

    use_llm = not args.no_llm
    if use_llm:
        try:
            final_report = _llm_assessment(
                args.person,
                note_docs,
                heuristic_report,
                status_topics=status_topics,
            )
        except Exception:
            final_report = heuristic_report

    output = args.output.strip()
    if not output:
        today_label = date.today().isoformat().replace("-", "_")
        output = (
            f"outputs/people_manager/{today_label}_{_slugify(args.person)}"
            "_manager_report.md"
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_report, encoding="utf-8")

    print(f"Manager report generated: {output_path}")
    print(f"Person: {args.person}")
    print(f"Notes analyzed: {len(note_docs)}")
    print(f"Status topics merged: {len(status_topics)}")


if __name__ == "__main__":
    main()
