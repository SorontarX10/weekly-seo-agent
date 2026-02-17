from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class SEOPresentationsClient:
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    FOLDER_MIME = "application/vnd.google-apps.folder"
    SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
    GOOGLE_SLIDES_MIME = "application/vnd.google-apps.presentation"
    PRESENTATION_MIMES = {
        GOOGLE_SLIDES_MIME,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    }
    GENERIC_NOTE_PATTERNS = (
        r"\bseo team demo\b",
        r"\bdemo seo monthly\b",
        r"\bcopy of\b",
        r"\bagenda\b",
        r"\bpodsumowanie\b",
        r"\bsummary\b",
        r"\bin google search\b",
        r"\bmonthly\b",
        r"\btable of contents\b",
        r"\bspis tresci\b",
        r"\bq[1-4]\b",
        r"\broadmap 20\d{2}\b",
        r"^\d{4}[._-]\d{2}[._-]\d{2}$",
        r"^\d{4}[._-]\d{2}[._-]\d{2}\s+[a-z0-9\s._-]+$",
    )
    METRIC_HINT_RE = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|pp|k|m|mln|tys|ms|s|x)\b",
        flags=re.IGNORECASE,
    )
    ACTION_HINT_TOKENS = (
        "test",
        "a/b",
        "experiment",
        "wynik",
        "result",
        "rollout",
        "fix",
        "issue",
        "problem",
        "plan",
        "roadmap",
        "monitoring",
        "alert",
        "backlog",
        "sprint",
    )
    SEO_BUSINESS_TOKENS = (
        "traffic",
        "gmv",
        "revenue",
        "ctr",
        "cvr",
        "click",
        "impression",
        "visibility",
        "ranking",
        "position",
        "query",
        "page",
        "template",
        "page name",
        "brand",
        "non-brand",
        "gsc",
        "senuto",
        "cwv",
        "lcp",
        "inp",
        "cls",
        "index",
        "crawl",
        "google update",
        "discover",
        "wosp",
        "walent",
        "wielkan",
        "ferie",
        "season",
    )

    def __init__(
        self,
        client_secret_path: str,
        token_path: str,
        folder_reference: str,
        max_files_per_year: int = 20,
        max_text_files_per_year: int = 8,
    ) -> None:
        self.client_secret_path = client_secret_path.strip()
        self.token_path = token_path.strip() or ".google_drive_token.json"
        self.folder_reference = folder_reference.strip()
        self.max_files_per_year = max_files_per_year
        self.max_text_files_per_year = max_text_files_per_year
        self._drive = None
        self._slides = None
        self._slides_api_disabled_reported = False

    @staticmethod
    def _extract_folder_id(folder_reference: str) -> str:
        raw = folder_reference.strip()
        if not raw:
            return ""
        if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", raw):
            return raw

        parsed = urlparse(raw)
        if not parsed.scheme and not parsed.netloc:
            return ""

        path_match = re.search(r"/folders/([a-zA-Z0-9_-]{20,})", parsed.path)
        if path_match:
            return path_match.group(1)

        query = parse_qs(parsed.query)
        if "id" in query and query["id"]:
            candidate = query["id"][0].strip()
            if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", candidate):
                return candidate
        return ""

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _parse_filename_date(name: str) -> date | None:
        raw = name.strip()
        patterns = (
            r"(?P<y>20\d{2})[-_.](?P<m>\d{1,2})[-_.](?P<d>\d{1,2})",
            r"(?P<d>\d{1,2})[-_.](?P<m>\d{1,2})[-_.](?P<y>20\d{2})",
        )
        for pattern in patterns:
            match = re.search(pattern, raw)
            if not match:
                continue
            try:
                year = int(match.group("y"))
                month = int(match.group("m"))
                day = int(match.group("d"))
                return date(year, month, day)
            except ValueError:
                continue
        return None

    @staticmethod
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

    def _load_credentials(self) -> Credentials:
        secret_path = Path(self.client_secret_path)
        if not secret_path.exists():
            raise RuntimeError(
                "Google credentials file not found for SEO presentations: "
                f"{self.client_secret_path}"
            )

        secret_payload = json.loads(secret_path.read_text(encoding="utf-8"))
        if secret_payload.get("type") == "service_account":
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

    def _slides_service(self):
        if self._slides is None:
            self._slides = build(
                "slides",
                "v1",
                credentials=self._load_credentials(),
                cache_discovery=False,
            )
        return self._slides

    def _find_year_folder(self, root_folder_id: str, year: int) -> dict[str, Any] | None:
        drive = self._drive_service()
        year_label = str(year)
        exact_query = (
            f"mimeType='{self.FOLDER_MIME}' and trashed=false "
            f"and '{root_folder_id}' in parents and name='{self._escape_query_value(year_label)}'"
        )
        response = drive.files().list(
            q=exact_query,
            spaces="drive",
            fields="files(id,name,mimeType,modifiedTime,createdTime)",
            pageSize=3,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files = response.get("files", [])
        if files:
            return files[0]

        fuzzy_query = (
            f"mimeType='{self.FOLDER_MIME}' and trashed=false "
            f"and '{root_folder_id}' in parents"
        )
        response = drive.files().list(
            q=fuzzy_query,
            spaces="drive",
            fields="files(id,name,mimeType,modifiedTime,createdTime)",
            pageSize=200,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        for row in response.get("files", []):
            if year_label in str(row.get("name", "")):
                return row

        # Fallback: search recursively up to 3 levels below the configured root.
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(root_folder_id, 0)]
        max_depth = 3
        max_checked = 300
        checked = 0
        while queue and checked < max_checked:
            parent_id, depth = queue.pop(0)
            if parent_id in visited or depth > max_depth:
                continue
            visited.add(parent_id)
            checked += 1
            children_query = (
                f"mimeType='{self.FOLDER_MIME}' and trashed=false "
                f"and '{parent_id}' in parents"
            )
            child_response = drive.files().list(
                q=children_query,
                spaces="drive",
                fields="files(id,name,mimeType,modifiedTime,createdTime)",
                pageSize=200,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute()
            for child in child_response.get("files", []):
                child_id = str(child.get("id", "")).strip()
                child_name = str(child.get("name", "")).strip()
                if year_label in child_name:
                    return child
                if child_id and depth < max_depth:
                    queue.append((child_id, depth + 1))
        return None

    def _list_presentations_in_folder(self, folder_id: str) -> list[dict[str, Any]]:
        drive = self._drive_service()
        query = f"'{folder_id}' in parents and trashed=false"
        response = drive.files().list(
            q=query,
            spaces="drive",
            fields=(
                "files(id,name,mimeType,modifiedTime,createdTime,webViewLink,"
                "shortcutDetails(targetId,targetMimeType))"
            ),
            pageSize=500,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        rows: list[dict[str, Any]] = []
        for row in response.get("files", []):
            if not isinstance(row, dict):
                continue
            row = self._resolve_shortcut_row(row)
            parsed = self._build_presentation_file_row(row)
            if parsed:
                rows.append(parsed)
        rows = self._dedupe_presentation_rows(rows)
        return rows[: self.max_files_per_year]

    def _list_presentations_recursive(
        self, root_folder_id: str, max_depth: int = 4
    ) -> list[dict[str, Any]]:
        drive = self._drive_service()
        queue: list[tuple[str, int]] = [(root_folder_id, 0)]
        visited: set[str] = set()
        rows: list[dict[str, Any]] = []

        while queue:
            parent_id, depth = queue.pop(0)
            if parent_id in visited or depth > max_depth:
                continue
            visited.add(parent_id)

            query = f"'{parent_id}' in parents and trashed=false"
            response = drive.files().list(
                q=query,
                spaces="drive",
                fields=(
                    "files(id,name,mimeType,modifiedTime,createdTime,webViewLink,"
                    "shortcutDetails(targetId,targetMimeType))"
                ),
                pageSize=500,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute()
            for row in response.get("files", []):
                if not isinstance(row, dict):
                    continue
                row = self._resolve_shortcut_row(row)
                mime = str(row.get("mimeType", "")).strip()
                row_id = str(row.get("id", "")).strip()
                if not row_id:
                    continue
                if mime == self.FOLDER_MIME and depth < max_depth:
                    queue.append((row_id, depth + 1))
                    continue
                parsed = self._build_presentation_file_row(row)
                if parsed:
                    rows.append(parsed)

        return self._dedupe_presentation_rows(rows)

    def _build_presentation_file_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        mime = str(row.get("mimeType", "")).strip()
        if mime not in self.PRESENTATION_MIMES:
            return None
        name = str(row.get("name", "")).strip()
        parsed = self._parse_filename_date(name)
        modified_raw = str(row.get("modifiedTime", "")).strip()
        modified_dt = self._parse_iso_datetime(modified_raw)
        sort_day = parsed or (modified_dt.date() if modified_dt else date.min)
        return {
            "id": str(row.get("id", "")).strip(),
            "name": name,
            "mimeType": mime,
            "modifiedTime": modified_raw,
            "webViewLink": str(row.get("webViewLink", "")).strip(),
            "file_date": parsed.isoformat() if parsed else "",
            "sort_day": sort_day,
        }

    def _resolve_shortcut_row(self, row: dict[str, Any]) -> dict[str, Any]:
        mime = str(row.get("mimeType", "")).strip()
        if mime != self.SHORTCUT_MIME:
            return row

        details = row.get("shortcutDetails")
        if not isinstance(details, dict):
            return row
        target_id = str(details.get("targetId", "")).strip()
        target_mime = str(details.get("targetMimeType", "")).strip()
        if not target_id:
            return row

        target_meta = self._get_file_metadata(target_id)
        if target_meta:
            return target_meta

        if target_mime:
            row["id"] = target_id
            row["mimeType"] = target_mime
        return row

    def _get_file_metadata(self, file_id: str) -> dict[str, Any] | None:
        drive = self._drive_service()
        try:
            payload = drive.files().get(
                fileId=file_id,
                fields="id,name,mimeType,modifiedTime,createdTime,webViewLink",
                supportsAllDrives=True,
            ).execute()
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        ascii_text = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )
        return re.sub(r"\s+", " ", ascii_text).strip()

    @staticmethod
    def _looks_like_url_or_slug(text: str) -> bool:
        raw = text.strip()
        lowered = raw.lower()
        if raw.startswith("…") or raw.startswith("..."):
            return True
        if "http://" in lowered or "https://" in lowered or "www." in lowered:
            return True
        if lowered.count("%") >= 3:
            return True
        if raw.count("-") >= 5 and re.search(r"\d", raw):
            return True
        if len(raw) >= 90 and raw.count("/") >= 2:
            return True
        tokens = re.split(r"\s+", lowered)
        long_tokens = [token for token in tokens if len(token) >= 28 and re.search(r"[a-z0-9]", token)]
        if len(long_tokens) >= 2:
            return True
        if any(len(token) >= 40 and "-" in token and re.search(r"\d", token) for token in tokens):
            return True
        return False

    @classmethod
    def _canonical_file_name(cls, name: str) -> str:
        normalized = cls._normalize_for_match(name)
        normalized = re.sub(r"^copy of\s+", "", normalized)
        normalized = re.sub(r"^kopia\s+", "", normalized)
        normalized = re.sub(r"\.(pptx|ppt|gslides)$", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def _is_probably_copy_file(cls, name: str) -> bool:
        normalized = cls._normalize_for_match(name)
        return normalized.startswith("copy of ") or normalized.startswith("kopia ")

    @classmethod
    def _is_generic_highlight(cls, text: str, file_name: str = "") -> bool:
        normalized = cls._normalize_for_match(text)
        if not normalized:
            return True
        words = normalized.split()
        if normalized.startswith("from:"):
            return True
        if cls.METRIC_HINT_RE.search(normalized):
            return False
        action_hits = sum(1 for token in cls.ACTION_HINT_TOKENS if token in normalized)
        business_hits = sum(1 for token in cls.SEO_BUSINESS_TOKENS if token in normalized)
        if len(words) <= 3 and action_hits == 0:
            return True
        if len(words) <= 5 and business_hits <= 1 and action_hits == 0:
            return True
        for pattern in cls.GENERIC_NOTE_PATTERNS:
            if re.search(pattern, normalized):
                return True
        if file_name:
            file_base = cls._canonical_file_name(file_name)
            if file_base and normalized in file_base:
                return True
            if file_base and file_base in normalized and len(normalized) <= len(file_base) + 12:
                return True
        return False

    @classmethod
    def _score_highlight_line(cls, text: str, slide_index: int) -> int:
        normalized = cls._normalize_for_match(text)
        score = 0

        metric_match = cls.METRIC_HINT_RE.search(normalized)
        if metric_match:
            score += 6

        comparison_tokens = ("yoy", "wow", "mom", "vs ", "delta", "change", "spad", "wzrost")
        if any(token in normalized for token in comparison_tokens):
            score += 2

        business_hits = sum(1 for token in cls.SEO_BUSINESS_TOKENS if token in normalized)
        score += min(5, business_hits)

        action_hits = sum(1 for token in cls.ACTION_HINT_TOKENS if token in normalized)
        score += min(3, action_hits)

        if slide_index == 0:
            score -= 4
        elif slide_index == 1:
            score -= 2

        if len(normalized) > 200:
            score -= 1

        return score

    def _dedupe_presentation_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        best_score: dict[tuple[str, str], int] = {}

        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            canonical_name = self._canonical_file_name(name)
            if not canonical_name:
                continue
            file_date = str(row.get("file_date", "")).strip()
            sort_day = row.get("sort_day")
            if not file_date and isinstance(sort_day, date):
                file_date = sort_day.isoformat()
            key = (canonical_name, file_date)

            score = 0
            if row.get("mimeType") == self.GOOGLE_SLIDES_MIME:
                score += 3
            if not self._is_probably_copy_file(name):
                score += 2
            modified_raw = str(row.get("modifiedTime", "")).strip()
            if modified_raw:
                score += 1

            existing = best_score.get(key)
            if existing is None or score > existing:
                best_score[key] = score
                best_by_key[key] = row

        out = list(best_by_key.values())
        out.sort(key=lambda item: item.get("sort_day", date.min), reverse=True)
        return out

    def _extract_slide_highlights(
        self,
        presentation_id: str,
        file_name: str = "",
        max_items: int = 8,
    ) -> list[str]:
        slides = self._slides_service()
        payload = slides.presentations().get(presentationId=presentation_id).execute()
        candidates: list[tuple[int, int, str]] = []
        seen: set[str] = set()

        for slide_index, slide in enumerate(payload.get("slides", [])):
            if not isinstance(slide, dict):
                continue
            for element in slide.get("pageElements", []):
                if not isinstance(element, dict):
                    continue
                shape = element.get("shape")
                if not isinstance(shape, dict):
                    continue
                text_body = shape.get("text")
                if not isinstance(text_body, dict):
                    continue
                fragments: list[str] = []
                for text_element in text_body.get("textElements", []):
                    if not isinstance(text_element, dict):
                        continue
                    run = text_element.get("textRun")
                    if not isinstance(run, dict):
                        continue
                    raw = str(run.get("content", ""))
                    if raw:
                        fragments.append(raw)
                if not fragments:
                    continue

                combined = "".join(fragments)
                for raw_line in combined.splitlines():
                    text = self._normalize_text(raw_line).strip(" -*\t")
                    if self._looks_like_url_or_slug(text):
                        continue
                    if len(text) < 18:
                        continue
                    if len(text) > 260:
                        text = text[:260].rstrip() + "..."
                    norm = self._normalize_for_match(text)
                    if not norm or norm in seen:
                        continue
                    if self._is_generic_highlight(text, file_name=file_name):
                        continue
                    score = self._score_highlight_line(text, slide_index=slide_index)
                    if score < 4:
                        continue
                    seen.add(norm)
                    candidates.append((score, slide_index, text))

        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
        highlights: list[str] = []
        for _, _, text in candidates:
            highlights.append(text)
            if len(highlights) >= max_items:
                break
        return highlights

    @staticmethod
    def _is_slides_api_disabled_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "slides.googleapis.com" in message
            and ("service_disabled" in message or "not been used in project" in message)
        )

    @staticmethod
    def _resolve_file_year(file_row: dict[str, Any]) -> int | None:
        file_date = str(file_row.get("file_date", "")).strip()
        if file_date:
            try:
                return date.fromisoformat(file_date[:10]).year
            except ValueError:
                pass
        modified_raw = str(file_row.get("modifiedTime", "")).strip()
        modified_dt = SEOPresentationsClient._parse_iso_datetime(modified_raw)
        if modified_dt:
            return modified_dt.year
        return None

    def _attach_highlights(
        self,
        year: int,
        files: list[dict[str, Any]],
        errors: list[str],
        highlights: list[dict[str, str]],
    ) -> None:
        seen_note_keys: set[str] = set()
        for row in highlights:
            if not isinstance(row, dict):
                continue
            if str(row.get("year", "")).strip() != str(year):
                continue
            note = self._normalize_for_match(str(row.get("note", "")).strip())
            if note:
                seen_note_keys.add(note)

        for index, file_row in enumerate(files):
            file_row["highlights"] = []
            if (
                file_row.get("mimeType") == self.GOOGLE_SLIDES_MIME
                and index < self.max_text_files_per_year
                and not self._slides_api_disabled_reported
            ):
                try:
                    file_row["highlights"] = self._extract_slide_highlights(
                        presentation_id=str(file_row.get("id", "")).strip(),
                        file_name=str(file_row.get("name", "")).strip(),
                        max_items=6,
                    )
                except Exception as exc:
                    if self._is_slides_api_disabled_error(exc):
                        if not self._slides_api_disabled_reported:
                            errors.append(
                                "Google Slides API is disabled for current credentials. "
                                "Enable `slides.googleapis.com` to extract slide-level highlights."
                            )
                        self._slides_api_disabled_reported = True
                    else:
                        errors.append(
                            f"Slides parsing failed for '{file_row.get('name', '')}': {exc}"
                        )
            for line in file_row.get("highlights", [])[:2]:
                note_key = self._normalize_for_match(str(line))
                if not note_key or note_key in seen_note_keys:
                    continue
                seen_note_keys.add(note_key)
                highlights.append(
                    {
                        "year": str(year),
                        "file": str(file_row.get("name", "")),
                        "date": str(file_row.get("file_date") or file_row.get("modifiedTime", "")[:10]),
                        "note": line,
                        "url": str(file_row.get("webViewLink", "")).strip(),
                    }
                )

    def collect_context(
        self,
        run_date: date,
    ) -> dict[str, Any]:
        folder_id = self._extract_folder_id(self.folder_reference)
        if not folder_id:
            return {
                "enabled": False,
                "errors": ["SEO presentations folder reference is missing or invalid."],
                "years": [],
                "highlights": [],
            }

        years = [run_date.year, run_date.year - 1]
        errors: list[str] = []
        year_rows_by_year: dict[int, dict[str, Any]] = {}
        highlights: list[dict[str, str]] = []
        backfill_years: list[int] = []

        for year in years:
            year_folder = self._find_year_folder(folder_id, year)
            if not year_folder:
                backfill_years.append(year)
                continue

            files = self._list_presentations_in_folder(str(year_folder.get("id", "")).strip())
            if not files:
                files = self._list_presentations_recursive(
                    str(year_folder.get("id", "")).strip(),
                    max_depth=3,
                )[: self.max_files_per_year]
            year_rows_by_year[year] = {
                "year": str(year),
                "folder_id": str(year_folder.get("id", "")).strip(),
                "folder_name": str(year_folder.get("name", "")).strip(),
                "file_count": len(files),
                "files": files,
            }
            if files:
                self._attach_highlights(year=year, files=files, errors=errors, highlights=highlights)
            else:
                backfill_years.append(year)

        if backfill_years:
            fallback_files = self._list_presentations_recursive(folder_id)
            files_by_year: dict[int, list[dict[str, Any]]] = {
                year: [] for year in backfill_years
            }
            for file_row in fallback_files:
                file_year = self._resolve_file_year(file_row)
                if file_year in files_by_year:
                    files_by_year[file_year].append(file_row)

            for year in backfill_years:
                files = files_by_year.get(year, [])
                files.sort(key=lambda item: item.get("sort_day", date.min), reverse=True)
                files = files[: self.max_files_per_year]
                if not files:
                    continue
                self._attach_highlights(year=year, files=files, errors=errors, highlights=highlights)
                year_rows_by_year[year] = {
                    "year": str(year),
                    "folder_id": folder_id,
                    "folder_name": "recursive scan fallback",
                    "file_count": len(files),
                    "files": files,
                }

        year_rows: list[dict[str, Any]] = []
        for year in years:
            row = year_rows_by_year.get(year)
            if row:
                year_rows.append(row)
            else:
                errors.append(f"Year folder not found for {year}.")

        return {
            "enabled": True,
            "source": "Google Drive SEO Team Presentations",
            "root_folder_id": folder_id,
            "years": year_rows,
            "highlights": highlights[:30],
            "errors": errors,
        }
