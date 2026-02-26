from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class ContinuityClient:
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    FOLDER_MIME = "application/vnd.google-apps.folder"
    DOC_MIME = "application/vnd.google-apps.document"
    SHEET_MIME = "application/vnd.google-apps.spreadsheet"
    BRAND_TOKENS = ("allegro", "allegro.pl", "allegro pl")

    def __init__(
        self,
        client_secret_path: str,
        token_path: str,
        reports_folder_name: str,
        reports_folder_id: str = "",
        status_file_reference: str = "",
        status_search_folder_reference: str = "",
        max_recent_reports: int = 3,
        yoy_tolerance_days: int = 28,
        max_status_rows: int = 12,
    ) -> None:
        self.client_secret_path = client_secret_path.strip()
        self.token_path = token_path.strip() or ".google_drive_token.json"
        self.reports_folder_name = reports_folder_name.strip() or "SEO Weekly Reports"
        self.reports_folder_id = reports_folder_id.strip()
        self.status_file_reference = status_file_reference.strip()
        self.status_search_folder_reference = status_search_folder_reference.strip()
        self.max_recent_reports = max(1, max_recent_reports)
        self.yoy_tolerance_days = max(7, yoy_tolerance_days)
        self.max_status_rows = max(1, max_status_rows)
        self._drive = None
        self._docs = None
        self._sheets = None

    @staticmethod
    def _normalize_text(value: str) -> str:
        ascii_text = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )
        return re.sub(r"\s+", " ", ascii_text).strip()

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

    @staticmethod
    def _parse_report_date(name: str) -> date | None:
        raw = name.strip()
        patterns = (
            r"(?P<y>20\d{2})[_\-.](?P<m>\d{1,2})[_\-.](?P<d>\d{1,2})",
            r"(?P<d>\d{1,2})[_\-.](?P<m>\d{1,2})[_\-.](?P<y>20\d{2})",
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

    @classmethod
    def _parse_flexible_date(cls, raw: object) -> date | None:
        if raw is None:
            return None

        if isinstance(raw, date):
            return raw

        if isinstance(raw, (int, float)):
            if raw > 20000:
                serial = int(float(raw))
                base = date(1899, 12, 30)
                return base + timedelta(days=serial)
            return None

        text = str(raw).strip()
        if not text:
            return None

        if text.isdigit() and len(text) >= 5:
            serial = int(text)
            if serial > 20000:
                base = date(1899, 12, 30)
                return base + timedelta(days=serial)

        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d.%m.%Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%y",
            "%d/%m/%y",
            "%m/%d/%Y",
            "%m-%d-%Y",
        ):
            try:
                return datetime.strptime(text[:10], fmt).date()
            except ValueError:
                continue

        try:
            if "T" in text and text.endswith("Z"):
                return datetime.fromisoformat(text[:-1] + "+00:00").date()
            if "T" in text:
                return datetime.fromisoformat(text).date()
        except ValueError:
            pass

        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _score_line(line: str) -> int:
        lowered = line.lower()
        keywords = (
            "traffic",
            "gmv",
            "click",
            "ctr",
            "position",
            "seo",
            "ranking",
            "core update",
            "wosp",
            "walent",
            "easter",
            "season",
            "holiday",
            "ferie",
        )
        return sum(1 for token in keywords if token in lowered)

    @classmethod
    def _extract_highlights(cls, text: str, max_items: int = 4) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", raw_line).strip(" -*\t")
            if len(cleaned) < 30 or len(cleaned) > 240:
                continue
            lines.append(cleaned)

        if not lines:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                if len(cleaned) < 30 or len(cleaned) > 240:
                    continue
                lines.append(cleaned)

        scored = sorted(
            ((cls._score_line(line), idx, line) for idx, line in enumerate(lines)),
            key=lambda item: (-item[0], item[1]),
        )

        unique: list[str] = []
        seen: set[str] = set()
        for _, _, line in scored:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(line)
            if len(unique) >= max_items:
                break
        return unique

    def _load_credentials(self) -> Credentials:
        secret_path = Path(self.client_secret_path)
        if not secret_path.exists():
            raise RuntimeError(
                "Google credentials file not found for continuity context: "
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

    def _resolve_reports_folder_id(self) -> str:
        if self.reports_folder_id:
            return self.reports_folder_id

        drive = self._drive_service()
        escaped = self._escape_query_value(self.reports_folder_name)
        query = (
            f"mimeType='{self.FOLDER_MIME}' and name='{escaped}' "
            "and trashed=false and 'root' in parents"
        )
        response = (
            drive.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id,name)",
                pageSize=1,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        if not files:
            raise RuntimeError(
                "Reports folder not found in Drive root: "
                f"{self.reports_folder_name}"
            )
        return str(files[0].get("id", "")).strip()

    def _list_report_documents(self, folder_id: str) -> list[dict[str, Any]]:
        drive = self._drive_service()
        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and mimeType='{self.DOC_MIME}'"
        )
        response = drive.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name,mimeType,modifiedTime,createdTime,webViewLink)",
            pageSize=500,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        rows: list[dict[str, Any]] = []
        for row in response.get("files", []):
            name = str(row.get("name", "")).strip()
            report_day = self._parse_report_date(name)
            modified_raw = str(row.get("modifiedTime", "")).strip()
            modified_dt = self._parse_iso_datetime(modified_raw)
            if report_day is None and modified_dt is not None:
                report_day = modified_dt.date()
            rows.append(
                {
                    "id": str(row.get("id", "")).strip(),
                    "name": name,
                    "report_date": report_day,
                    "modifiedTime": modified_raw,
                    "webViewLink": str(row.get("webViewLink", "")).strip(),
                }
            )

        rows = [row for row in rows if row.get("id") and row.get("report_date")]
        rows.sort(
            key=lambda item: (
                item.get("report_date", date.min),
                item.get("modifiedTime", ""),
            ),
            reverse=True,
        )
        return rows

    def _select_recent_and_yoy(
        self,
        rows: list[dict[str, Any]],
        run_date: date,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        eligible = [
            row
            for row in rows
            if isinstance(row.get("report_date"), date)
            and row["report_date"] < run_date
        ]

        recent = eligible[: self.max_recent_reports]

        target = run_date - timedelta(weeks=52)
        yoy_candidate: dict[str, Any] | None = None
        best_distance = 99999
        for row in eligible:
            report_day = row.get("report_date")
            if not isinstance(report_day, date):
                continue
            distance = abs((report_day - target).days)
            if distance > self.yoy_tolerance_days:
                continue
            if distance < best_distance:
                best_distance = distance
                yoy_candidate = row

        if yoy_candidate and any(str(item.get("id")) == str(yoy_candidate.get("id")) for item in recent):
            # Keep YoY context distinct from the short recent list.
            yoy_candidate = None

        return recent, yoy_candidate

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

    def _extract_doc_text(self, document_id: str, max_chars: int = 6000) -> str:
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

    def _build_report_entry(self, row: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        report_date = row.get("report_date")
        report_date_label = report_date.isoformat() if isinstance(report_date, date) else ""
        entry = {
            "id": str(row.get("id", "")).strip(),
            "name": str(row.get("name", "")).strip(),
            "date": report_date_label,
            "url": str(row.get("webViewLink", "")).strip(),
            "highlights": [],
            "excerpt": "",
        }

        document_id = entry["id"]
        if not document_id:
            return entry

        try:
            text = self._extract_doc_text(document_id=document_id, max_chars=5500)
            entry["excerpt"] = text[:1200]
            entry["highlights"] = self._extract_highlights(text, max_items=4)
        except Exception as exc:
            errors.append(f"Report text extraction failed for '{entry['name']}': {exc}")

        return entry

    def collect_historical_reports(self, run_date: date) -> dict[str, Any]:
        if not self.client_secret_path:
            return {
                "enabled": False,
                "errors": ["Google credentials not configured for historical reports."],
                "recent_reports": [],
                "yoy_report": {},
            }

        errors: list[str] = []
        try:
            folder_id = self._resolve_reports_folder_id()
        except Exception as exc:
            return {
                "enabled": False,
                "errors": [str(exc)],
                "recent_reports": [],
                "yoy_report": {},
            }

        try:
            rows = self._list_report_documents(folder_id=folder_id)
        except Exception as exc:
            return {
                "enabled": False,
                "errors": [f"Failed to list historical reports: {exc}"],
                "recent_reports": [],
                "yoy_report": {},
            }

        recent_rows, yoy_row = self._select_recent_and_yoy(rows=rows, run_date=run_date)
        recent_entries = [self._build_report_entry(row, errors=errors) for row in recent_rows]
        yoy_entry = self._build_report_entry(yoy_row, errors=errors) if yoy_row else {}

        return {
            "enabled": True,
            "source": "Google Drive weekly report archive",
            "folder_id": folder_id,
            "available_reports": len(rows),
            "recent_reports": recent_entries,
            "yoy_report": yoy_entry,
            "errors": errors,
        }

    def _get_drive_file(self, file_id: str) -> dict[str, Any] | None:
        drive = self._drive_service()
        try:
            file_meta = drive.files().get(
                fileId=file_id,
                fields="id,name,mimeType,modifiedTime,webViewLink",
                supportsAllDrives=True,
            ).execute()
        except Exception:
            return None
        if not isinstance(file_meta, dict):
            return None
        return file_meta

    @staticmethod
    def _status_name_score(name: str) -> int:
        lowered = name.lower()
        score = 0
        for token in ("status", "statusy", "seo", "demo", "weekly"):
            if token in lowered:
                score += 1
        return score

    @classmethod
    def _is_ignored_sheet_title(cls, title: str) -> bool:
        normalized = cls._normalize_text(title)
        ignored_tokens = (
            "biblioteka",
            "library",
            "slownik",
            "dictionary",
            "instrukcja",
            "manual",
            "template",
            "szablon",
            "readme",
        )
        return any(token in normalized for token in ignored_tokens)

    def _search_status_in_folder(self, folder_id: str, max_depth: int = 4) -> dict[str, Any] | None:
        drive = self._drive_service()
        queue: list[tuple[str, int]] = [(folder_id, 0)]
        visited: set[str] = set()
        candidates: list[dict[str, Any]] = []

        while queue:
            parent_id, depth = queue.pop(0)
            if parent_id in visited or depth > max_depth:
                continue
            visited.add(parent_id)

            query = f"'{parent_id}' in parents and trashed=false"
            response = drive.files().list(
                q=query,
                spaces="drive",
                fields="files(id,name,mimeType,modifiedTime,webViewLink)",
                pageSize=500,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute()

            for row in response.get("files", []):
                if not isinstance(row, dict):
                    continue
                row_id = str(row.get("id", "")).strip()
                mime = str(row.get("mimeType", "")).strip()
                if not row_id:
                    continue
                if mime == self.SHEET_MIME:
                    row_name = str(row.get("name", "")).strip()
                    row["_name_score"] = self._status_name_score(row_name)
                    candidates.append(row)
                elif mime == self.FOLDER_MIME and depth < max_depth:
                    queue.append((row_id, depth + 1))

        if not candidates:
            return None

        candidates.sort(
            key=lambda row: (
                int(row.get("_name_score", 0)),
                self._parse_iso_datetime(str(row.get("modifiedTime", ""))) or datetime.min,
            ),
            reverse=True,
        )
        return candidates[0]

    def _search_status_globally(self) -> dict[str, Any] | None:
        drive = self._drive_service()
        query = (
            f"mimeType='{self.SHEET_MIME}' and trashed=false "
            "and (name contains 'status' or name contains 'Status')"
        )
        response = drive.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name,mimeType,modifiedTime,webViewLink)",
            pageSize=20,
            orderBy="modifiedTime desc",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files = response.get("files", [])
        if not files:
            return None
        files.sort(
            key=lambda row: (
                self._status_name_score(str(row.get("name", ""))),
                self._parse_iso_datetime(str(row.get("modifiedTime", ""))) or datetime.min,
            ),
            reverse=True,
        )
        return files[0]

    def _resolve_status_file(self) -> dict[str, Any] | None:
        explicit_id = self._extract_drive_id(self.status_file_reference)
        if explicit_id:
            meta = self._get_drive_file(explicit_id)
            if meta:
                mime = str(meta.get("mimeType", "")).strip()
                if mime == self.SHEET_MIME:
                    return meta
                if mime == self.FOLDER_MIME:
                    found = self._search_status_in_folder(explicit_id)
                    if found:
                        return found

        folder_id = self._extract_drive_id(self.status_search_folder_reference)
        if folder_id:
            found = self._search_status_in_folder(folder_id)
            if found:
                return found

        return self._search_status_globally()

    @classmethod
    def _pick_date_column(cls, headers: list[str]) -> int | None:
        normalized = [cls._normalize_text(header) for header in headers]
        priority_tokens = (
            ("status date", "data statusu"),
            ("date", "data"),
        )
        for tokens in priority_tokens:
            for idx, name in enumerate(normalized):
                if any(token in name for token in tokens):
                    return idx
        return None

    @staticmethod
    def _trade_plan_yoy_hypothesis(
        *,
        current_spend: float,
        yoy_spend: float,
        current_clicks: float,
        yoy_clicks: float,
        current_impressions: float,
        yoy_impressions: float,
    ) -> dict[str, object]:
        if yoy_spend <= 0.0 and yoy_clicks <= 0.0 and yoy_impressions <= 0.0:
            return {
                "impact": "neutral",
                "confidence": 35,
                "reason": "YoY baseline is missing, so impact cannot be inferred reliably.",
            }

        spend_delta_pct = ((current_spend - yoy_spend) / yoy_spend * 100.0) if yoy_spend else 0.0
        clicks_delta_pct = ((current_clicks - yoy_clicks) / yoy_clicks * 100.0) if yoy_clicks else 0.0
        impressions_delta_pct = (
            ((current_impressions - yoy_impressions) / yoy_impressions * 100.0) if yoy_impressions else 0.0
        )

        if yoy_spend > 0.0 and current_spend > 0.0 and abs(spend_delta_pct) < 8.0:
            return {
                "impact": "neutral",
                "confidence": 62,
                "reason": (
                    "Planned spend is close to last year, which points to a neutral YoY demand-allocation effect "
                    "unless external demand changed materially."
                ),
            }

        if spend_delta_pct >= 12.0:
            if clicks_delta_pct >= -5.0 and impressions_delta_pct >= -5.0:
                return {
                    "impact": "positive",
                    "confidence": 74,
                    "reason": (
                        "Planned support is materially higher YoY and visibility signals are not weaker, "
                        "so this likely increased demand capture potential."
                    ),
                }
            return {
                "impact": "negative",
                "confidence": 68,
                "reason": (
                    "Planned support is higher YoY but clicks/impressions are weaker vs last year, "
                    "which suggests lower expected efficiency or less favorable demand mix."
                ),
            }

        if spend_delta_pct <= -12.0:
            if clicks_delta_pct >= 5.0 or impressions_delta_pct >= 5.0:
                return {
                    "impact": "positive",
                    "confidence": 66,
                    "reason": (
                        "Support is lower YoY while visibility signals remain stronger, "
                        "which implies more efficient demand capture vs last year."
                    ),
                }
            return {
                "impact": "negative",
                "confidence": 72,
                "reason": (
                    "Planned support is materially lower YoY and no stronger visibility offset is visible, "
                    "so the expected effect on demand allocation is negative."
                ),
            }

        return {
            "impact": "neutral",
            "confidence": 58,
            "reason": (
                "YoY differences are mixed across support and visibility indicators, "
                "so the net effect is likely neutral in this window."
            ),
        }

    @classmethod
    def _pick_text_column(cls, headers: list[str], tokens: tuple[str, ...]) -> int | None:
        normalized = [cls._normalize_text(header) for header in headers]
        for idx, name in enumerate(normalized):
            if any(token in name for token in tokens):
                return idx
        return None

    @staticmethod
    def _cell_text(row: list[object], idx: int | None) -> str:
        if idx is None or idx < 0 or idx >= len(row):
            return ""
        return str(row[idx]).strip()

    @staticmethod
    def _set_cell_text(row: list[object], idx: int, value: str) -> None:
        if idx < 0:
            return
        if idx >= len(row):
            row.extend([""] * (idx - len(row) + 1))
        row[idx] = value

    @classmethod
    def _fill_down_sparse_rows(
        cls,
        rows: list[list[object]],
        *,
        filldown_indices: tuple[int | None, ...],
    ) -> list[list[object]]:
        tracked = [idx for idx in filldown_indices if isinstance(idx, int) and idx >= 0]
        if not tracked:
            return list(rows)

        carry: dict[int, str] = {}
        out: list[list[object]] = []
        for raw in rows:
            if not isinstance(raw, list):
                continue
            row = list(raw)
            for idx in tracked:
                value = cls._cell_text(row, idx)
                if value:
                    carry[idx] = value
                    continue
                fallback = carry.get(idx, "")
                if fallback:
                    cls._set_cell_text(row, idx, fallback)
            out.append(row)
        return out

    @classmethod
    def _is_non_brand_phrase(cls, value: str) -> bool:
        normalized = cls._normalize_text(value)
        if not normalized:
            return False
        return not any(token in normalized for token in cls.BRAND_TOKENS)

    @staticmethod
    def _parse_numeric_cell(raw: object) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)

        text = str(raw).strip()
        if not text:
            return None

        cleaned = text.lower().replace("\u00a0", " ").strip()
        for token in ("pln", "zloty", "zł", "zl", "szt", "pcs"):
            cleaned = cleaned.replace(token, "")
        cleaned = cleaned.replace("%", "").strip()

        multiplier = 1.0
        if cleaned.endswith("k"):
            multiplier = 1000.0
            cleaned = cleaned[:-1]
        elif cleaned.endswith("m"):
            multiplier = 1000000.0
            cleaned = cleaned[:-1]

        cleaned = cleaned.replace(" ", "")
        cleaned = re.sub(r"[^0-9,.\-]", "", cleaned)
        if not cleaned or cleaned in {"-", "--"}:
            return None

        if "." in cleaned and "," not in cleaned and re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", cleaned):
            cleaned = cleaned.replace(".", "")

        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None

    @classmethod
    def _numeric_ratio_for_column(
        cls,
        rows: list[list[object]],
        column_idx: int,
        max_probe: int = 120,
    ) -> float:
        if column_idx < 0:
            return 0.0
        probe = rows[:max_probe]
        if not probe:
            return 0.0
        parsed = 0
        total = 0
        for row in probe:
            if column_idx >= len(row):
                continue
            total += 1
            if cls._parse_numeric_cell(row[column_idx]) is not None:
                parsed += 1
        if total == 0:
            return 0.0
        return parsed / total

    @classmethod
    def _pick_keyword_column(cls, headers: list[str]) -> int | None:
        keyword_tokens = (
            "keyword",
            "keywords",
            "query",
            "fraza",
            "phrase",
            "trend",
            "topic",
            "produkt",
            "product",
            "item",
            "nazwa",
        )
        idx = cls._pick_text_column(headers, keyword_tokens)
        if idx is not None:
            return idx
        for i, name in enumerate(headers):
            text = cls._normalize_text(name)
            if text and "date" not in text and "data" not in text:
                return i
        return None

    @classmethod
    def _pick_page_column(cls, headers: list[str]) -> int | None:
        page_tokens = ("page", "url", "link", "strona")
        return cls._pick_text_column(headers, page_tokens)

    @classmethod
    def _pick_position_column(cls, headers: list[str]) -> int | None:
        position_tokens = (
            "median_position",
            "median position",
            "position",
            "pozycja",
            "avg_position",
            "average position",
        )
        return cls._pick_text_column(headers, position_tokens)

    @staticmethod
    def _matches_target_domain(page_value: str, target_domain: str) -> bool:
        domain = target_domain.strip().lower()
        if not domain:
            return True
        raw = str(page_value).strip()
        if not raw:
            return True
        parsed = urlparse(raw)
        host = (parsed.netloc or parsed.path).strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host == domain

    @classmethod
    def _pick_metric_columns_yoy(
        cls,
        headers: list[str],
        rows: list[list[object]],
        run_date: date,
    ) -> tuple[int | None, int | None]:
        normalized = [cls._normalize_text(header) for header in headers]
        year_current = str(run_date.year)
        year_previous = str(run_date.year - 1)

        def score_header(name: str, ratio: float, for_previous: bool) -> float:
            if "date" in name or "data" in name or "dzien" in name or "day" in name:
                return -10.0
            score = ratio * 3.0
            if for_previous:
                if year_previous in name:
                    score += 8.0
                if any(token in name for token in ("prev", "previous", "last year", "ly", "rok temu", "ubieg")):
                    score += 4.0
            else:
                if year_current in name:
                    score += 8.0
                if any(token in name for token in ("current", "this year", "ytd", "biez", "aktual")):
                    score += 4.0
            if any(token in name for token in ("trend", "volume", "search", "score", "traffic", "value")):
                score += 1.0
            return score

        scored_current: list[tuple[float, int]] = []
        scored_previous: list[tuple[float, int]] = []
        numeric_candidates: list[tuple[float, int]] = []

        for idx, name in enumerate(normalized):
            ratio = cls._numeric_ratio_for_column(rows, idx)
            if ratio < 0.35:
                continue
            numeric_candidates.append((ratio, idx))
            scored_current.append((score_header(name, ratio, for_previous=False), idx))
            scored_previous.append((score_header(name, ratio, for_previous=True), idx))

        scored_current.sort(reverse=True)
        scored_previous.sort(reverse=True)

        current_idx = scored_current[0][1] if scored_current and scored_current[0][0] >= 4.0 else None
        previous_idx = scored_previous[0][1] if scored_previous and scored_previous[0][0] >= 4.0 else None

        if current_idx is not None and previous_idx == current_idx:
            previous_idx = None
            for score, idx in scored_previous:
                if idx != current_idx and score >= 4.0:
                    previous_idx = idx
                    break

        return current_idx, previous_idx

    @classmethod
    def _pick_primary_metric_column(
        cls,
        headers: list[str],
        rows: list[list[object]],
        run_date: date,
    ) -> int | None:
        normalized = [cls._normalize_text(header) for header in headers]
        year_current = str(run_date.year)
        candidates: list[tuple[float, int]] = []
        for idx, name in enumerate(normalized):
            ratio = cls._numeric_ratio_for_column(rows, idx)
            if ratio < 0.35:
                continue
            score = ratio * 3.0
            if year_current in name:
                score += 3.0
            if any(token in name for token in ("current", "this year", "volume", "search", "trend", "score", "traffic", "value")):
                score += 2.0
            if "date" in name or "data" in name or "dzien" in name or "day" in name:
                score -= 6.0
            candidates.append((score, idx))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        if candidates[0][0] <= 0:
            return None
        return candidates[0][1]

    @classmethod
    def _pick_timestamp_column(cls, headers: list[str]) -> int | None:
        timestamp_tokens = (
            "timestamp",
            "date",
            "data",
            "day",
            "dzien",
        )
        return cls._pick_text_column(headers, timestamp_tokens)

    @classmethod
    def _pick_preferred_metric_column(
        cls,
        headers: list[str],
        rows: list[list[object]],
        run_date: date,
    ) -> int | None:
        normalized = [cls._normalize_text(header) for header in headers]
        preferred_tokens = (
            "seo_score",
            "seo score",
            "sv",
            "clicks_estimation",
            "clicks estimation",
            "clicks",
            "impressions_estimation",
            "impressions estimation",
        )
        for token in preferred_tokens:
            idx = cls._pick_text_column(headers, (token,))
            if idx is None:
                continue
            ratio = cls._numeric_ratio_for_column(rows, idx)
            if ratio >= 0.35:
                return idx
        return cls._pick_primary_metric_column(headers, rows, run_date)

    def _read_spreadsheet_tables(
        self,
        spreadsheet_id: str,
        max_sheets: int = 8,
        max_rows_per_sheet: int = 600,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        sheets = self._sheets_service()
        meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_rows = meta.get("sheets", []) if isinstance(meta, dict) else []

        all_titles: list[str] = []
        for row in sheet_rows:
            if not isinstance(row, dict):
                continue
            properties = row.get("properties")
            if not isinstance(properties, dict):
                continue
            title = str(properties.get("title", "")).strip()
            if title and not self._is_ignored_sheet_title(title):
                all_titles.append(title)

        selected_titles = all_titles[: max(1, max_sheets)]
        tables: list[dict[str, Any]] = []
        errors: list[str] = []

        for title in selected_titles:
            safe_title = title.replace("'", "''")
            range_name = f"'{safe_title}'!A1:ZZ"
            try:
                payload = sheets.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    majorDimension="ROWS",
                ).execute()
            except Exception as exc:
                errors.append(f"Sheet read failed for '{title}': {exc}")
                continue

            values = payload.get("values", []) if isinstance(payload, dict) else []
            if not isinstance(values, list) or len(values) < 2:
                continue

            header_row = values[0]
            if not isinstance(header_row, list):
                continue
            headers = [str(cell).strip() for cell in header_row]
            body_rows = [row for row in values[1 : max_rows_per_sheet + 1] if isinstance(row, list)]
            tables.append({"sheet": title, "headers": headers, "rows": body_rows})

        return tables, selected_titles, errors

    @staticmethod
    def _dedupe_rows_by_trend(
        rows: list[dict[str, Any]],
        value_key: str = "current_value",
    ) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for row in rows:
            trend = str(row.get("trend", "")).strip()
            if not trend:
                continue
            key = trend.lower()
            if key not in deduped:
                deduped[key] = row
                continue
            current_best = float(deduped[key].get(value_key, 0.0))
            current_new = float(row.get(value_key, 0.0))
            if current_new > current_best:
                deduped[key] = row
        return list(deduped.values())

    def _resolve_sheet_file(self, reference: str) -> dict[str, Any] | None:
        file_id = self._extract_drive_id(reference)
        if not file_id:
            return None
        meta = self._get_drive_file(file_id)
        if not meta:
            return None
        if str(meta.get("mimeType", "")).strip() != self.SHEET_MIME:
            return None
        return meta

    def _extract_non_brand_yoy_rows(
        self,
        tables: list[dict[str, Any]],
        run_date: date,
        top_rows: int,
        target_domain: str = "",
        yoy_tolerance_days: int = 45,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for table in tables:
            if not isinstance(table, dict):
                continue
            headers = table.get("headers", [])
            body_rows = table.get("rows", [])
            sheet_name = str(table.get("sheet", "")).strip()
            if not isinstance(headers, list) or not isinstance(body_rows, list):
                continue

            keyword_idx = self._pick_keyword_column(headers)
            page_idx = self._pick_page_column(headers)
            position_idx = self._pick_position_column(headers)
            timestamp_idx = self._pick_timestamp_column(headers)
            metric_idx = self._pick_preferred_metric_column(headers, body_rows, run_date)
            if keyword_idx is None or metric_idx is None:
                continue

            if timestamp_idx is None:
                continue

            snapshot_dates: list[date] = []
            for row in body_rows:
                if not isinstance(row, list) or timestamp_idx >= len(row):
                    continue
                row_date = self._parse_flexible_date(row[timestamp_idx])
                if row_date:
                    snapshot_dates.append(row_date)
            if not snapshot_dates:
                continue

            current_snapshot = max(day for day in snapshot_dates if day <= run_date) if any(
                day <= run_date for day in snapshot_dates
            ) else max(snapshot_dates)
            target_prev = current_snapshot - timedelta(weeks=52)
            previous_snapshot = min(
                snapshot_dates,
                key=lambda day: abs((day - target_prev).days),
            )
            if abs((previous_snapshot - target_prev).days) > max(7, yoy_tolerance_days):
                continue

            current_by_trend: dict[str, float] = {}
            previous_by_trend: dict[str, float] = {}
            trend_labels: dict[str, str] = {}
            page_by_trend: dict[str, str] = {}
            position_by_trend: dict[str, float] = {}

            for row in body_rows:
                if not isinstance(row, list):
                    continue
                trend = str(row[keyword_idx]).strip() if keyword_idx < len(row) else ""
                if not trend or not self._is_non_brand_phrase(trend):
                    continue
                if page_idx is not None and page_idx < len(row):
                    page_value = str(row[page_idx]).strip()
                    if page_value and not self._matches_target_domain(page_value, target_domain):
                        continue

                row_date = None
                if timestamp_idx < len(row):
                    row_date = self._parse_flexible_date(row[timestamp_idx])
                if row_date is None:
                    continue

                metric_value = self._parse_numeric_cell(
                    row[metric_idx] if metric_idx < len(row) else None
                )
                if metric_value is None:
                    continue
                page_value = ""
                if page_idx is not None and page_idx < len(row):
                    page_value = str(row[page_idx]).strip()
                position_value = None
                if position_idx is not None and position_idx < len(row):
                    position_value = self._parse_numeric_cell(row[position_idx])

                key = trend.lower()
                trend_labels.setdefault(key, trend)
                if row_date == current_snapshot:
                    current_by_trend[key] = max(
                        metric_value,
                        current_by_trend.get(key, float("-inf")),
                    )
                    if page_value:
                        page_by_trend[key] = page_value
                    if position_value is not None:
                        prev_pos = position_by_trend.get(key)
                        if prev_pos is None or position_value < prev_pos:
                            position_by_trend[key] = float(position_value)
                elif row_date == previous_snapshot:
                    previous_by_trend[key] = max(
                        metric_value,
                        previous_by_trend.get(key, float("-inf")),
                    )

            all_keys = sorted(set(current_by_trend) | set(previous_by_trend))
            for key in all_keys:
                current_value = float(current_by_trend.get(key, 0.0))
                previous_value = float(previous_by_trend.get(key, 0.0))
                if current_value == 0 and previous_value == 0:
                    continue
                delta = current_value - previous_value
                if previous_value <= 0:
                    delta_pct = 100.0 if current_value > 0 else 0.0
                else:
                    delta_pct = (delta / previous_value) * 100.0
                out.append(
                    {
                        "trend": trend_labels.get(key, key),
                        "current_value": current_value,
                        "previous_value": previous_value,
                        "delta_value": delta,
                        "delta_pct": delta_pct,
                        "sheet": sheet_name,
                        "current_snapshot_date": current_snapshot.isoformat(),
                        "previous_snapshot_date": previous_snapshot.isoformat(),
                        "page": page_by_trend.get(key, ""),
                        "median_position": float(position_by_trend.get(key, 0.0)),
                    }
                )

        deduped = self._dedupe_rows_by_trend(out, value_key="current_value")
        deduped.sort(
            key=lambda item: (
                float(item.get("current_value", 0.0)),
                abs(float(item.get("delta_value", 0.0))),
            ),
            reverse=True,
        )
        return deduped[: max(1, top_rows)]

    def _extract_non_brand_current_rows(
        self,
        tables: list[dict[str, Any]],
        run_date: date,
        top_rows: int,
        target_domain: str = "",
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for table in tables:
            if not isinstance(table, dict):
                continue
            headers = table.get("headers", [])
            body_rows = table.get("rows", [])
            sheet_name = str(table.get("sheet", "")).strip()
            if not isinstance(headers, list) or not isinstance(body_rows, list):
                continue

            keyword_idx = self._pick_keyword_column(headers)
            page_idx = self._pick_page_column(headers)
            position_idx = self._pick_position_column(headers)
            value_idx = self._pick_preferred_metric_column(headers, body_rows, run_date)
            date_idx = self._pick_timestamp_column(headers)
            if keyword_idx is None or value_idx is None:
                continue

            snapshot_date: date | None = None
            if date_idx is not None:
                snapshot_dates: list[date] = []
                for row in body_rows:
                    if not isinstance(row, list) or date_idx >= len(row):
                        continue
                    row_date = self._parse_flexible_date(row[date_idx])
                    if row_date:
                        snapshot_dates.append(row_date)
                if snapshot_dates:
                    if any(day <= run_date for day in snapshot_dates):
                        snapshot_date = max(day for day in snapshot_dates if day <= run_date)
                    else:
                        snapshot_date = max(snapshot_dates)

            for row in body_rows:
                if not isinstance(row, list):
                    continue
                trend = str(row[keyword_idx]).strip() if keyword_idx < len(row) else ""
                if not trend or not self._is_non_brand_phrase(trend):
                    continue
                if page_idx is not None and page_idx < len(row):
                    page_value = str(row[page_idx]).strip()
                    if page_value and not self._matches_target_domain(page_value, target_domain):
                        continue
                value = self._parse_numeric_cell(row[value_idx] if value_idx < len(row) else None)
                if value is None:
                    continue
                page_value = str(row[page_idx]).strip() if page_idx is not None and page_idx < len(row) else ""
                position_value = (
                    self._parse_numeric_cell(row[position_idx])
                    if position_idx is not None and position_idx < len(row)
                    else None
                )
                trend_date = None
                if date_idx is not None and date_idx < len(row):
                    trend_date = self._parse_flexible_date(row[date_idx])
                    if snapshot_date and trend_date != snapshot_date:
                        continue
                out.append(
                    {
                        "trend": trend,
                        "value": value,
                        "date": (trend_date or snapshot_date).isoformat() if (trend_date or snapshot_date) else "",
                        "sheet": sheet_name,
                        "page": page_value,
                        "median_position": float(position_value) if position_value is not None else 0.0,
                    }
                )

        deduped = self._dedupe_rows_by_trend(out, value_key="value")
        deduped.sort(key=lambda item: float(item.get("value", 0.0)), reverse=True)
        return deduped[: max(1, top_rows)]

    def _extract_non_brand_upcoming_rows(
        self,
        tables: list[dict[str, Any]],
        run_date: date,
        horizon_days: int,
        top_rows: int,
        target_domain: str = "",
    ) -> list[dict[str, Any]]:
        horizon_end = run_date + timedelta(days=max(1, horizon_days))
        out: list[dict[str, Any]] = []
        for table in tables:
            if not isinstance(table, dict):
                continue
            headers = table.get("headers", [])
            body_rows = table.get("rows", [])
            sheet_name = str(table.get("sheet", "")).strip()
            if not isinstance(headers, list) or not isinstance(body_rows, list):
                continue

            keyword_idx = self._pick_keyword_column(headers)
            page_idx = self._pick_page_column(headers)
            position_idx = self._pick_position_column(headers)
            date_idx = self._pick_timestamp_column(headers)
            value_idx = self._pick_preferred_metric_column(headers, body_rows, run_date)
            if keyword_idx is None:
                continue

            for row in body_rows:
                if not isinstance(row, list):
                    continue
                trend = str(row[keyword_idx]).strip() if keyword_idx < len(row) else ""
                if not trend or not self._is_non_brand_phrase(trend):
                    continue
                if page_idx is not None and page_idx < len(row):
                    page_value = str(row[page_idx]).strip()
                    if page_value and not self._matches_target_domain(page_value, target_domain):
                        continue

                trend_date: date | None = None
                if date_idx is not None and date_idx < len(row):
                    trend_date = self._parse_flexible_date(row[date_idx])
                    if trend_date and not (run_date < trend_date <= horizon_end):
                        continue

                value = 0.0
                if value_idx is not None and value_idx < len(row):
                    parsed = self._parse_numeric_cell(row[value_idx])
                    if parsed is not None:
                        value = parsed
                page_value = str(row[page_idx]).strip() if page_idx is not None and page_idx < len(row) else ""
                position_value = (
                    self._parse_numeric_cell(row[position_idx])
                    if position_idx is not None and position_idx < len(row)
                    else None
                )
                out.append(
                    {
                        "date": trend_date.isoformat() if trend_date else "",
                        "trend": trend,
                        "value": value,
                        "sheet": sheet_name,
                        "page": page_value,
                        "median_position": float(position_value) if position_value is not None else 0.0,
                    }
                )

        deduped = self._dedupe_rows_by_trend(out, value_key="value")
        deduped.sort(
            key=lambda item: (
                float(item.get("value", 0.0)),
                item.get("date", ""),
            ),
            reverse=True,
        )
        return deduped[: max(1, top_rows)]

    def _read_status_entries(self, spreadsheet_id: str, run_date: date) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        sheets = self._sheets_service()
        meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_rows = meta.get("sheets", []) if isinstance(meta, dict) else []

        all_titles: list[str] = []
        for row in sheet_rows:
            if not isinstance(row, dict):
                continue
            properties = row.get("properties")
            if not isinstance(properties, dict):
                continue
            title = str(properties.get("title", "")).strip()
            if title:
                all_titles.append(title)

        candidate_titles = [
            title for title in all_titles if not self._is_ignored_sheet_title(title)
        ]

        wanted_years = {str(run_date.year), str(run_date.year - 1)}
        selected_titles = [
            title
            for title in candidate_titles
            if any(year in title for year in wanted_years)
        ]
        if not selected_titles:
            selected_titles = [
                title
                for title in candidate_titles
                if self._status_name_score(title) > 0
            ][:4]
        if not selected_titles:
            selected_titles = candidate_titles[:3]

        entries: list[dict[str, Any]] = []
        errors: list[str] = []

        for title in selected_titles:
            safe_title = title.replace("'", "''")
            range_name = f"'{safe_title}'!A1:ZZ"
            try:
                payload = sheets.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    majorDimension="ROWS",
                ).execute()
            except Exception as exc:
                errors.append(f"Sheet read failed for '{title}': {exc}")
                continue

            values = payload.get("values", []) if isinstance(payload, dict) else []
            if not isinstance(values, list) or len(values) < 2:
                continue

            header_row = values[0]
            if not isinstance(header_row, list):
                continue
            headers = [str(cell).strip() for cell in header_row]

            date_idx = self._pick_date_column(headers)
            if date_idx is None:
                errors.append(f"Date column not found in sheet '{title}'.")
                continue

            topic_idx = self._pick_text_column(headers, ("topic", "temat", "title", "issue"))
            note_idx = self._pick_text_column(headers, ("status", "note", "comment", "akcja", "action"))

            for row in values[1:]:
                if not isinstance(row, list):
                    continue
                raw_date = row[date_idx] if date_idx < len(row) else ""
                row_date = self._parse_flexible_date(raw_date)
                if not row_date or row_date > run_date:
                    continue

                topic = ""
                if topic_idx is not None and topic_idx < len(row):
                    topic = str(row[topic_idx]).strip()

                summary = ""
                if note_idx is not None and note_idx < len(row):
                    summary = str(row[note_idx]).strip()

                if not topic:
                    candidates = [
                        str(cell).strip()
                        for idx, cell in enumerate(row)
                        if idx != date_idx and str(cell).strip()
                    ]
                    if candidates:
                        topic = candidates[0][:140]
                        summary = summary or (candidates[1][:240] if len(candidates) > 1 else "")

                entries.append(
                    {
                        "date": row_date.isoformat(),
                        "sheet": title,
                        "topic": topic,
                        "summary": summary,
                    }
                )

        entries.sort(key=lambda item: item.get("date", ""), reverse=True)
        return entries[: self.max_status_rows], selected_titles, errors

    def collect_status_updates(self, run_date: date) -> dict[str, Any]:
        if not self.client_secret_path:
            return {
                "enabled": False,
                "errors": ["Google credentials not configured for status log."],
                "entries": [],
            }

        meta = self._resolve_status_file()
        if not meta:
            return {
                "enabled": False,
                "errors": ["Status spreadsheet not found in configured Drive scope."],
                "entries": [],
            }

        spreadsheet_id = str(meta.get("id", "")).strip()
        if not spreadsheet_id:
            return {
                "enabled": False,
                "errors": ["Status spreadsheet id is empty."],
                "entries": [],
            }

        try:
            entries, selected_sheets, read_errors = self._read_status_entries(
                spreadsheet_id=spreadsheet_id,
                run_date=run_date,
            )
        except Exception as exc:
            return {
                "enabled": False,
                "errors": [f"Status spreadsheet read failed: {exc}"],
                "entries": [],
            }

        return {
            "enabled": True,
            "source": "Google Sheets SEO status log",
            "file_id": spreadsheet_id,
            "file_name": str(meta.get("name", "")).strip(),
            "url": str(meta.get("webViewLink", "")).strip(),
            "selected_sheets": selected_sheets,
            "entries": entries,
            "errors": read_errors,
        }

    def collect_product_trends(
        self,
        run_date: date,
        comparison_sheet_reference: str = "",
        upcoming_sheet_reference: str = "",
        current_sheet_reference: str = "",
        target_domain: str = "",
        top_rows: int = 12,
        horizon_days: int = 31,
    ) -> dict[str, Any]:
        if not self.client_secret_path:
            return {
                "enabled": False,
                "errors": ["Google credentials not configured for product trend sheets."],
                "top_yoy_non_brand": [],
                "upcoming_31d": [],
                "current_non_brand": [],
            }

        effective_top_rows = max(1, top_rows)
        effective_horizon = max(1, horizon_days)
        errors: list[str] = []

        comparison_meta = None
        comparison_ref = comparison_sheet_reference.strip() or current_sheet_reference.strip()
        if comparison_ref:
            comparison_meta = self._resolve_sheet_file(comparison_ref)
            if comparison_meta is None:
                errors.append("Comparison trend sheet not found or is not a Google Sheet.")
        else:
            errors.append("Comparison trend sheet reference is empty.")

        upcoming_meta = None
        if upcoming_sheet_reference.strip():
            upcoming_meta = self._resolve_sheet_file(upcoming_sheet_reference)
            if upcoming_meta is None:
                errors.append("Upcoming trend sheet not found or is not a Google Sheet.")
        else:
            errors.append("Upcoming trend sheet reference is empty.")

        current_meta = None
        if current_sheet_reference.strip():
            current_meta = self._resolve_sheet_file(current_sheet_reference)
            if current_meta is None:
                errors.append("Current trend sheet not found or is not a Google Sheet.")
        else:
            errors.append("Current trend sheet reference is empty.")

        comparison_rows: list[dict[str, Any]] = []
        comparison_selected_sheets: list[str] = []
        if comparison_meta:
            comparison_tables, comparison_selected_sheets, read_errors = self._read_spreadsheet_tables(
                spreadsheet_id=str(comparison_meta.get("id", "")).strip(),
                max_sheets=8,
                max_rows_per_sheet=700,
            )
            errors.extend(read_errors)
            comparison_rows = self._extract_non_brand_yoy_rows(
                tables=comparison_tables,
                run_date=run_date,
                top_rows=effective_top_rows,
                target_domain=target_domain,
            )
            if not comparison_rows:
                errors.append(
                    "No non-brand YoY trend rows detected in comparison sheet; report will use GSC fallback for YoY trends."
                )

        upcoming_rows: list[dict[str, Any]] = []
        upcoming_selected_sheets: list[str] = []
        if upcoming_meta:
            upcoming_tables, upcoming_selected_sheets, read_errors = self._read_spreadsheet_tables(
                spreadsheet_id=str(upcoming_meta.get("id", "")).strip(),
                max_sheets=8,
                max_rows_per_sheet=700,
            )
            errors.extend(read_errors)
            upcoming_rows = self._extract_non_brand_upcoming_rows(
                tables=upcoming_tables,
                run_date=run_date,
                horizon_days=effective_horizon,
                top_rows=effective_top_rows,
                target_domain=target_domain,
            )
            if not upcoming_rows:
                errors.append("No non-brand upcoming trend rows detected in the configured horizon.")

        current_rows: list[dict[str, Any]] = []
        current_selected_sheets: list[str] = []
        if current_meta:
            current_tables, current_selected_sheets, read_errors = self._read_spreadsheet_tables(
                spreadsheet_id=str(current_meta.get("id", "")).strip(),
                max_sheets=8,
                max_rows_per_sheet=700,
            )
            errors.extend(read_errors)
            current_rows = self._extract_non_brand_current_rows(
                tables=current_tables,
                run_date=run_date,
                top_rows=effective_top_rows,
                target_domain=target_domain,
            )
            if not current_rows:
                errors.append("No non-brand current trend rows detected.")

        enabled = bool(comparison_rows or upcoming_rows or current_rows)
        return {
            "enabled": enabled,
            "source": "Google Sheets product trend trackers",
            "top_rows": effective_top_rows,
            "horizon_days": effective_horizon,
            "target_domain": target_domain.strip().lower(),
            "comparison_sheet": {
                "file_id": str(comparison_meta.get("id", "")).strip() if isinstance(comparison_meta, dict) else "",
                "file_name": str(comparison_meta.get("name", "")).strip() if isinstance(comparison_meta, dict) else "",
                "url": str(comparison_meta.get("webViewLink", "")).strip() if isinstance(comparison_meta, dict) else "",
                "selected_sheets": comparison_selected_sheets,
            },
            "upcoming_sheet": {
                "file_id": str(upcoming_meta.get("id", "")).strip() if isinstance(upcoming_meta, dict) else "",
                "file_name": str(upcoming_meta.get("name", "")).strip() if isinstance(upcoming_meta, dict) else "",
                "url": str(upcoming_meta.get("webViewLink", "")).strip() if isinstance(upcoming_meta, dict) else "",
                "selected_sheets": upcoming_selected_sheets,
            },
            "current_sheet": {
                "file_id": str(current_meta.get("id", "")).strip() if isinstance(current_meta, dict) else "",
                "file_name": str(current_meta.get("name", "")).strip() if isinstance(current_meta, dict) else "",
                "url": str(current_meta.get("webViewLink", "")).strip() if isinstance(current_meta, dict) else "",
                "selected_sheets": current_selected_sheets,
            },
            "top_yoy_non_brand": comparison_rows,
            "upcoming_31d": upcoming_rows,
            "current_non_brand": current_rows,
            "errors": errors,
        }

    def collect_trade_plan_context(
        self,
        *,
        run_date: date,
        sheet_reference: str,
        tab_name: str,
        current_window_start: date,
        current_window_end: date,
        previous_window_start: date,
        previous_window_end: date,
        yoy_window_start: date | None = None,
        yoy_window_end: date | None = None,
        yoy_sheet_reference: str = "",
        yoy_tab_name: str = "",
        include_yoy_enrichment: bool = True,
        top_rows: int = 12,
    ) -> dict[str, Any]:
        if not self.client_secret_path:
            return {
                "enabled": False,
                "errors": ["Google credentials not configured for trade plan sheet."],
                "rows": [],
            }

        sheet_meta = self._resolve_sheet_file(sheet_reference)
        if sheet_meta is None:
            return {
                "enabled": False,
                "errors": ["Trade plan sheet not found or is not a Google Sheet."],
                "rows": [],
            }

        spreadsheet_id = str(sheet_meta.get("id", "")).strip()
        if not spreadsheet_id:
            return {
                "enabled": False,
                "errors": ["Trade plan sheet id is missing."],
                "rows": [],
            }

        safe_tab = tab_name.replace("'", "''").strip()
        if not safe_tab:
            return {
                "enabled": False,
                "errors": ["Trade plan tab is empty."],
                "rows": [],
            }

        sheets = self._sheets_service()
        try:
            payload = sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f"'{safe_tab}'!A1:ZZ",
                majorDimension="ROWS",
            ).execute()
        except Exception as exc:
            return {
                "enabled": False,
                "errors": [f"Trade plan tab read failed: {exc}"],
                "rows": [],
            }

        values = payload.get("values", []) if isinstance(payload, dict) else []
        if not isinstance(values, list) or len(values) < 2:
            return {
                "enabled": False,
                "errors": [f"Trade plan tab '{tab_name}' has no usable rows."],
                "rows": [],
            }

        header_candidates: list[tuple[int, list[str], int | None, int | None, int | None, int]] = []
        date_header_tokens = (
            "campaign start",
            "week start",
            "start date",
            "launch date",
            "week commencing",
            "week",
            "date",
            "data",
            "dzien",
            "tydzien",
        )
        campaign_header_tokens = (
            "campaign name",
            "initiative name",
            "campaign title",
            "activity name",
            "project name",
            "nazwa kampanii",
            "nazwa akcji",
            "nazwa",
        )
        campaign_fallback_tokens = (
            "campaign",
            "initiative",
            "activity",
            "project",
            "event",
            "akcja",
            "projekt",
        )
        max_header_scan = min(8, len(values))
        for idx in range(max_header_scan):
            row = values[idx]
            if not isinstance(row, list):
                continue
            headers = [str(cell).strip() for cell in row]
            if not any(headers):
                continue
            date_idx_probe = self._pick_text_column(
                headers,
                date_header_tokens,
            )
            campaign_idx_probe = self._pick_text_column(
                headers,
                campaign_header_tokens,
            )
            if campaign_idx_probe is None:
                campaign_idx_probe = self._pick_text_column(
                    headers,
                    campaign_fallback_tokens,
                )
            channel_idx_probe = self._pick_text_column(
                headers,
                ("channel", "kanal", "source", "medium", "publisher", "platform"),
            )
            start_idx_probe = self._pick_text_column(
                headers,
                ("campaign start", "start", "week start", "from"),
            )
            end_idx_probe = self._pick_text_column(
                headers,
                ("campaign end", "end", "week end", "to"),
            )
            score = 0
            score += 4 if date_idx_probe is not None else 0
            score += 3 if campaign_idx_probe is not None else 0
            score += 2 if channel_idx_probe is not None else 0
            score += 2 if start_idx_probe is not None else 0
            score += 1 if end_idx_probe is not None else 0
            score += min(sum(1 for h in headers if h), 8) // 3
            normalized_headers = [self._normalize_text(h) for h in headers]
            # Prefer compact "label" rows over long question/description rows.
            non_empty_lengths = [len(h) for h in headers if h]
            if non_empty_lengths:
                avg_len = sum(non_empty_lengths) / len(non_empty_lengths)
                if avg_len <= 18:
                    score += 5
                elif avg_len >= 35:
                    score -= 4
            question_like = sum(1 for h in headers if "?" in h)
            if question_like >= 4:
                score -= 4
            # Strong hint that this is the actual business header row.
            if any("campaign name" in h for h in normalized_headers):
                score += 6
            if any(h in {"priority", "type", "executing team"} for h in normalized_headers):
                score += 3
            if score > 0:
                header_candidates.append(
                    (idx, headers, date_idx_probe, campaign_idx_probe, channel_idx_probe, score)
                )

        if not header_candidates:
            return {
                "enabled": False,
                "errors": [f"Trade plan tab '{tab_name}' has no usable rows."],
                "rows": [],
            }

        header_candidates.sort(key=lambda item: item[5], reverse=True)
        header_row_idx, headers, _, _, _, _ = header_candidates[0]
        body_rows = [row for row in values[header_row_idx + 1 :] if isinstance(row, list)]
        if not headers or not body_rows:
            return {
                "enabled": False,
                "errors": [f"Trade plan tab '{tab_name}' has no usable rows."],
                "rows": [],
            }

        date_idx = self._pick_text_column(
            headers,
            date_header_tokens,
        )
        campaign_start_idx = self._pick_text_column(
            headers,
            ("campaign start", "start", "week start", "from", "launch date", "week commencing"),
        )
        campaign_end_idx = self._pick_text_column(
            headers,
            ("campaign end", "end", "week end", "to", "finish date", "close date"),
        )
        campaign_idx = self._pick_text_column(
            headers,
            campaign_header_tokens,
        )
        if campaign_idx is None:
            campaign_idx = self._pick_text_column(
                headers,
                campaign_fallback_tokens,
            )
        channel_idx = self._pick_text_column(
            headers,
            ("channel", "kanal", "source", "medium", "publisher", "platform", "media"),
        )
        team_idx = self._pick_text_column(
            headers,
            ("executing team", "team", "owner", "responsible"),
        )
        type_idx = self._pick_text_column(
            headers,
            ("type", "campaign/content"),
        )
        category_idx = self._pick_text_column(
            headers,
            ("category", "kategoria", "department", "dzial", "segment"),
        )
        spend_idx = self._pick_text_column(
            headers,
            ("spend", "cost", "koszt", "wydatki"),
        )
        duration_idx = self._pick_text_column(
            headers,
            ("duration", "total duration", "duration [days]"),
        )
        core_phase_idx = self._pick_text_column(
            headers,
            ("core phase", "core phase [days]"),
        )
        impressions_idx = self._pick_text_column(
            headers,
            ("impressions", "wyswietlenia", "odsłony", "odslony"),
        )
        clicks_idx = self._pick_text_column(
            headers,
            ("clicks", "klikniecia", "click"),
        )

        def _parse_trade_plan_date(raw: object) -> date | None:
            parsed = self._parse_flexible_date(raw)
            if parsed is not None:
                return parsed
            text = str(raw).strip()
            if not text:
                return None
            # Accept day/month without year (common in trade plans, e.g. "12/01", "05.03").
            compact = re.sub(r"\s+", "", text)
            match = re.search(r"(?P<d>\d{1,2})[./-](?P<m>\d{1,2})(?:[./-](?P<y>\d{2,4}))?", compact)
            if not match:
                return None
            day = int(match.group("d"))
            month = int(match.group("m"))
            year_raw = match.group("y")
            if year_raw:
                year = int(year_raw)
                if year < 100:
                    year += 2000
                try:
                    return date(year, month, day)
                except ValueError:
                    return None
            # Infer year nearest to analyzed window (not today's run date).
            anchor_day = current_window_end
            candidates: list[date] = []
            for year in (anchor_day.year - 1, anchor_day.year, anchor_day.year + 1):
                try:
                    candidates.append(date(year, month, day))
                except ValueError:
                    continue
            if not candidates:
                return None
            return min(candidates, key=lambda item: abs((item - anchor_day).days))

        def _derive_row_day(row: list[object]) -> date | None:
            if date_idx is not None and date_idx < len(row):
                parsed = _parse_trade_plan_date(row[date_idx])
                if parsed is not None:
                    return parsed
            start_day = None
            end_day = None
            if campaign_start_idx is not None and campaign_start_idx < len(row):
                start_day = _parse_trade_plan_date(row[campaign_start_idx])
            if campaign_end_idx is not None and campaign_end_idx < len(row):
                end_day = _parse_trade_plan_date(row[campaign_end_idx])
            if start_day and end_day:
                # Assign to the start date for bucketing.
                return start_day
            return start_day or end_day

        if date_idx is None and campaign_start_idx is None and campaign_end_idx is None:
            return {
                "enabled": False,
                "errors": [f"Trade plan tab '{tab_name}' is missing a date-like column."],
                "rows": [],
            }

        # Fill down sparse values produced by merged cells in Sheets tabs.
        body_rows = self._fill_down_sparse_rows(
            body_rows,
            filldown_indices=(
                date_idx,
                campaign_start_idx,
                campaign_end_idx,
                campaign_idx,
                channel_idx,
                team_idx,
                type_idx,
                category_idx,
            ),
        )

        def _bucket(row_day: date) -> str:
            if current_window_start <= row_day <= current_window_end:
                return "current"
            if previous_window_start <= row_day <= previous_window_end:
                return "previous"
            if (
                isinstance(yoy_window_start, date)
                and isinstance(yoy_window_end, date)
                and yoy_window_start <= row_day <= yoy_window_end
            ):
                return "yoy"
            if current_window_end < row_day <= current_window_end + timedelta(days=31):
                return "forward_31d"
            return ""

        parsed_rows: list[dict[str, Any]] = []
        channel_summary: dict[str, dict[str, float]] = {}
        campaign_summary: dict[str, dict[str, float | str]] = {}

        for row in body_rows:
            row_day = _derive_row_day(row)
            if row_day is None:
                continue
            bucket = _bucket(row_day)
            if not bucket:
                continue

            campaign = (
                str(row[campaign_idx]).strip()
                if campaign_idx is not None and campaign_idx < len(row)
                else ""
            ) or "Unlabeled campaign"
            channel = (
                str(row[channel_idx]).strip()
                if channel_idx is not None and channel_idx < len(row)
                else ""
            )
            if not channel and team_idx is not None and team_idx < len(row):
                channel = str(row[team_idx]).strip()
            if not channel and type_idx is not None and type_idx < len(row):
                channel = str(row[type_idx]).strip()
            channel = channel or "Unknown channel"
            category = (
                str(row[category_idx]).strip()
                if category_idx is not None and category_idx < len(row)
                else ""
            ) or "Unspecified category"
            spend = (
                self._parse_numeric_cell(row[spend_idx])
                if spend_idx is not None and spend_idx < len(row)
                else None
            )
            impressions = (
                self._parse_numeric_cell(row[impressions_idx])
                if impressions_idx is not None and impressions_idx < len(row)
                else None
            )
            clicks = (
                self._parse_numeric_cell(row[clicks_idx])
                if clicks_idx is not None and clicks_idx < len(row)
                else None
            )
            duration_days = (
                self._parse_numeric_cell(row[duration_idx])
                if duration_idx is not None and duration_idx < len(row)
                else None
            )
            core_phase_days = (
                self._parse_numeric_cell(row[core_phase_idx])
                if core_phase_idx is not None and core_phase_idx < len(row)
                else None
            )
            # Trade-plan tabs often have no spend/clicks/impressions.
            # Use planning intensity proxy from duration/core-phase to keep YoY comparability.
            if spend is None:
                if core_phase_days is not None:
                    spend = float(core_phase_days)
                elif duration_days is not None:
                    spend = float(duration_days)
                else:
                    spend = 0.0

            parsed_rows.append(
                {
                    "date": row_day.isoformat(),
                    "bucket": bucket,
                    "campaign": campaign,
                    "channel": channel,
                    "category": category,
                    "spend": float(spend or 0.0),
                    "duration_days": float(duration_days or 0.0),
                    "core_phase_days": float(core_phase_days or 0.0),
                    "impressions": float(impressions or 0.0),
                    "clicks": float(clicks or 0.0),
                }
            )

            channel_key = channel.strip().lower()
            channel_row = channel_summary.setdefault(
                channel_key,
                {
                    "channel": channel,
                    "current_spend": 0.0,
                    "previous_spend": 0.0,
                    "yoy_spend": 0.0,
                    "current_impressions": 0.0,
                    "previous_impressions": 0.0,
                    "yoy_impressions": 0.0,
                    "current_clicks": 0.0,
                    "previous_clicks": 0.0,
                    "yoy_clicks": 0.0,
                },
            )
            if bucket == "current":
                channel_row["current_spend"] += float(spend or 0.0)
                channel_row["current_impressions"] += float(impressions or 0.0)
                channel_row["current_clicks"] += float(clicks or 0.0)
            elif bucket == "previous":
                channel_row["previous_spend"] += float(spend or 0.0)
                channel_row["previous_impressions"] += float(impressions or 0.0)
                channel_row["previous_clicks"] += float(clicks or 0.0)
            elif bucket == "yoy":
                channel_row["yoy_spend"] += float(spend or 0.0)
                channel_row["yoy_impressions"] += float(impressions or 0.0)
                channel_row["yoy_clicks"] += float(clicks or 0.0)

            campaign_key = f"{campaign.strip().lower()}::{category.strip().lower()}"
            campaign_row = campaign_summary.setdefault(
                campaign_key,
                {
                    "campaign": campaign,
                    "category": category,
                    "first_date": row_day.isoformat(),
                    "last_date": row_day.isoformat(),
                    "current_spend": 0.0,
                    "previous_spend": 0.0,
                    "yoy_spend": 0.0,
                    "current_impressions": 0.0,
                    "previous_impressions": 0.0,
                    "yoy_impressions": 0.0,
                    "current_clicks": 0.0,
                    "previous_clicks": 0.0,
                    "yoy_clicks": 0.0,
                    "forward_spend": 0.0,
                },
            )
            if row_day.isoformat() < str(campaign_row.get("first_date", "")):
                campaign_row["first_date"] = row_day.isoformat()
            if row_day.isoformat() > str(campaign_row.get("last_date", "")):
                campaign_row["last_date"] = row_day.isoformat()
            if bucket == "current":
                campaign_row["current_spend"] = float(campaign_row.get("current_spend", 0.0)) + float(spend or 0.0)
                campaign_row["current_impressions"] = float(campaign_row.get("current_impressions", 0.0)) + float(impressions or 0.0)
                campaign_row["current_clicks"] = float(campaign_row.get("current_clicks", 0.0)) + float(clicks or 0.0)
            elif bucket == "previous":
                campaign_row["previous_spend"] = float(campaign_row.get("previous_spend", 0.0)) + float(spend or 0.0)
                campaign_row["previous_impressions"] = float(campaign_row.get("previous_impressions", 0.0)) + float(impressions or 0.0)
                campaign_row["previous_clicks"] = float(campaign_row.get("previous_clicks", 0.0)) + float(clicks or 0.0)
            elif bucket == "yoy":
                campaign_row["yoy_spend"] = float(campaign_row.get("yoy_spend", 0.0)) + float(spend or 0.0)
                campaign_row["yoy_impressions"] = float(campaign_row.get("yoy_impressions", 0.0)) + float(impressions or 0.0)
                campaign_row["yoy_clicks"] = float(campaign_row.get("yoy_clicks", 0.0)) + float(clicks or 0.0)
            elif bucket == "forward_31d":
                campaign_row["forward_spend"] = float(campaign_row.get("forward_spend", 0.0)) + float(spend or 0.0)

        channel_rows: list[dict[str, Any]] = []
        for row in channel_summary.values():
            prev_spend = float(row.get("previous_spend", 0.0))
            current_spend = float(row.get("current_spend", 0.0))
            yoy_spend = float(row.get("yoy_spend", 0.0))
            prev_impr = float(row.get("previous_impressions", 0.0))
            current_impr = float(row.get("current_impressions", 0.0))
            yoy_impr = float(row.get("yoy_impressions", 0.0))
            prev_clicks = float(row.get("previous_clicks", 0.0))
            current_clicks = float(row.get("current_clicks", 0.0))
            yoy_clicks = float(row.get("yoy_clicks", 0.0))
            yoy_hypothesis = self._trade_plan_yoy_hypothesis(
                current_spend=current_spend,
                yoy_spend=yoy_spend,
                current_clicks=current_clicks,
                yoy_clicks=yoy_clicks,
                current_impressions=current_impr,
                yoy_impressions=yoy_impr,
            )
            channel_rows.append(
                {
                    "channel": str(row.get("channel", "")),
                    "current_spend": current_spend,
                    "previous_spend": prev_spend,
                    "yoy_spend": yoy_spend,
                    "delta_spend": current_spend - prev_spend,
                    "delta_spend_pct": ((current_spend - prev_spend) / prev_spend * 100.0) if prev_spend else None,
                    "delta_spend_vs_yoy": current_spend - yoy_spend,
                    "delta_spend_pct_vs_yoy": ((current_spend - yoy_spend) / yoy_spend * 100.0) if yoy_spend else None,
                    "current_impressions": current_impr,
                    "previous_impressions": prev_impr,
                    "yoy_impressions": yoy_impr,
                    "delta_impressions": current_impr - prev_impr,
                    "delta_impressions_pct": ((current_impr - prev_impr) / prev_impr * 100.0) if prev_impr else None,
                    "delta_impressions_vs_yoy": current_impr - yoy_impr,
                    "delta_impressions_pct_vs_yoy": ((current_impr - yoy_impr) / yoy_impr * 100.0) if yoy_impr else None,
                    "current_clicks": current_clicks,
                    "previous_clicks": prev_clicks,
                    "yoy_clicks": yoy_clicks,
                    "delta_clicks": current_clicks - prev_clicks,
                    "delta_clicks_pct": ((current_clicks - prev_clicks) / prev_clicks * 100.0) if prev_clicks else None,
                    "delta_clicks_vs_yoy": current_clicks - yoy_clicks,
                    "delta_clicks_pct_vs_yoy": ((current_clicks - yoy_clicks) / yoy_clicks * 100.0) if yoy_clicks else None,
                    "yoy_hypothesis_impact": str(yoy_hypothesis.get("impact", "neutral")),
                    "yoy_hypothesis_confidence": int(yoy_hypothesis.get("confidence", 50) or 50),
                    "yoy_hypothesis_reason": str(yoy_hypothesis.get("reason", "")).strip(),
                }
            )
        channel_rows.sort(key=lambda item: abs(float(item.get("delta_spend", 0.0))), reverse=True)

        campaign_rows_full: list[dict[str, Any]] = []
        for item in campaign_summary.values():
            current_spend = float(item.get("current_spend", 0.0))
            previous_spend = float(item.get("previous_spend", 0.0))
            yoy_spend = float(item.get("yoy_spend", 0.0))
            current_impressions = float(item.get("current_impressions", 0.0))
            previous_impressions = float(item.get("previous_impressions", 0.0))
            yoy_impressions = float(item.get("yoy_impressions", 0.0))
            current_clicks = float(item.get("current_clicks", 0.0))
            previous_clicks = float(item.get("previous_clicks", 0.0))
            yoy_clicks = float(item.get("yoy_clicks", 0.0))
            yoy_hypothesis = self._trade_plan_yoy_hypothesis(
                current_spend=current_spend,
                yoy_spend=yoy_spend,
                current_clicks=current_clicks,
                yoy_clicks=yoy_clicks,
                current_impressions=current_impressions,
                yoy_impressions=yoy_impressions,
            )
            campaign_rows_full.append(
                {
                    **item,
                    "delta_spend": current_spend - previous_spend,
                    "delta_spend_pct": ((current_spend - previous_spend) / previous_spend * 100.0) if previous_spend else None,
                    "delta_spend_vs_yoy": current_spend - yoy_spend,
                    "delta_spend_pct_vs_yoy": ((current_spend - yoy_spend) / yoy_spend * 100.0) if yoy_spend else None,
                    "delta_impressions": current_impressions - previous_impressions,
                    "delta_impressions_pct": ((current_impressions - previous_impressions) / previous_impressions * 100.0) if previous_impressions else None,
                    "delta_impressions_vs_yoy": current_impressions - yoy_impressions,
                    "delta_impressions_pct_vs_yoy": ((current_impressions - yoy_impressions) / yoy_impressions * 100.0) if yoy_impressions else None,
                    "delta_clicks": current_clicks - previous_clicks,
                    "delta_clicks_pct": ((current_clicks - previous_clicks) / previous_clicks * 100.0) if previous_clicks else None,
                    "delta_clicks_vs_yoy": current_clicks - yoy_clicks,
                    "delta_clicks_pct_vs_yoy": ((current_clicks - yoy_clicks) / yoy_clicks * 100.0) if yoy_clicks else None,
                    "yoy_hypothesis_impact": str(yoy_hypothesis.get("impact", "neutral")),
                    "yoy_hypothesis_confidence": int(yoy_hypothesis.get("confidence", 50) or 50),
                    "yoy_hypothesis_reason": str(yoy_hypothesis.get("reason", "")).strip(),
                }
            )

        campaign_rows = sorted(
            campaign_rows_full,
            key=lambda item: (
                float(item.get("current_spend", 0.0)),
                float(item.get("current_impressions", 0.0)),
            ),
            reverse=True,
        )[: max(1, int(top_rows))]

        yoy_sheet_meta: dict[str, str] = {}
        if (
            include_yoy_enrichment
            and isinstance(yoy_window_start, date)
            and isinstance(yoy_window_end, date)
            and (yoy_sheet_reference.strip() or yoy_tab_name.strip())
        ):
            yoy_reference = yoy_sheet_reference.strip() or sheet_reference
            candidate_tab = yoy_tab_name.strip()
            if not candidate_tab and str(run_date.year) in tab_name:
                candidate_tab = tab_name.replace(str(run_date.year), str(run_date.year - 1))
            if not candidate_tab:
                candidate_tab = tab_name
            try:
                yoy_ctx = self.collect_trade_plan_context(
                    run_date=run_date,
                    sheet_reference=yoy_reference,
                    tab_name=candidate_tab,
                    current_window_start=yoy_window_start,
                    current_window_end=yoy_window_end,
                    previous_window_start=yoy_window_start - timedelta(days=7),
                    previous_window_end=yoy_window_start - timedelta(days=1),
                    yoy_window_start=None,
                    yoy_window_end=None,
                    yoy_sheet_reference="",
                    yoy_tab_name="",
                    include_yoy_enrichment=False,
                    top_rows=max(50, int(top_rows) * 8),
                )
                yoy_channel_rows_probe = yoy_ctx.get("channel_split", []) if isinstance(yoy_ctx, dict) else []
                has_direct_yoy = False
                if isinstance(yoy_channel_rows_probe, list):
                    has_direct_yoy = any(
                        isinstance(row, dict) and abs(float(row.get("current_spend", 0.0) or 0.0)) > 0.0
                        for row in yoy_channel_rows_probe
                    )
                if not has_direct_yoy:
                    yoy_ctx = self.collect_trade_plan_context(
                        run_date=run_date,
                        sheet_reference=yoy_reference,
                        tab_name=candidate_tab,
                        current_window_start=yoy_window_start - timedelta(days=14),
                        current_window_end=yoy_window_end + timedelta(days=14),
                        previous_window_start=yoy_window_start - timedelta(days=28),
                        previous_window_end=yoy_window_start - timedelta(days=15),
                        yoy_window_start=None,
                        yoy_window_end=None,
                        yoy_sheet_reference="",
                        yoy_tab_name="",
                        include_yoy_enrichment=False,
                        top_rows=max(50, int(top_rows) * 8),
                    )
                yoy_sheet = yoy_ctx.get("sheet", {}) if isinstance(yoy_ctx, dict) else {}
                if isinstance(yoy_sheet, dict):
                    yoy_sheet_meta = {
                        "id": str(yoy_sheet.get("id", "")).strip(),
                        "name": str(yoy_sheet.get("name", "")).strip(),
                        "url": str(yoy_sheet.get("url", "")).strip(),
                        "tab": str(yoy_sheet.get("tab", "")).strip(),
                    }
                yoy_channel_rows = yoy_ctx.get("channel_split", []) if isinstance(yoy_ctx, dict) else []
                channel_yoy_map: dict[str, dict[str, Any]] = {}
                if isinstance(yoy_channel_rows, list):
                    for row in yoy_channel_rows:
                        if not isinstance(row, dict):
                            continue
                        key = str(row.get("channel", "")).strip().lower()
                        if key:
                            channel_yoy_map[key] = row
                if channel_yoy_map:
                    for row in channel_rows:
                        key = str(row.get("channel", "")).strip().lower()
                        yoy_row = channel_yoy_map.get(key)
                        if not yoy_row:
                            continue
                        row["yoy_spend"] = float(yoy_row.get("current_spend", 0.0) or 0.0)
                        row["yoy_impressions"] = float(yoy_row.get("current_impressions", 0.0) or 0.0)
                        row["yoy_clicks"] = float(yoy_row.get("current_clicks", 0.0) or 0.0)
                        yoy_spend = float(row.get("yoy_spend", 0.0) or 0.0)
                        yoy_impr = float(row.get("yoy_impressions", 0.0) or 0.0)
                        yoy_clicks = float(row.get("yoy_clicks", 0.0) or 0.0)
                        current_spend = float(row.get("current_spend", 0.0) or 0.0)
                        current_impr = float(row.get("current_impressions", 0.0) or 0.0)
                        current_clicks = float(row.get("current_clicks", 0.0) or 0.0)
                        row["delta_spend_vs_yoy"] = current_spend - yoy_spend
                        row["delta_spend_pct_vs_yoy"] = (
                            ((current_spend - yoy_spend) / yoy_spend * 100.0) if yoy_spend else None
                        )
                        row["delta_impressions_vs_yoy"] = current_impr - yoy_impr
                        row["delta_impressions_pct_vs_yoy"] = (
                            ((current_impr - yoy_impr) / yoy_impr * 100.0) if yoy_impr else None
                        )
                        row["delta_clicks_vs_yoy"] = current_clicks - yoy_clicks
                        row["delta_clicks_pct_vs_yoy"] = (
                            ((current_clicks - yoy_clicks) / yoy_clicks * 100.0) if yoy_clicks else None
                        )
                        hypothesis = self._trade_plan_yoy_hypothesis(
                            current_spend=current_spend,
                            yoy_spend=yoy_spend,
                            current_clicks=current_clicks,
                            yoy_clicks=yoy_clicks,
                            current_impressions=current_impr,
                            yoy_impressions=yoy_impr,
                        )
                        row["yoy_hypothesis_impact"] = str(hypothesis.get("impact", "neutral"))
                        row["yoy_hypothesis_confidence"] = int(hypothesis.get("confidence", 50) or 50)
                        row["yoy_hypothesis_reason"] = str(hypothesis.get("reason", "")).strip()

                yoy_campaign_rows = yoy_ctx.get("campaign_rows", []) if isinstance(yoy_ctx, dict) else []
                campaign_yoy_map: dict[str, dict[str, Any]] = {}
                if isinstance(yoy_campaign_rows, list):
                    for row in yoy_campaign_rows:
                        if not isinstance(row, dict):
                            continue
                        key = (
                            str(row.get("campaign", "")).strip().lower()
                            + "::"
                            + str(row.get("category", "")).strip().lower()
                        )
                        if key != "::":
                            campaign_yoy_map[key] = row
                if campaign_yoy_map:
                    for row in campaign_rows:
                        key = (
                            str(row.get("campaign", "")).strip().lower()
                            + "::"
                            + str(row.get("category", "")).strip().lower()
                        )
                        yoy_row = campaign_yoy_map.get(key)
                        if not yoy_row:
                            continue
                        row["yoy_spend"] = float(yoy_row.get("current_spend", 0.0) or 0.0)
                        row["yoy_impressions"] = float(yoy_row.get("current_impressions", 0.0) or 0.0)
                        row["yoy_clicks"] = float(yoy_row.get("current_clicks", 0.0) or 0.0)
                        yoy_spend = float(row.get("yoy_spend", 0.0) or 0.0)
                        yoy_impr = float(row.get("yoy_impressions", 0.0) or 0.0)
                        yoy_clicks = float(row.get("yoy_clicks", 0.0) or 0.0)
                        current_spend = float(row.get("current_spend", 0.0) or 0.0)
                        current_impr = float(row.get("current_impressions", 0.0) or 0.0)
                        current_clicks = float(row.get("current_clicks", 0.0) or 0.0)
                        row["delta_spend_vs_yoy"] = current_spend - yoy_spend
                        row["delta_spend_pct_vs_yoy"] = (
                            ((current_spend - yoy_spend) / yoy_spend * 100.0) if yoy_spend else None
                        )
                        row["delta_impressions_vs_yoy"] = current_impr - yoy_impr
                        row["delta_impressions_pct_vs_yoy"] = (
                            ((current_impr - yoy_impr) / yoy_impr * 100.0) if yoy_impr else None
                        )
                        row["delta_clicks_vs_yoy"] = current_clicks - yoy_clicks
                        row["delta_clicks_pct_vs_yoy"] = (
                            ((current_clicks - yoy_clicks) / yoy_clicks * 100.0) if yoy_clicks else None
                        )
                        hypothesis = self._trade_plan_yoy_hypothesis(
                            current_spend=current_spend,
                            yoy_spend=yoy_spend,
                            current_clicks=current_clicks,
                            yoy_clicks=yoy_clicks,
                            current_impressions=current_impr,
                            yoy_impressions=yoy_impr,
                        )
                        row["yoy_hypothesis_impact"] = str(hypothesis.get("impact", "neutral"))
                        row["yoy_hypothesis_confidence"] = int(hypothesis.get("confidence", 50) or 50)
                        row["yoy_hypothesis_reason"] = str(hypothesis.get("reason", "")).strip()
            except Exception:
                # Keep base trade-plan output if YoY enrichment fails.
                pass

        return {
            "enabled": bool(parsed_rows),
            "source": "Google Sheets trade plan",
            "sheet": {
                "id": spreadsheet_id,
                "name": str(sheet_meta.get("name", "")).strip(),
                "url": str(sheet_meta.get("webViewLink", "")).strip(),
                "tab": tab_name,
            },
            "windows": {
                "current": {
                    "start": current_window_start.isoformat(),
                    "end": current_window_end.isoformat(),
                },
                "previous": {
                    "start": previous_window_start.isoformat(),
                    "end": previous_window_end.isoformat(),
                },
                "yoy": {
                    "start": yoy_window_start.isoformat() if isinstance(yoy_window_start, date) else "",
                    "end": yoy_window_end.isoformat() if isinstance(yoy_window_end, date) else "",
                },
            },
            "yoy_sheet": yoy_sheet_meta,
            "top_rows": max(1, int(top_rows)),
            "channel_split": channel_rows[: max(1, int(top_rows))],
            "campaign_rows": campaign_rows,
            "rows": parsed_rows[:1200],
            "errors": [],
        }
