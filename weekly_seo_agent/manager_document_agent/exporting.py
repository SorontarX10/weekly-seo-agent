from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from docx import Document as DocxDocument
from docx.text.paragraph import Paragraph


def export_markdown_like_to_docx(*, title: str, content: str, output_dir: Path, document_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{document_id}.docx"
    target = output_dir / filename

    docx = DocxDocument()
    content_lines = content.splitlines()
    normalized_title = _normalize_for_heading_compare(title)
    first_h1 = _extract_first_h1(content_lines)
    if normalized_title and normalized_title != _normalize_for_heading_compare(first_h1):
        docx.add_heading(title.strip(), level=0)

    for raw_line in content_lines:
        line = raw_line.rstrip()
        if not line:
            docx.add_paragraph("")
            continue
        if line.startswith("### "):
            _add_heading_with_inline_markdown(docx, line[4:].strip(), level=3)
            continue
        if line.startswith("## "):
            _add_heading_with_inline_markdown(docx, line[3:].strip(), level=2)
            continue
        if line.startswith("# "):
            _add_heading_with_inline_markdown(docx, line[2:].strip(), level=1)
            continue
        if line.startswith("- "):
            paragraph = docx.add_paragraph(style="List Bullet")
            _append_inline_markdown_runs(paragraph, line[2:].strip())
            continue
        paragraph = docx.add_paragraph()
        _append_inline_markdown_runs(paragraph, line)

    docx.save(str(target))
    return target


def build_drive_export_name(*, title: str, document_id: str) -> str:
    normalized_title = _normalize_drive_title(title)
    safe_id = re.sub(r"[^a-zA-Z0-9]+", "", (document_id or "").strip())[:8] or "DOC"
    return f"{normalized_title} [MDA-{safe_id}]"


def _normalize_drive_title(title: str) -> str:
    base = re.sub(r"\s+", " ", (title or "").strip())
    if not base:
        base = "Manager Document"
    base = re.sub(r"[\\/:*?\"<>|]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip(" .-_")
    if not base:
        base = "Manager Document"
    if len(base) > 120:
        base = base[:120].rstrip()
    return base


def _extract_first_h1(lines: list[str]) -> str:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            return line[2:].strip()
        return ""
    return ""


def _normalize_for_heading_compare(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _add_heading_with_inline_markdown(docx: DocxDocument, text: str, *, level: int) -> None:
    paragraph = docx.add_heading("", level=level)
    _append_inline_markdown_runs(paragraph, text)


def _append_inline_markdown_runs(paragraph: Paragraph, text: str) -> None:
    token_pattern = re.compile(r"(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`)")
    cursor = 0
    for match in token_pattern.finditer(text):
        if match.start() > cursor:
            paragraph.add_run(_strip_inline_markdown_markers(text[cursor:match.start()]))
        token = match.group(0)
        content = token[1:-1]
        run = paragraph.add_run(content)
        if token.startswith("**") or token.startswith("__"):
            run.bold = True
            run.text = token[2:-2]
        elif token.startswith("*") or token.startswith("_"):
            run.italic = True
            run.text = token[1:-1]
        elif token.startswith("`"):
            run.text = token[1:-1]
        cursor = match.end()
    if cursor < len(text):
        paragraph.add_run(_strip_inline_markdown_markers(text[cursor:]))


def _strip_inline_markdown_markers(value: str) -> str:
    cleaned = value.replace("**", "").replace("__", "")
    cleaned = cleaned.replace("`", "")
    return cleaned
