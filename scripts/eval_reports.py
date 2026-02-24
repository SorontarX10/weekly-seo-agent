#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

from docx import Document

from weekly_seo_agent.weekly_reporting_agent.evaluation import evaluate_report_text


def _extract_text(path: Path) -> str:
    doc = Document(path)
    return "\n".join(par.text for par in doc.paragraphs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated weekly report DOCX files.")
    parser.add_argument("--dir", default="SEO Weekly Reports", help="Directory with generated .docx files.")
    parser.add_argument("--min-score", type=int, default=75, help="Minimum passing score.")
    parser.add_argument(
        "--prefix",
        default=f"{date.today().strftime('%Y_%m_%d')}_",
        help="Evaluate only files starting with this prefix (default: today's prefix).",
    )
    args = parser.parse_args()

    report_dir = Path(args.dir)
    if not report_dir.exists():
        print(f"Directory not found: {report_dir}")
        return 1

    files = sorted(report_dir.glob("*.docx"))
    if args.prefix:
        files = [path for path in files if path.name.startswith(args.prefix)]
    if not files:
        print("No DOCX files found to evaluate.")
        return 1

    failed = False
    for path in files:
        text = _extract_text(path)
        result = evaluate_report_text(text)
        score = int(result.get("score", 0) or 0)
        passed = bool(result.get("passed", False)) and score >= int(args.min_score)
        mark = "PASS" if passed else "FAIL"
        print(f"{mark} | {path.name} | score={score}")
        if not passed:
            failed = True
            for issue in result.get("issues", []):
                print(f"  - {issue}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
