#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
from pathlib import Path
import zlib

import requests

from weekly_seo_agent.workflow import build_workflow_app


def _encode_mermaid(mermaid: str) -> str:
    payload = json.dumps(
        {"code": mermaid, "mermaid": {"theme": "default"}},
        separators=(",", ":"),
    ).encode("utf-8")
    compressor = zlib.compressobj(level=9, wbits=zlib.MAX_WBITS)
    compressed = compressor.compress(payload) + compressor.flush()
    return base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")


def _fetch_mermaid_render(
    mermaid: str,
    *,
    render_type: str,
    timeout_sec: int = 25,
    retries: int = 2,
) -> bytes:
    encoded = _encode_mermaid(mermaid)
    url = f"https://mermaid.ink/{render_type}/pako:{encoded}"
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_sec)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # pragma: no cover - network instability path
            last_exc = exc
    raise RuntimeError(f"Mermaid render failed ({render_type}): {last_exc}")


def _overview_mermaid() -> str:
    app = build_workflow_app()
    graph = app.get_graph()
    return graph.draw_mermaid().strip() + "\n"


def _collect_mermaid() -> str:
    return """
flowchart TD
    A[collect_and_analyze] --> B[compute_windows]
    B --> C[GSC fetch totals + rows]
    C --> D[query filtering]
    D --> E[analyze_rows]
    E --> F[ExternalSignalsClient.collect]
    F --> G[collect_additional_context]
    G --> H[ferie + segment diagnostics]
    H --> I[Senuto intelligence]
    I --> J[Allegro Trends intelligence]
    J --> K[GA4 intelligence]
    K --> L[build_markdown_report]

    C -.-> T1[(Google Search Console API)]
    F -.-> T2[(Weather + Google status/blog + news RSS)]
    G -.-> T3[(Google Drive + Docs + Sheets + GDELT + Google Trends + NBP/IMGW)]
    H -.-> T4[(OpenHolidays API)]
    I -.-> T5[(Senuto API)]
    J -.-> T6[(Allegro Trends API)]
    K -.-> T7[(GA4 Data API)]
""".strip() + "\n"


def _llm_generate_mermaid() -> str:
    return """
flowchart TD
    A[llm_generate] --> B[Step 1: extract_key_data_packets]
    B --> C[Packet 1: GSC metrics + movers]
    B --> D[Packet 2: external + seasonality]
    B --> E[Packet 3: market intelligence]
    B --> F[Packet 4: appendix evidence sample]
    C --> G[Step 2: map LLM call]
    D --> G
    E --> G
    F --> G
    G --> H[Step 3: reduce LLM call]
    H --> I[normalize markdown]
    I --> J[deduplicate + date context injection]
    J --> K[llm_commentary_draft]
""".strip() + "\n"


def _llm_validate_mermaid() -> str:
    return """
flowchart TD
    A[llm_validate] --> B[compose candidate report]
    B --> C[Pass 1 LLM: structural/factual issues]
    B --> D[Pass 2 LLM: summary/narrative vs appendix consistency]
    C --> E[Pass 3 LLM gate]
    D --> E
    E --> F{approved?}
    F -->|yes| G[llm_validation_passed=true]
    F -->|no and round < max| H[feedback notes for rewrite]
    F -->|no and round >= max| I[force pass with validator notes]
""".strip() + "\n"


def _llm_finalize_mermaid() -> str:
    return """
flowchart TD
    A[llm_finalize] --> B[compose final report]
    B --> C[attach validator notes if any]
    C --> D[return llm_commentary + final_report]
""".strip() + "\n"


def _system_detailed_mermaid() -> str:
    return """
flowchart TD
    A[Start run_weekly_workflow] --> B[collect_and_analyze]
    B --> C[llm_generate]
    C --> D[llm_validate]
    D -->|needs rewrite| C
    D -->|approved / max rounds| E[llm_finalize]
    E --> F[END]
""".strip() + "\n"


def _write_diagram_bundle(out_dir: Path, name: str, mermaid: str) -> tuple[Path, Path | None, Path | None]:
    mmd_path = out_dir / f"{name}.mmd"
    mmd_path.write_text(mermaid, encoding="utf-8")

    svg_path = out_dir / f"{name}.svg"
    png_path = out_dir / f"{name}.png"
    got_svg = False
    got_png = False
    try:
        svg_path.write_bytes(_fetch_mermaid_render(mermaid, render_type="svg"))
        got_svg = True
    except Exception:
        pass
    try:
        png_path.write_bytes(_fetch_mermaid_render(mermaid, render_type="img"))
        got_png = True
    except Exception:
        pass
    return mmd_path, (svg_path if got_svg else None), (png_path if got_png else None)


def _write_index(out_dir: Path, rows: list[tuple[str, Path, Path | None, Path | None]]) -> Path:
    index_path = out_dir / "langgraph_diagrams_index.md"
    lines = [
        "# LangGraph Diagrams",
        "",
        "Recommended: open `.svg` files for readable zoomable diagrams.",
        "",
        "| Diagram | Mermaid | SVG | PNG |",
        "|---|---|---|---|",
    ]
    for label, mmd, svg, png in rows:
        lines.append(
            f"| {label} | [{mmd.name}]({mmd.name}) | "
            f"{f'[{svg.name}]({svg.name})' if svg else '-'} | "
            f"{f'[{png.name}]({png.name})' if png else '-'} |"
        )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    diagrams: list[tuple[str, str, str]] = [
        ("Workflow overview", "langgraph_workflow_overview", _overview_mermaid()),
        ("System flow detailed", "langgraph_workflow_detailed", _system_detailed_mermaid()),
        ("Collect and analyze", "langgraph_collect_and_analyze", _collect_mermaid()),
        ("LLM generate pipeline", "langgraph_llm_generate", _llm_generate_mermaid()),
        ("LLM validate pipeline", "langgraph_llm_validate", _llm_validate_mermaid()),
        ("LLM finalize pipeline", "langgraph_llm_finalize", _llm_finalize_mermaid()),
    ]

    written: list[tuple[str, Path, Path | None, Path | None]] = []
    for label, stem, mermaid in diagrams:
        mmd, svg, png = _write_diagram_bundle(out_dir, stem, mermaid)
        written.append((label, mmd, svg, png))
        print(f"{label}: {mmd}")
        print(f"  SVG: {svg if svg else 'not generated'}")
        print(f"  PNG: {png if png else 'not generated'}")

    index_path = _write_index(out_dir, written)
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
