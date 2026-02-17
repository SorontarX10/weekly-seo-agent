import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Dict, Set, Any
from urllib.parse import urlparse

from weekly_seo_agent.clients.continuity_client import ContinuityClient
from weekly_seo_agent.clients.external_signals import ExternalSignalsClient
from weekly_seo_agent.clients.gsc_client import GSCClient
from weekly_seo_agent.clients.senuto_client import SenutoClient
from weekly_seo_agent.config import AgentConfig
from weekly_seo_agent.llm import build_gaia_llm
from weekly_seo_agent.models import DateWindow, MetricSummary

SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/18Qxrh3sTiMQ-5wCQqrJwLWh8bdYujWzvb9ItXqAg214/edit?usp=sharing"
SOURCE_SHEET = "Final Version"
TARGET_SHEET = "LLM Version"
ROADMAP_SHEET = "Roadmap 2026"

CLIENT_SECRET = "client_secret_2_577772038295-f6vgq53l3f84r5atsog5fjvspqtouo4d.apps.googleusercontent.com.json"
TOKEN_PATH = ".google_drive_token.json"


@dataclass
class TaskRow:
    task_id: str
    task: str
    owners: Set[str] = field(default_factory=set)


def _normalize_owner(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[\/,&;]+", value)
    cleaned = []
    for p in parts:
        item = p.strip()
        if not item:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", item):
            continue
        cleaned.append(item)
    return cleaned


def _extract_json(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if not match:
        raise RuntimeError("LLM response did not contain JSON.")
    return json.loads(match.group(0))


def _build_prompt(tasks: List[TaskRow]) -> str:
    items = [
        {
            "id": row.task_id,
            "task": row.task,
            "owners": "/".join(sorted(row.owners)),
        }
        for row in tasks
    ]
    return json.dumps({"tasks": items}, ensure_ascii=False)


def _classify_page(url: str) -> str:
    if not url:
        return "other"
    parsed = urlparse(url)
    path = parsed.path or "/"
    query = parsed.query or ""
    if path in ("", "/"):
        return "homepage"
    if "/oferta/" in path or "/produkt/" in path or "/offer/" in path:
        return "product"
    if "/kategoria/" in path or "/category/" in path or "/katalog/" in path:
        return "listing"
    if (
        "string=" in query
        or "q=" in query
        or "phrase=" in query
        or "/listing" in path
        or "/list" in path
        or "/szukaj" in path
        or "/search" in path
        or "/wyszukaj" in path
    ):
        return "listing"
    return "other"


def _gsc_page_shares(client: GSCClient, window: DateWindow) -> Dict[str, float]:
    rows = client.fetch_rows(window=window, dimensions=["page"])
    total_clicks = sum(r.clicks for r in rows)
    by_type = {"homepage": 0.0, "listing": 0.0, "product": 0.0, "other": 0.0}
    for row in rows:
        page_type = _classify_page(row.key)
        by_type[page_type] += row.clicks
    if total_clicks <= 0:
        return {k: 0.0 for k in by_type}
    return {k: (v / total_clicks) for k, v in by_type.items()}


def _weighted_shares(shares_all: Dict[str, float], shares_pl: Dict[str, float], shares_csh: Dict[str, float]) -> Dict[str, float]:
    weights = {"pl": 0.7, "all": 0.2, "csh": 0.1}
    combined = {}
    for key in ("homepage", "listing", "product", "other"):
        combined[key] = (
            shares_pl.get(key, 0.0) * weights["pl"]
            + shares_all.get(key, 0.0) * weights["all"]
            + shares_csh.get(key, 0.0) * weights["csh"]
        )
    return combined


def _senuto_summary(config: AgentConfig) -> str:
    try:
        client = SenutoClient(
            token=config.senuto_token,
            email=config.senuto_email,
            password=config.senuto_password,
            token_endpoint=config.senuto_token_endpoint,
            base_url=config.senuto_base_url,
            domain=config.senuto_domain,
            visibility_endpoint=config.senuto_visibility_endpoint,
            fetch_mode=config.senuto_fetch_mode,
            country_id=config.senuto_country_id,
            date_interval=config.senuto_date_interval,
            visibility_metric=config.senuto_visibility_metric,
        )
        end = date.today()
        start = end - timedelta(days=90)
        points = client.fetch_visibility(start_date=start, end_date=end)
        if not points:
            return "Senuto visibility: no data."
        points_sorted = sorted(points, key=lambda p: p.day)
        first = points_sorted[0].visibility
        last = points_sorted[-1].visibility
        delta = last - first
        pct = (delta / first * 100.0) if first else 0.0
        trend = "up" if delta > 0 else "down" if delta < 0 else "flat"
        return f"Senuto visibility 90d trend: {trend} ({delta:+.2f}, {pct:+.1f}%)."
    except Exception as exc:
        return f"Senuto visibility: error ({exc})."


def _external_signals_summary(config: AgentConfig) -> str:
    try:
        client = ExternalSignalsClient(
            latitude=config.weather_latitude,
            longitude=config.weather_longitude,
            weather_label=config.weather_label,
            market_country_code=config.report_country_code,
            status_endpoint=config.google_status_endpoint,
            blog_rss_url=config.google_blog_rss,
            holidays_country_code=config.holidays_country_code,
            holidays_api_base_url=config.holidays_api_base_url,
            holidays_language_code=config.holidays_language_code,
            news_scraping_enabled=config.news_scraping_enabled,
            news_rss_urls_pl=config.news_rss_urls_pl,
            news_rss_urls_global=config.news_rss_urls_global,
            news_html_urls_pl=config.news_html_urls_pl,
            news_html_urls_global=config.news_html_urls_global,
            news_keywords=config.news_keywords,
            news_max_signals=min(8, config.news_max_signals),
        )
        end = date.today()
        current = DateWindow(name="current", start=end - timedelta(days=27), end=end)
        previous = DateWindow(name="previous", start=end - timedelta(days=55), end=end - timedelta(days=28))
        signals, _ = client.collect(current_window=current, previous_window=previous)
        if not signals:
            return "External signals: none."
        top = signals[:6]
        titles = "; ".join(f"{s.source}: {s.title}" for s in top)
        return f"External signals: {titles}"
    except Exception as exc:
        return f"External signals: error ({exc})."


def _llm_cluster(
    tasks: List[TaskRow],
    traffic_context: str,
    senuto_context: str,
    external_context: str,
    target_min: int,
    target_max: int,
) -> List[Dict[str, Any]]:
    config = AgentConfig.from_env()
    llm = build_gaia_llm(config)

    system = (
        "You are a senior SEO program manager. Your job is to deduplicate tasks by meaning, "
        "not by keywords. Be conservative: only merge tasks that are effectively the same work item. "
        "If tasks differ by market, product area, page type, or scope, keep them separate. "
        "Example: canonicals for listings vs canonicals for Allegro Lokalnie are different and must not be merged. "
        "Avoid broad thematic bundling. Prefer many smaller clusters over a few big ones. "
        "Return ONLY valid JSON."
    )
    payload = _build_prompt(tasks)
    attempt_notes = ""
    for attempt in range(3):
        user = (
            "Group tasks into unique initiatives with minimal merging. "
            "Only merge when the tasks are near-identical in scope and intent. "
            "If unsure, DO NOT merge. "
            "Hard rule: do NOT merge if tasks reference different markets (PL, CZ, SK, HU, IE), "
            "different products/brands (e.g., Allegro Lokalnie), or different page types "
            "(listing vs product vs homepage). "
            f"Target number of clusters: {target_min}-{target_max}. "
            "IMPORTANT: Do NOT lose any content. When merging, preserve full original task texts. "
            "You will output only IDs; the system will preserve full text by itself. "
            "Assign a target quarter for 2026 (Q1/Q2/Q3/Q4). "
        "Scoring must be grounded in traffic drivers. Treat search pages as listing pages. "
        "Use the provided traffic shares and prioritize listing pages, product pages, and homepage. "
        "Also consider Senuto trend and external signals "
        "as context (not as the main signal). "
            "Output JSON with shape: {\n"
            "  \"clusters\": [\n"
            "    {\n"
            "      \"task_ids\": [\"T1\", ...],\n"
            "      \"quarter\": \"Q1|Q2|Q3|Q4\",\n"
            "      \"rationale\": \"short reason\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Traffic context:\n"
            f"{traffic_context}\n"
            "Senuto context:\n"
            f"{senuto_context}\n"
            "External context:\n"
            f"{external_context}\n"
            f"{attempt_notes}"
            "Input:\n"
        )
        response = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user + payload},
            ]
        )
        content = response.content if hasattr(response, "content") else str(response)
        data = _extract_json(content)
        clusters = data.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise RuntimeError("LLM returned empty clusters.")
        if target_min <= len(clusters) <= target_max:
            return clusters
        if len(clusters) > target_max:
            attempt_notes = (
                "Previous attempt created too many clusters; merge more obvious duplicates but keep scope boundaries. "
            )
        else:
            attempt_notes = (
                "Previous attempt created too few clusters; split apart different scopes/markets/page types. "
            )
    return clusters


def _llm_score_tasks(
    tasks: List[TaskRow],
    traffic_context: str,
    senuto_context: str,
    external_context: str,
) -> Dict[str, Dict[str, Any]]:
    config = AgentConfig.from_env()
    llm = build_gaia_llm(config)

    system = (
        "You are a senior SEO program manager. Score each task by expected GMV impact "
        "based on traffic drivers (listing, product, homepage). Assign the single best driver label: "
        "listing, product, homepage, infra, content, international, other. "
        "Return ONLY valid JSON."
    )
    scores: Dict[str, Dict[str, Any]] = {}
    batch_size = 20
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        payload = json.dumps(
            {"tasks": [{"id": t.task_id, "task": t.task} for t in chunk]},
            ensure_ascii=False,
        )
        user = (
            "Score each task with:\n"
            "- driver: listing|product|homepage|infra|content|international|other\n"
            "- impact: 0-100 (higher means more GMV impact)\n"
            "- rationale: short reason\n"
            "Use traffic context as primary signal, Senuto and external as supporting context.\n"
            "Output JSON: {\"scores\":[{\"id\":\"T1\",\"driver\":\"listing\",\"impact\":75,\"rationale\":\"...\"}, ...]}\n"
            "Traffic context:\n"
            f"{traffic_context}\n"
            "Senuto context:\n"
            f"{senuto_context}\n"
            "External context:\n"
            f"{external_context}\n"
            "Input:\n"
            f"{payload}\n"
        )
        response = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        content = response.content if hasattr(response, "content") else str(response)
        data = _extract_json(content)
        rows = data.get("scores", [])
        if not isinstance(rows, list):
            raise RuntimeError("LLM scoring returned invalid JSON.")
        for row in rows:
            task_id = str(row.get("id", "")).strip()
            if not task_id:
                continue
            driver = str(row.get("driver", "other")).strip().lower()
            impact = int(row.get("impact", 0))
            scores[task_id] = {
                "driver": driver if driver else "other",
                "impact": max(0, min(100, impact)),
                "rationale": str(row.get("rationale", "")).strip(),
            }
    return scores


def _ensure_sheet(sheets, spreadsheet_id: str, title: str) -> int:
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        if props.get("title") == title:
            return int(props.get("sheetId"))

    resp = sheets.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": title}}}]},
    ).execute()
    sheet_id = resp["replies"][0]["addSheet"]["properties"]["sheetId"]
    return int(sheet_id)


def _apply_roadmap_formatting(sheets, spreadsheet_id: str, sheet_id: int, rows: int) -> None:
    # Add conditional format: if cell == "###" then fill color
    requests = []
    for col_idx, color in zip([2, 3, 4, 5], [
        {"red": 0.80, "green": 0.93, "blue": 0.80},
        {"red": 0.82, "green": 0.89, "blue": 0.98},
        {"red": 0.98, "green": 0.90, "blue": 0.80},
        {"red": 0.96, "green": 0.85, "blue": 0.93},
    ]):
        requests.append(
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [
                            {
                                "sheetId": sheet_id,
                                "startRowIndex": 1,
                                "endRowIndex": rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1,
                            }
                        ],
                        "booleanRule": {
                            "condition": {
                                "type": "TEXT_EQ",
                                "values": [{"userEnteredValue": "###"}],
                            },
                            "format": {
                                "backgroundColor": color,
                                "textFormat": {"bold": True},
                            },
                        },
                    },
                    "index": 0,
                }
            }
        )

    sheets.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests},
    ).execute()


def main():
    spreadsheet_id = re.search(r"/d/([a-zA-Z0-9-_]+)", SPREADSHEET_URL).group(1)
    client = ContinuityClient(
        client_secret_path=CLIENT_SECRET,
        token_path=TOKEN_PATH,
        reports_folder_name="SEO Weekly Reports",
        reports_folder_id="",
    )
    sheets = client._sheets_service()

    config = AgentConfig.from_env()
    today = date.today()
    window = DateWindow(name="last_28d", start=today - timedelta(days=27), end=today)

    gsc_all = GSCClient(
        site_url=config.gsc_site_url,
        credentials_path=config.gsc_credentials_path,
        oauth_client_secret_path=config.gsc_oauth_client_secret_path,
        oauth_refresh_token=config.gsc_oauth_refresh_token,
        oauth_token_uri=config.gsc_oauth_token_uri,
        country_filter="",
        row_limit=config.gsc_row_limit,
    )
    gsc_pl = GSCClient(
        site_url=config.gsc_site_url,
        credentials_path=config.gsc_credentials_path,
        oauth_client_secret_path=config.gsc_oauth_client_secret_path,
        oauth_refresh_token=config.gsc_oauth_refresh_token,
        oauth_token_uri=config.gsc_oauth_token_uri,
        country_filter="pl",
        row_limit=config.gsc_row_limit,
    )
    gsc_cz = GSCClient(
        site_url=config.gsc_site_url,
        credentials_path=config.gsc_credentials_path,
        oauth_client_secret_path=config.gsc_oauth_client_secret_path,
        oauth_refresh_token=config.gsc_oauth_refresh_token,
        oauth_token_uri=config.gsc_oauth_token_uri,
        country_filter="cz",
        row_limit=config.gsc_row_limit,
    )
    gsc_sk = GSCClient(
        site_url=config.gsc_site_url,
        credentials_path=config.gsc_credentials_path,
        oauth_client_secret_path=config.gsc_oauth_client_secret_path,
        oauth_refresh_token=config.gsc_oauth_refresh_token,
        oauth_token_uri=config.gsc_oauth_token_uri,
        country_filter="sk",
        row_limit=config.gsc_row_limit,
    )
    gsc_hu = GSCClient(
        site_url=config.gsc_site_url,
        credentials_path=config.gsc_credentials_path,
        oauth_client_secret_path=config.gsc_oauth_client_secret_path,
        oauth_refresh_token=config.gsc_oauth_refresh_token,
        oauth_token_uri=config.gsc_oauth_token_uri,
        country_filter="hu",
        row_limit=config.gsc_row_limit,
    )

    shares_all = _gsc_page_shares(gsc_all, window)
    shares_pl = _gsc_page_shares(gsc_pl, window)
    shares_cz = _gsc_page_shares(gsc_cz, window)
    shares_sk = _gsc_page_shares(gsc_sk, window)
    shares_hu = _gsc_page_shares(gsc_hu, window)
    shares_csh = {
        "homepage": (shares_cz.get("homepage", 0.0) + shares_sk.get("homepage", 0.0) + shares_hu.get("homepage", 0.0)) / 3.0,
        "listing": (shares_cz.get("listing", 0.0) + shares_sk.get("listing", 0.0) + shares_hu.get("listing", 0.0)) / 3.0,
        "product": (shares_cz.get("product", 0.0) + shares_sk.get("product", 0.0) + shares_hu.get("product", 0.0)) / 3.0,
        "other": (shares_cz.get("other", 0.0) + shares_sk.get("other", 0.0) + shares_hu.get("other", 0.0)) / 3.0,
    }
    weighted = _weighted_shares(shares_all, shares_pl, shares_csh)

    traffic_context = (
        f"Last 28d GSC click shares by page type (ALL): {shares_all}. "
        f"PL: {shares_pl}. CZ/SK/HU avg: {shares_csh}. "
        f"Weighted (PL 70%, ALL 20%, CZ/SK/HU 10%): {weighted}. "
        "Treat listing, product, homepage as primary traffic drivers."
    )
    senuto_context = _senuto_summary(config)
    external_context = _external_signals_summary(config)

    resp = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{SOURCE_SHEET}'!A1:Z1000",
    ).execute()
    values = resp.get("values", [])
    if not values:
        raise RuntimeError(f"No data in {SOURCE_SHEET} sheet.")

    headers = values[0]
    header_map = {h.strip().lower(): idx for idx, h in enumerate(headers) if h}
    task_idx = header_map.get("task", 0)
    owner_idx = header_map.get("owner", 1)
    wannabe_idx = header_map.get("wannabe owners")
    rows_raw = values[1:]

    task_rows: List[TaskRow] = []
    for i, row in enumerate(rows_raw, start=1):
        task = row[task_idx].strip() if len(row) > task_idx else ""
        if not task:
            continue
        owner = row[owner_idx].strip() if len(row) > owner_idx else ""
        wannabe = row[wannabe_idx].strip() if wannabe_idx is not None and len(row) > wannabe_idx else ""
        owners = set(_normalize_owner(owner)) | set(_normalize_owner(wannabe))
        task_rows.append(TaskRow(task_id=f"T{i}", task=task, owners=owners))

    target_min = max(70, int(len(task_rows) * 0.7))
    target_max = max(85, int(len(task_rows) * 0.85))
    clusters = _llm_cluster(
        task_rows,
        traffic_context,
        senuto_context,
        external_context,
        target_min=target_min,
        target_max=target_max,
    )

    # Build merged rows
    id_to_task = {row.task_id: row for row in task_rows}
    merged_task_rows: List[TaskRow] = []
    merged_rows = []
    for cluster in clusters:
        ids = cluster.get("task_ids", [])
        quarter = str(cluster.get("quarter", "Q4")).strip().upper()
        if quarter not in {"Q1", "Q2", "Q3", "Q4"}:
            quarter = "Q4"

        owners = set()
        task_texts = []
        for task_id in ids:
            row = id_to_task.get(task_id)
            if row:
                owners |= row.owners
                task_texts.append(row.task)
        if not task_texts:
            continue
        merged_task = " / ".join(task_texts)
        merged_task_id = f"M{len(merged_task_rows) + 1}"
        merged_task_rows.append(TaskRow(task_id=merged_task_id, task=merged_task, owners=owners))
        merged_rows.append({
            "task": merged_task,
            "owner": "/".join(sorted(owners)),
            "quarter": quarter,
        })

    scores = _llm_score_tasks(merged_task_rows, traffic_context, senuto_context, external_context)
    for idx, row in enumerate(merged_rows):
        task_row = merged_task_rows[idx]
        score = scores.get(task_row.task_id, {})
        row["driver"] = score.get("driver", "other")
        row["impact"] = score.get("impact", 0)

    merged_rows.sort(key=lambda r: (-r["impact"], r["task"].lower()))

    new_headers = ["Task", "Owner", "Driver", "Impact (GMV)", "Quarter (2026)"]
    new_values = [new_headers]
    for row in merged_rows:
        new_values.append([
            row["task"],
            row["owner"],
            row["driver"],
            str(row["impact"]),
            row["quarter"],
        ])

    sheets.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"'{TARGET_SHEET}'!A1:Z1000",
    ).execute()
    sheets.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{TARGET_SHEET}'!A1",
        valueInputOption="RAW",
        body={"values": new_values},
    ).execute()

    roadmap_sheet_id = _ensure_sheet(sheets, spreadsheet_id, ROADMAP_SHEET)

    roadmap = [["Task", "Owner", "Driver", "Q1", "Q2", "Q3", "Q4"]]
    for row in merged_rows:
        q = row["quarter"]
        marker = "###"
        roadmap.append([
            row["task"],
            row["owner"],
            row["driver"],
            marker if q == "Q1" else "",
            marker if q == "Q2" else "",
            marker if q == "Q3" else "",
            marker if q == "Q4" else "",
        ])

    sheets.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"'{ROADMAP_SHEET}'!A1:Z1000",
    ).execute()
    sheets.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{ROADMAP_SHEET}'!A1",
        valueInputOption="RAW",
        body={"values": roadmap},
    ).execute()

    _apply_roadmap_formatting(sheets, spreadsheet_id, roadmap_sheet_id, len(roadmap))

    print(f"Updated {TARGET_SHEET} with {len(merged_rows)} rows. Roadmap updated in {ROADMAP_SHEET}.")


if __name__ == "__main__":
    main()
