from __future__ import annotations

from weekly_seo_agent.models import MetricRow


def _extract_query_from_key(row_key: str) -> str:
    # Keys for multi-dimension reports are joined as "query | country | device".
    return row_key.split("|", 1)[0].strip().lower()


def filter_irrelevant_query_rows(
    rows: list[MetricRow],
    exclude_patterns: tuple[str, ...],
) -> tuple[list[MetricRow], int]:
    if not rows:
        return rows, 0

    cleaned_patterns = tuple(pattern.strip().lower() for pattern in exclude_patterns if pattern.strip())
    if not cleaned_patterns:
        return rows, 0

    kept: list[MetricRow] = []
    dropped = 0

    for row in rows:
        query = _extract_query_from_key(row.key)
        if query and any(pattern in query for pattern in cleaned_patterns):
            dropped += 1
            continue
        kept.append(row)

    return kept, dropped
