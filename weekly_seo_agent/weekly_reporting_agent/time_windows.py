from __future__ import annotations

from datetime import date, timedelta

from weekly_seo_agent.weekly_reporting_agent.models import DateWindow


def compute_windows(run_date: date | None = None) -> dict[str, DateWindow]:
    """Build weekly windows for current, previous and YoY (52 weeks ago).

    Current window is the last fully completed Monday-Sunday week before `run_date`.
    Previous window is the preceding Monday-Sunday week.
    YoY window is aligned by 52 weeks (364 days) to keep weekday alignment.
    """
    run_date = run_date or date.today()

    # Monday=0 ... Sunday=6; we always close on the previous Sunday.
    current_end = run_date - timedelta(days=run_date.weekday() + 1)
    current_start = current_end - timedelta(days=6)

    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=6)

    yoy_start = current_start - timedelta(weeks=52)
    yoy_end = current_end - timedelta(weeks=52)

    # Broader context overlay (last 28 complete days ending on current_end).
    context_28d_current_end = current_end
    context_28d_current_start = context_28d_current_end - timedelta(days=27)
    context_28d_previous_end = context_28d_current_start - timedelta(days=1)
    context_28d_previous_start = context_28d_previous_end - timedelta(days=27)
    context_28d_yoy_start = context_28d_current_start - timedelta(weeks=52)
    context_28d_yoy_end = context_28d_current_end - timedelta(weeks=52)

    return {
        "current_28d": DateWindow("Current week (Mon-Sun)", current_start, current_end),
        "previous_28d": DateWindow("Previous week (Mon-Sun)", previous_start, previous_end),
        "yoy_52w": DateWindow("YoY aligned week (52 weeks ago)", yoy_start, yoy_end),
        "current_28d_context": DateWindow(
            "Context last 28 days",
            context_28d_current_start,
            context_28d_current_end,
        ),
        "previous_28d_context": DateWindow(
            "Context previous 28 days",
            context_28d_previous_start,
            context_28d_previous_end,
        ),
        "yoy_28d_context_52w": DateWindow(
            "Context YoY 28 days (52 weeks ago)",
            context_28d_yoy_start,
            context_28d_yoy_end,
        ),
    }
