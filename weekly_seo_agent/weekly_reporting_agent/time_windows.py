from __future__ import annotations

from datetime import date, timedelta
import calendar

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


def compute_monthly_windows(
    run_date: date | None = None,
    target_month: str | None = None,
) -> dict[str, DateWindow]:
    """Build monthly windows for current, previous and YoY (same month previous year).

    - If ``target_month`` is provided (YYYY-MM), it is used as the current month window.
    - Otherwise, current month is the last fully completed month before ``run_date``.
    """
    run_date = run_date or date.today()

    if target_month:
        year_text, month_text = target_month.split("-", 1)
        year = int(year_text)
        month = int(month_text)
        current_start = date(year, month, 1)
    else:
        first_of_current_month = date(run_date.year, run_date.month, 1)
        current_end = first_of_current_month - timedelta(days=1)
        current_start = date(current_end.year, current_end.month, 1)

    current_last_day = calendar.monthrange(current_start.year, current_start.month)[1]
    current_end = date(current_start.year, current_start.month, current_last_day)

    previous_month_end = current_start - timedelta(days=1)
    previous_start = date(previous_month_end.year, previous_month_end.month, 1)
    previous_end = previous_month_end

    yoy_start = date(current_start.year - 1, current_start.month, 1)
    yoy_last_day = calendar.monthrange(yoy_start.year, yoy_start.month)[1]
    yoy_end = date(yoy_start.year, yoy_start.month, yoy_last_day)

    # Keep 28-day context overlays so existing diagnostics can still reuse them.
    context_28d_current_end = current_end
    context_28d_current_start = context_28d_current_end - timedelta(days=27)
    context_28d_previous_end = context_28d_current_start - timedelta(days=1)
    context_28d_previous_start = context_28d_previous_end - timedelta(days=27)
    context_28d_yoy_start = context_28d_current_start - timedelta(weeks=52)
    context_28d_yoy_end = context_28d_current_end - timedelta(weeks=52)

    month_label = current_start.strftime("%Y-%m")
    return {
        "current_28d": DateWindow(f"Current month ({month_label})", current_start, current_end),
        "previous_28d": DateWindow("Previous month", previous_start, previous_end),
        "yoy_52w": DateWindow("YoY same month (previous year)", yoy_start, yoy_end),
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
