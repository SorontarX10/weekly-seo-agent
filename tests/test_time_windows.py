from datetime import date, timedelta

from weekly_seo_agent.time_windows import compute_windows


def test_windows_have_expected_lengths() -> None:
    run_date = date.today()
    windows = compute_windows(run_date)

    assert windows["current_28d"].days == 7
    assert windows["previous_28d"].days == 7
    assert windows["yoy_52w"].days == 7

    current = windows["current_28d"]
    previous = windows["previous_28d"]

    expected_current_end = run_date - timedelta(days=run_date.weekday() + 1)
    assert current.end == expected_current_end
    assert current.start == current.end - timedelta(days=6)
    assert current.start.weekday() == 0  # Monday
    assert current.end.weekday() == 6  # Sunday

    assert previous.end == current.start - timedelta(days=1)
    assert previous.start == previous.end - timedelta(days=6)
    assert previous.start.weekday() == 0
    assert previous.end.weekday() == 6


def test_yoy_maintains_weekday_alignment() -> None:
    windows = compute_windows(date.today())

    current = windows["current_28d"]
    yoy = windows["yoy_52w"]

    assert current.start.weekday() == yoy.start.weekday()
    assert current.end.weekday() == yoy.end.weekday()
    assert (current.start - yoy.start).days == 364
    assert (current.end - yoy.end).days == 364
