from __future__ import annotations

import time

from weekly_seo_agent.weekly_reporting_agent.workflow import _await_daemon_task, _start_daemon_task


def test_daemon_task_returns_value() -> None:
    handle = _start_daemon_task(lambda: ("ok", 123))
    status, value = _await_daemon_task(
        handle,
        timeout_sec=1,
        task_name="value-task",
    )

    assert status == "ok"
    assert value == ("ok", 123)


def test_daemon_task_timeout_is_reported() -> None:
    handle = _start_daemon_task(lambda: time.sleep(2))
    status, value = _await_daemon_task(
        handle,
        timeout_sec=1,
        task_name="slow-task",
    )

    assert status == "timeout"
    assert isinstance(value, TimeoutError)


def test_daemon_task_exception_is_reported() -> None:
    def _raise() -> None:
        raise RuntimeError("boom")

    handle = _start_daemon_task(_raise)
    status, value = _await_daemon_task(
        handle,
        timeout_sec=1,
        task_name="error-task",
    )

    assert status == "error"
    assert isinstance(value, RuntimeError)
    assert "boom" in str(value)
