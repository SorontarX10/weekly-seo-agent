from __future__ import annotations

import threading

from .service import DocumentService


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._request_failures = 0
        self._request_latency_ms_total = 0.0

    def observe_request(self, *, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            if status_code >= 400:
                self._request_failures += 1
            self._request_latency_ms_total += latency_ms

    def snapshot(self, service: DocumentService) -> dict:
        with self._lock:
            request_count = self._request_count
            request_failures = self._request_failures
            request_avg_latency = (
                self._request_latency_ms_total / request_count if request_count else 0.0
            )
            request_failure_rate = (
                request_failures / request_count if request_count else 0.0
            )

        job_metrics = service.job_metrics()
        return {
            "requests": {
                "total": request_count,
                "failures": request_failures,
                "failure_rate": request_failure_rate,
                "avg_latency_ms": request_avg_latency,
            },
            "jobs": job_metrics,
        }
