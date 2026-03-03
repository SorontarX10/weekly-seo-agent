from __future__ import annotations

import os

from .logging_utils import log_event
from .service import DocumentService


def main() -> None:
    db_path = os.getenv("MANAGER_DOCUMENT_AGENT_DB_PATH", "outputs/manager_document_agent.db")
    older_than_days = int(os.getenv("MANAGER_DOCUMENT_AGENT_ARCHIVE_DAYS", "30"))

    service = DocumentService(db_path)
    try:
        archived_count = service.run_archive_job(older_than_days=older_than_days)
    finally:
        service.close()

    log_event(
        "nightly_archive_run",
        older_than_days=older_than_days,
        archived_count=archived_count,
    )
    print(f"archived_count={archived_count}")


if __name__ == "__main__":
    main()
