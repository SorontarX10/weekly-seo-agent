from __future__ import annotations

import os

import uvicorn

from .api import create_app


def main() -> None:
    host = os.getenv("MANAGER_DOCUMENT_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("MANAGER_DOCUMENT_AGENT_PORT", "8081"))
    db_path = os.getenv("MANAGER_DOCUMENT_AGENT_DB_PATH", "outputs/manager_document_agent.db")
    app = create_app(db_path=db_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
