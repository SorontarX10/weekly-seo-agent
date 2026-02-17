from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Google Search Console OAuth refresh token"
    )
    parser.add_argument(
        "--client-secret",
        default="",
        help="Path to OAuth client secret JSON (default: GSC_OAUTH_CLIENT_SECRET_PATH from .env)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file used with --write-env (default: .env)",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write generated token into env file under GSC_OAUTH_REFRESH_TOKEN",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser automatically",
    )
    return parser.parse_args()


def _quote_env(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _upsert_env(env_path: Path, key: str, value: str) -> None:
    quoted = _quote_env(value)
    lines: list[str] = []
    found = False

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            updated.append(f"{key}={quoted}")
            found = True
        else:
            updated.append(line)

    if not found:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(f"{key}={quoted}")

    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> None:
    try:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    args = _parse_args()

    client_secret = args.client_secret.strip() or os.getenv(
        "GSC_OAUTH_CLIENT_SECRET_PATH", ""
    ).strip()
    if not client_secret:
        raise SystemExit(
            "Missing OAuth client secret path. Set GSC_OAUTH_CLIENT_SECRET_PATH or pass --client-secret."
        )

    secret_path = Path(client_secret)
    if not secret_path.exists():
        raise SystemExit(f"Client secret file not found: {client_secret}")

    flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
    creds = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
        open_browser=not args.no_browser,
    )

    refresh_token = (creds.refresh_token or "").strip()
    if not refresh_token:
        raise SystemExit(
            "No refresh token received. Revoke app access for this OAuth client and rerun command with consent."
        )

    print("GSC_OAUTH_REFRESH_TOKEN generated successfully.")
    print(f"GSC_OAUTH_REFRESH_TOKEN={refresh_token}")

    if args.write_env:
        env_path = Path(args.env_file)
        _upsert_env(env_path, "GSC_OAUTH_REFRESH_TOKEN", refresh_token)
        print(f"Token written to: {env_path}")


if __name__ == "__main__":
    main()
