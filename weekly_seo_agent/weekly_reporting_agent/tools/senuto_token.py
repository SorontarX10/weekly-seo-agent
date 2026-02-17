from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from weekly_seo_agent.weekly_reporting_agent.clients.senuto_client import SenutoClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Senuto bearer token")
    parser.add_argument(
        "--email",
        default="",
        help="Senuto account email (default: SENUTO_EMAIL from .env)",
    )
    parser.add_argument(
        "--password",
        default="",
        help="Senuto account password (default: SENUTO_PASSWORD from .env)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Senuto API base URL (default: SENUTO_BASE_URL from .env)",
    )
    parser.add_argument(
        "--token-endpoint",
        default="",
        help="Token endpoint path or URL (default: SENUTO_TOKEN_ENDPOINT from .env)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file used with --write-env (default: .env)",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write generated token into env file under SENUTO_TOKEN",
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

    email = args.email.strip() or os.getenv("SENUTO_EMAIL", "").strip()
    password = args.password.strip() or os.getenv("SENUTO_PASSWORD", "").strip()
    base_url = args.base_url.strip() or os.getenv("SENUTO_BASE_URL", "https://api.senuto.com").strip()
    token_endpoint = (
        args.token_endpoint.strip()
        or os.getenv("SENUTO_TOKEN_ENDPOINT", "/api/users/token").strip()
    )

    if not email or not password:
        raise SystemExit(
            "Missing credentials. Set SENUTO_EMAIL + SENUTO_PASSWORD or pass --email/--password."
        )

    client = SenutoClient(
        token="",
        email=email,
        password=password,
        token_endpoint=token_endpoint,
        base_url=base_url,
        domain="allegro.pl",
        visibility_endpoint="/api/visibility_analysis/reports/domain_positions/getPositionsSumsDistributionChartData",
    )

    try:
        token = client.fetch_token(force_refresh=True)
    except Exception as exc:
        raise SystemExit(f"Failed to generate SENUTO_TOKEN: {exc}") from exc

    print("SENUTO_TOKEN generated successfully.")
    print(f"SENUTO_TOKEN={token}")

    if args.write_env:
        env_path = Path(args.env_file)
        _upsert_env(env_path, "SENUTO_TOKEN", token)
        print(f"Token written to: {env_path}")


if __name__ == "__main__":
    main()
