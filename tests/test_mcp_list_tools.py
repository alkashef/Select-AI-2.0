"""Minimal MCP stdio smoke test for Teradata MCP Server.

Behavior:
- Loads env from `config/.env`.
- Uses `DATABASE_URI` if set; otherwise builds it from `TD_*` and URL-encodes credentials.
- Starts `teradata-mcp-server` via stdio, initializes a session, lists tools, and exits.

Usage (Windows cmd):
  python tests\test_mcp_list_tools.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote as url_quote

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _build_database_uri() -> Optional[str]:
    raw = os.getenv("DATABASE_URI")
    if raw:
        raw = raw.strip()
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            raw = raw[1:-1]
        return raw
    user = os.getenv("TD_USER")
    pwd = os.getenv("TD_PASSWORD")
    host = os.getenv("TD_HOST")
    db = os.getenv("TD_NAME")
    port = os.getenv("TD_PORT", "1025")
    if all([user, pwd, host, db]):
        user_enc = url_quote(user or "", safe="")
        pwd_enc = url_quote(pwd or "", safe="")
        db_enc = url_quote(db or "", safe="")
        return f"teradata://{user_enc}:{pwd_enc}@{host}:{port}/{db_enc}"
    return None


async def main() -> int:
    _load_env()
    db_uri = _build_database_uri()
    if not db_uri:
        print("DATABASE_URI is not set and TD_* variables are incomplete. Update config/.env.")
        return 2

    server_params = StdioServerParameters(
        command="teradata-mcp-server",
        args=[],
        env={"DATABASE_URI": db_uri},
    )

    print("Launching teradata-mcp-server via stdio...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_resp = await session.list_tools()
                tools = tools_resp.tools
                if not tools:
                    print("No tools exposed by MCP server.")
                    return 1

                print("\n--- Available Tools ---")
                for t in tools:
                    name = getattr(t, "name", "") or ""
                    schema = getattr(t, "inputSchema", None)
                    props = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
                    print(f"- {name}: args={props}")

                return 0
    except Exception as e:
        print(f"\nMCP stdio session failed: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
