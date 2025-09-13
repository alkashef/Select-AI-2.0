"""MCP diagnostics: list tools, prompts, and resources.

Usage:
  python tests\test_mcp_diag.py
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
    env_path = Path(__file__).resolve().parents[1] / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _get_db_uri() -> Optional[str]:
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
        return f"teradata://{url_quote(user, safe='')}:{url_quote(pwd, safe='')}@{host}:{port}/{url_quote(db, safe='')}"
    return None


async def main() -> int:
    _load_env()
    uri = _get_db_uri()
    if not uri:
        print("DATABASE_URI missing and TD_* incomplete. Configure config/.env.")
        return 2

    server = StdioServerParameters(command="teradata-mcp-server", args=[], env={"DATABASE_URI": uri})
    print("Launching teradata-mcp-server via stdio...")
    try:
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = (await session.list_tools()).tools
                print("\n--- Tools ---")
                for t in tools:
                    name = getattr(t, "name", "") or ""
                    schema = getattr(t, "inputSchema", None)
                    props = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
                    print(f"- {name}: args={props}")

                prompts = (await session.list_prompts()).prompts
                print("\n--- Prompts ---")
                if prompts:
                    for p in prompts:
                        print(f"- {getattr(p, 'name', '')}: {getattr(p, 'description', '')}")
                else:
                    print("(none)")

                resources = (await session.list_resources()).resources
                print("\n--- Resources ---")
                if resources:
                    for r in resources:
                        print(f"- {getattr(r, 'name', '')}: {getattr(r, 'uri', '')}")
                else:
                    print("(none)")

                return 0
    except Exception as e:
        print(f"\nMCP diag failed: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
