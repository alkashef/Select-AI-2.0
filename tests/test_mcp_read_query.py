"""Advanced MCP example: run a simple SQL via base_readQuery.

Usage (Windows cmd):
        python tests\test_mcp_read_query.py --sql "SELECT CURRENT_DATE" --timeout 120

If --sql is omitted, uses TD_TEST_QUERY from env, else SELECT CURRENT_DATE;
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
from pathlib import Path
from typing import Optional, List
from urllib.parse import quote as url_quote, urlparse

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
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
    parser = argparse.ArgumentParser(description="Call base_readQuery via MCP stdio")
    parser.add_argument("--sql", type=str, default=None, help="SQL to run; ensure it's safe and small")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds for tool call")
    parser.add_argument("--profile", type=str, default=None, help="Optional teradata-mcp-server profile to use")
    parser.add_argument("--preflight-timeout", type=float, default=60.0, help="Timeout seconds for preflight tool call")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight DB version call")
    args = parser.parse_args()

    _load_env()
    uri = _get_db_uri()
    if not uri:
        print("DATABASE_URI missing and TD_* incomplete. Configure config/.env.")
        return 2

    sql = (args.sql or os.getenv("TD_TEST_QUERY") or "SELECT CURRENT_DATE").strip()
    # Many Teradata drivers don't require a trailing semicolon; strip if provided.
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()

    # Parse host/port for quick connectivity diagnostics
    parsed = urlparse(uri)
    host = parsed.hostname or ""
    port = parsed.port or 1025
    if host:
        print(f"Target DB: host={host} port={port} (from DATABASE_URI)")
        try:
            with socket.create_connection((host, port), timeout=3):
                pass
        except Exception as net_err:
            print(f"Warning: TCP connect to {host}:{port} failed quickly: {net_err!r}")
            print("- If this is expected due to firewall/VPN, the DB call may still hang until timeout.")

    server_args: List[str] = []
    if args.profile:
        server_args.extend(["--profile", args.profile])
    server = StdioServerParameters(command="teradata-mcp-server", args=server_args, env={"DATABASE_URI": uri})
    print("Launching teradata-mcp-server via stdio...")
    try:
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                target = next((t for t in tools if t.name == "base_readQuery"), None)
                if not target:
                    print("Tool 'base_readQuery' not found.")
                    return 1

                # Optional preflight: quick DB connectivity check
                preflight_tool = next((t for t in tools if t.name == "dba_databaseVersion"), None)
                if preflight_tool and not args.skip_preflight:
                    try:
                        await asyncio.wait_for(session.call_tool("dba_databaseVersion", {}), timeout=args.preflight_timeout)
                    except asyncio.TimeoutError:
                        print(f"\nPreflight 'dba_databaseVersion' timed out ({args.preflight_timeout:.0f}s). Likely DB connectivity/latency issue.")
                        print("- Verify DATABASE_URI host/port are reachable from this machine.")
                        print("- Check credentials/permissions and any VPN/firewall rules.")
                        print("- Try a higher --preflight-timeout or --skip-preflight to attempt the query anyway.")
                        return 1

                # Determine expected argument name from input schema
                arg_name = "sql"
                try:
                    schema = getattr(target, "inputSchema", {}) or {}
                    props: List[str] = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
                    # Prefer common synonyms if present
                    for candidate in ("sql", "query", "statement"):
                        if candidate in props:
                            arg_name = candidate
                            break
                    if props:
                        print(f"Detected base_readQuery args: {props} (using '{arg_name}')")
                except Exception:
                    pass

                print(f"\nRunning SQL: {sql}")
                result = await asyncio.wait_for(
                    session.call_tool("base_readQuery", {arg_name: sql}),
                    timeout=args.timeout,
                )

                if result.content:
                    print("\n--- Content ---")
                    for c in result.content:
                        if isinstance(c, types.TextContent):
                            print(c.text)
                        else:
                            print(str(c))

                if hasattr(result, "structuredContent") and result.structuredContent:
                    print("\n--- Structured Content ---")
                    print(result.structuredContent)

                return 0
    except asyncio.TimeoutError:
        print("\nTimed out waiting for tool response.")
        return 1
    except Exception as e:
        print(f"\nMCP readQuery failed: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
