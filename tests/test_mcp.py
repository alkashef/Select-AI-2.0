"""Minimal MCP client smoke test for Teradata MCP Server.

This script:
- Loads `config/.env` for TD_* vars and/or `DATABASE_URI`.
- Spawns `teradata-mcp-server` via stdio transport.
- Lists available tools and their expected arguments.
- Heuristically finds a SQL-like tool and calls it with a test query.

Usage (Windows cmd):
  python tests\test_mcp.py

Env options:
- DATABASE_URI=teradata://<USER>:<PASS>@<HOST>:1025/<DB>
- Or set TD_USER, TD_PASSWORD, TD_HOST, TD_NAME, and optional TD_TEST_QUERY
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict
import argparse
import anyio
from urllib.parse import quote as url_quote

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


def _load_env() -> None:
    """Load variables from config/.env if present."""
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _build_database_uri() -> Optional[str]:
    """Prefer DATABASE_URI; else synthesize from TD_* vars."""
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


def _pick_sql_tool(tools: List[types.Tool]) -> tuple[Optional[types.Tool], Optional[str]]:
    """Pick a tool likely to execute SQL and return it with its primary param name.

    Heuristics:
    - Prefer tools whose input schema has a 'sql' or 'query' property.
    - Else prefer tool names containing 'sql' or 'query'.
    - Fallback to the first property name if present.
    """

    def schema_props(tool: types.Tool) -> List[str]:
        props: List[str] = []
        schema = getattr(tool, "inputSchema", None)
        if isinstance(schema, dict):
            props = list((schema.get("properties") or {}).keys())
        return [p for p in props if isinstance(p, str)]

    candidates: List[tuple[types.Tool, List[str]]] = [(t, schema_props(t)) for t in tools]

    # 1) property-based match
    for t, props in candidates:
        lower = [p.lower() for p in props]
        if "sql" in lower:
            return t, props[lower.index("sql")]
        if "query" in lower:
            return t, props[lower.index("query")]

    # 2) name-based match
    for t, props in candidates:
        name = (t.name or "").lower()
        if "sql" in name or "query" in name:
            # Fallback to first prop if available
            return t, (props[0] if props else None)

    # 3) fallback to first tool with any property
    for t, props in candidates:
        if props:
            return t, props[0]

    return None, None


async def main() -> int:
    parser = argparse.ArgumentParser(description="MCP client smoke test for Teradata")
    parser.add_argument("--list-tools", action="store_true", help="List tools and exit")
    parser.add_argument("--tool", type=str, default=None, help="Explicit tool name to call")
    parser.add_argument("--param", action="append", default=[], help="Tool param as key=value (repeatable)")
    parser.add_argument("--sql", type=str, default=None, help="Convenience: set SQL for tools expecting a 'sql' param")
    parser.add_argument("--timeout", type=float, default=None, help="Seconds to wait for tool call (overrides env)")
    parser.add_argument("--debug", action="store_true", help="Print computed DATABASE_URI and extra diagnostics")
    args = parser.parse_args()

    _load_env()
    db_uri = _build_database_uri()
    if not db_uri:
        print("DATABASE_URI not set and TD_* vars incomplete. Set config/.env or environment.")
        return 2

    # Build server env: DATABASE_URI plus TD_* passthrough to be safe
    server_env = {"DATABASE_URI": db_uri}
    for k in ("TD_HOST", "TD_NAME", "TD_USER", "TD_PASSWORD", "TD_PORT"):
        v = os.getenv(k)
        if v is not None:
            server_env[k] = v

    if args.debug:
        # Redact password in URI for display
        redacted = db_uri
        if "@" in db_uri and ":" in db_uri.split("@", 1)[0]:
            head, tail = db_uri.split("@", 1)
            scheme, rest = head.split("//", 1)
            if ":" in rest:
                user, pwd = rest.split(":", 1)
                redacted = f"{scheme}//{user}:***@{tail}"
        print(f"Using DATABASE_URI: {redacted}")
        if all(os.getenv(k) for k in ("TD_HOST", "TD_NAME", "TD_USER", "TD_PASSWORD", "TD_PORT")):
            print("TD_* present and passed through to server env.")

    server_params = StdioServerParameters(
        command="teradata-mcp-server",
        args=[],
        env=server_env,
    )

    print("Launching teradata-mcp-server via stdio...")
    had_timeout = False
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize session
                await session.initialize()
                if args.debug:
                    print("Initialized MCP session (stdio).")

                # List tools
                tools_resp = await session.list_tools()
                tools = tools_resp.tools
                if not tools:
                    print("No tools exposed by MCP server.")
                    return 1

                print("\n--- Available Tools ---")
                for t in tools:
                    schema = getattr(t, "inputSchema", None)
                    props = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
                    print(f"- {t.name}: args={props}")

                if args.list_tools:
                    return 0

                # Optionally select an explicit tool
                selected_tool = None
                if args.tool:
                    for t in tools:
                        if t.name == args.tool:
                            selected_tool = t
                            break
                    if not selected_tool:
                        names = ", ".join(sorted(t.name or "" for t in tools))
                        print(f"Tool '{args.tool}' not found. Available: {names}")
                        return 1
                    # Infer a default param if possible
                    schema = getattr(selected_tool, "inputSchema", None)
                    props = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
                    default_param = props[0] if props else None
                    tool, param = selected_tool, default_param
                else:
                    # Pick a SQL-like tool
                    tool, param = _pick_sql_tool(tools)
                if not tool:
                    print("\nNo suitable SQL-like tool found.")
                    return 1

                # Build params
                call_args: Dict[str, str] = {}
                # Parse key=value from --param
                for kv in args.param:
                    if "=" not in kv:
                        print(f"Ignoring malformed --param '{kv}', expected key=value")
                        continue
                    k, v = kv.split("=", 1)
                    call_args[k] = v

                # Convenience --sql mapping (ensure semicolon at end)
                if args.sql:
                    _sql = args.sql.strip()
                    if not _sql.endswith(";"):
                        _sql += ";"
                    call_args["sql"] = _sql

                # If nothing provided, auto-fill for readQuery style tools
                if not call_args:
                    if param == "sql" or (tool.name and tool.name.lower().endswith("readquery")):
                        test_query = os.getenv("TD_TEST_QUERY") or "SELECT CURRENT_DATE"
                        _sql = test_query.strip()
                        if not _sql.endswith(";"):
                            _sql += ";"
                        call_args["sql"] = _sql

                # If the tool has an inferred single parameter and we have exactly one arg with a different key, remap
                if not call_args and param:
                    print(f"\nSelected tool '{tool.name}' but couldn't infer parameter values; skipping call.")
                    return 0

                # Call timeout
                timeout_env = os.getenv("MCP_TOOL_TIMEOUT_SEC")
                timeout_sec = args.timeout if args.timeout is not None else float(timeout_env or "20")

                if call_args:
                    print(f"\nCalling tool '{tool.name}' with args {call_args} (timeout={timeout_sec:.0f}s)...")
                else:
                    print(f"\nCalling tool '{tool.name}' with no args (timeout={timeout_sec:.0f}s)...")

                try:
                    result = await asyncio.wait_for(
                        session.call_tool(tool.name, call_args),
                        timeout=timeout_sec,
                    )
                except asyncio.TimeoutError:
                    had_timeout = True
                    if args.tool:
                        print("\nTool call timed out.")
                        return 1
                    print("\nTool call timed out. Trying a fallback no-arg tool...")
                    # Try a simple no-arg diagnostic tool
                    fallback_name = None
                    for cand in ("dba_databaseVersion", "base_databaseList"):
                        if any(t.name == cand for t in tools):
                            fallback_name = cand
                            break
                    if not fallback_name:
                        for t in tools:
                            schema = getattr(t, "inputSchema", None)
                            props = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
                            if not props:
                                fallback_name = t.name
                                break
                    if fallback_name:
                        print(f"Calling fallback tool '{fallback_name}'...")
                        try:
                            result = await asyncio.wait_for(
                                session.call_tool(fallback_name, {}),
                                timeout=10,
                            )
                        except Exception as e:
                            print(f"Fallback tool '{fallback_name}' failed: {e!r}")
                            return 1
                    else:
                        print("No suitable fallback tool found.")
                        return 1
                except Exception as e:
                    print(f"\nTool call raised an error: {e!r}")
                    return 1

                # Print unstructured content
                if result.content:
                    print("\n--- Tool Content ---")
                    for c in result.content:
                        if isinstance(c, types.TextContent):
                            print(c.text)
                        else:
                            print(str(c))

                # Print structured content if provided
                if hasattr(result, "structuredContent") and result.structuredContent:
                    print("\n--- Structured Content ---")
                    print(result.structuredContent)

                # Print error flag if present
                if hasattr(result, "isError") and getattr(result, "isError"):
                    print("\nResult flagged as error by server.")

                return 0
    except BaseException as e:
        # The stdio transport may raise BrokenResourceError (possibly wrapped in a BaseExceptionGroup)
        # during shutdown if the subprocess closes streams while we exit early (e.g., after a timeout).
        if isinstance(e, BaseExceptionGroup):
            try:
                all_broken = all(isinstance(x, (anyio.BrokenResourceError, anyio.ClosedResourceError)) for x in e.exceptions)
            except Exception:
                all_broken = False
            if all_broken:
                if had_timeout:
                    print("\nTool call timed out and the server closed the stdio connection.")
                    return 1
                print("\nThe MCP server closed the stdio connection immediately. Check DATABASE_URI and server availability.")
                if args.debug:
                    print("Tip: Try running the server directly after setting DATABASE_URI in this shell:")
                    print("  teradata-mcp-server --help")
                    print("  teradata-mcp-server")
                return 1
        if isinstance(e, (anyio.BrokenResourceError, anyio.ClosedResourceError)):
            if had_timeout:
                print("\nTool call timed out and the server closed the stdio connection.")
                return 1
            print("\nThe MCP server closed the stdio connection. Check DATABASE_URI and server availability.")
            if args.debug:
                print("Tip: Try running the server directly after setting DATABASE_URI in this shell:")
                print("  teradata-mcp-server --help")
                print("  teradata-mcp-server")
            return 1
        print(f"\nUnexpected error: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
