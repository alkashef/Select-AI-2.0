"""HTTP MCP example: run a simple SQL via base_readQuery.

Usage (Windows cmd):
    python tests\test_mcp_read_query.py --url http://localhost:8001/mcp/ --sql "SELECT CURRENT_DATE" --timeout 120

If --sql is omitted, uses TD_TEST_QUERY from env, else SELECT CURRENT_DATE.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
try:
    from tests.mcp_helpers import load_env_from_config, print_mcp_result
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.mcp_helpers import load_env_from_config, print_mcp_result


def _load_env() -> None:
    load_env_from_config()


async def run(url: str, sql: str, timeout: float) -> int:
    client = MultiServerMCPClient({
        "mcp_server": {
            "url": url,
            "transport": "streamable_http",
        }
    })

    session_ctx = client.session("mcp_server")
    async with session_ctx as session:
        tools = await load_mcp_tools(session)
        tools_by_name: Dict[str, Any] = {t.name: t for t in tools}
        if "base_readQuery" not in tools_by_name:
            print("Tool 'base_readQuery' not available.")
            return 1

        tool = tools_by_name["base_readQuery"]
        print(f"Running base_readQuery: {sql}")
        try:
            result = await asyncio.wait_for(tool.ainvoke({"sql": sql}), timeout=timeout)
        except asyncio.TimeoutError:
            print("Timed out waiting for base_readQuery response.")
            return 1

        print_mcp_result(result)

        return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="Call base_readQuery via HTTP MCP")
    parser.add_argument("--url", default=os.getenv("MCP_URL", "http://localhost:8001/mcp/"), help="MCP server HTTP endpoint")
    parser.add_argument("--sql", type=str, default=None, help="SQL to run; ensure it's safe and small")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds for tool call")
    args = parser.parse_args()

    _load_env()
    sql = (args.sql or os.getenv("TD_TEST_QUERY") or "SELECT CURRENT_DATE").strip()
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()

    try:
        return await run(args.url, sql, args.timeout)
    except Exception as e:
        print(f"MCP HTTP readQuery failed: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
