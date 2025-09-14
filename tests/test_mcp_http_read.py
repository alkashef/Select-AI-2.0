"""HTTP MCP test against a running teradata-mcp-server.

Prereq:
- Start the server separately in streamable-http mode (Anaconda prompt):
    teradata-mcp-server --mcp_transport streamable-http --mcp_port 8001 --profile all
  Ensure it shows: http://localhost:8001/mcp/

Usage (Windows cmd):
  python tests\test_mcp_http_base_read_query.py --url http://localhost:8001/mcp/ --sql "SELECT CURRENT_DATE"

Notes:
- Uses langchain-mcp-adapters HTTP client to connect and invoke tools.
- Prints both raw text and parsed JSON when available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


async def run(url: str, sql: str, timeout: float) -> int:
    # Initialize HTTP client for MCP server
    client = MultiServerMCPClient({
        "mcp_server": {
            "url": url,
            "transport": "streamable_http",
        }
    })

    # Create and enter a session context
    session_ctx = client.session("mcp_server")
    async with session_ctx as session:
        # Load tools
        tools = await load_mcp_tools(session)
        tools_by_name: Dict[str, Any] = {t.name: t for t in tools}
        if not tools_by_name:
            print("No tools found on the MCP server.")
            return 1

        print("--- Tools ---")
        for name in tools_by_name:
            print(f"- {name}")

        if "base_readQuery" not in tools_by_name:
            print("Tool 'base_readQuery' not available.")
            return 1

        tool = tools_by_name["base_readQuery"]
        print(f"\nRunning base_readQuery: {sql}")
        try:
            # Invoke via LangChain MCP tool interface
            result = await asyncio.wait_for(tool.ainvoke({"sql": sql}), timeout=timeout)
        except asyncio.TimeoutError:
            print("Timed out waiting for base_readQuery response.")
            return 1

        # Normalize and display
        if isinstance(result, str):
            try:
                obj = json.loads(result)
                print(json.dumps(obj, indent=2))
            except json.JSONDecodeError:
                print(result)
        elif isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            # Some adapters return a list with text objects
            try:
                text = getattr(result[0], "text", None) if result else None
                if text:
                    try:
                        obj = json.loads(text)
                        print(json.dumps(obj, indent=2))
                    except json.JSONDecodeError:
                        print(text)
                else:
                    print(str(result))
            except Exception:
                print(str(result))

        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP MCP client test for Teradata")
    parser.add_argument("--url", default=os.getenv("MCP_URL", "http://localhost:8001/mcp/"), help="MCP server HTTP endpoint")
    parser.add_argument("--sql", default="SELECT CURRENT_DATE", help="SQL to run via base_readQuery")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds")
    args = parser.parse_args()

    return asyncio.run(run(args.url, args.sql, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
