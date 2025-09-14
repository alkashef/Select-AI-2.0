"""HTTP MCP smoke test for Teradata MCP Server.

Behavior:
- Loads env from `config/.env`.
- Connects to a running MCP HTTP server at `MCP_URL`.
- Lists available tools and exits.

Usage (Windows cmd):
    python tests\test_mcp_list_tools.py --url http://localhost:8001/mcp/
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


async def run(url: str, timeout: float) -> int:
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
        if not tools_by_name:
            print("No tools exposed by MCP server.")
            return 1

        print("\n--- Available Tools ---")
        for name in tools_by_name:
            print(f"- {name}")
        return 0


async def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(description="List tools from HTTP MCP server")
    parser.add_argument("--url", default=os.getenv("MCP_URL", "http://localhost:8001/mcp/"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    try:
        return await run(args.url, args.timeout)
    except Exception as e:
        print(f"MCP HTTP client failed: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
