"""HTTP MCP smoke test (tool-only) with YAML support.

Behavior:
- Loads env from `config/.env`.
- Uses YAML config (optional) to resolve the MCP URL.
- Connects to MCP HTTP server and lists available tools.

Usage (Windows cmd):
    python tests\test_mcp_list_tools.py --config tests\mcp_schema_sample.yml
    python tests\test_mcp_list_tools.py --url http://localhost:8001/mcp/
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
try:
    from tests.mcp_helpers import load_env_from_config, load_yaml_config, resolve_mcp_url
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.mcp_helpers import load_env_from_config, load_yaml_config, resolve_mcp_url


def _load_env() -> None:
    load_env_from_config()


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
    parser = argparse.ArgumentParser(description="List tools from HTTP MCP server (tool-only)")
    parser.add_argument("--config", default="tests/mcp_schema_sample.yml")
    parser.add_argument("--url", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    url = resolve_mcp_url(args.url, cfg)

    try:
        return await run(url, args.timeout)
    except Exception as e:
        print(f"MCP HTTP client failed: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
