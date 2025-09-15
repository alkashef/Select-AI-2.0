"""MCP diagnostics over HTTP (tool-only): list tools, prompts, resources.

Usage:
    python -m tests.test_mcp_diag
    python -m tests.test_mcp_diag --url http://localhost:8001/mcp/
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Dict, Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from tests.mcp_runner import spawn_http_server, stop_server
from tests.mcp_helpers import load_env_from_config, load_yaml_config, resolve_mcp_url


def _load_env() -> None:
    load_env_from_config()


async def run(url: str) -> int:
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

        print("\n--- Tools ---")
        if tools_by_name:
            for name in tools_by_name:
                print(f"- {name}")
        else:
            print("(none)")

        # Prompts and resources are not standardized in HTTP adapter yet; attempt via raw session if exposed
        try:
            prompts = await session.list_prompts()
            prompt_list = getattr(prompts, "prompts", []) or []
        except Exception:
            prompt_list = []

        print("\n--- Prompts ---")
        if prompt_list:
            for p in prompt_list:
                name = getattr(p, "name", "")
                desc = getattr(p, "description", "")
                print(f"- {name}: {desc}")
        else:
            print("(none)")

        try:
            resources = await session.list_resources()
            resource_list = getattr(resources, "resources", []) or []
        except Exception:
            resource_list = []

        print("\n--- Resources ---")
        if resource_list:
            for r in resource_list:
                name = getattr(r, "name", "")
                uri = getattr(r, "uri", "")
                print(f"- {name}: {uri}")
        else:
            print("(none)")

        return 0


async def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(description="MCP diagnostics over HTTP (tool-only)")
    parser.add_argument("--config", default="tests/mcp_from_app.yml", help="From-app HTTP spawn config")
    parser.add_argument("--url", default=None, help="If provided, skip spawn and use this MCP URL")
    args = parser.parse_args()
    proc = None
    try:
        if args.url:
            url = args.url
        else:
            proc, url = spawn_http_server(args.config)
        return await run(url)
    except Exception as e:
        print(f"MCP HTTP diagnostics failed: {e!r}")
        return 1
    finally:
        if proc is not None:
            stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
