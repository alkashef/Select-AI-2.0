"""STDIO MCP smoke test: spawn the server process and run one tool.

Usage (Windows cmd):
    python tests\\test_mcp_stdio.py --config tests\\mcp_stdio.yml

Notes:
 - This avoids a separate server terminal by launching the server via STDIO.
 - We build `DATABASE_URI` (with optional LOGMECH/ENCRYPTDATA) and pass it to
   the spawned process so it mirrors your working HTTP setup.
 - Keeps client SQL-ignorant; only calls server tools.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, Optional
import re

import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compose_server_env(env_cfg: dict) -> Dict[str, str]:
    env: Dict[str, str] = {}
    # Require DATABASE_URI either in YAML or OS env (.env loaded)
    dburi = env_cfg.get("DATABASE_URI") or os.environ.get("DATABASE_URI")
    if not dburi:
        raise RuntimeError("DATABASE_URI is required. Set it in tests/mcp_stdio.yml or config/.env")
    env["DATABASE_URI"] = dburi
    # Optional secure flags
    for key in ("LOGMECH", "ENCRYPTDATA", "CHARSET"):
        val = env_cfg.get(key)
        if val in (None, ""):
            val = os.environ.get(key)
        if val not in (None, ""):
            env[key] = str(val)
    return env


async def run_stdio(config_path: str) -> int:
    load_dotenv("config/.env")
    cfg = _load_yaml(config_path)
    server_cfg = cfg.get("server", {})
    env_cfg = cfg.get("env", {})
    client_cfg = cfg.get("client", {})

    command = server_cfg.get("command", "teradata-mcp-server")
    args = server_cfg.get("args", [])
    cwd = server_cfg.get("cwd")
    timeout = float(client_cfg.get("timeout", 60))
    tool_name = client_cfg.get("tool", "base_databaseList")

    server_env = _compose_server_env(env_cfg)

    # Quick visibility: show command and essential env (mask password)
    print("Command:", command, *args)
    if "DATABASE_URI" in server_env:
        # Mask password in the URI for display
        safe_uri = re.sub(r"(://[^:@]+:)[^@]+(@)", r"\1****\2", server_env["DATABASE_URI"])
        print("DATABASE_URI:", safe_uri)
    for key in ("LOGMECH", "ENCRYPTDATA", "CHARSET"):
        if key in server_env:
            print(f"{key}:", server_env[key])
    if "DATABASE_URI" not in server_env:
        print("DATABASE_URI is required in YAML or .env; aborting.")
        return 2

    client = MultiServerMCPClient({
        "mcp_server": {
            "command": command,
            "args": args,
            "cwd": cwd,
            "transport": "stdio",
            "env": server_env,
        }
    })

    try:
        async with client.session("mcp_server") as session:
            tools = await load_mcp_tools(session)
            selected = None
            for t in tools:
                if getattr(t, "name", None) == tool_name:
                    selected = t
                    break
            if selected is None:
                print(f"Tool not found: {tool_name}")
                print("Available tools:")
                for t in tools:
                    print(f" - {t.name}")
                return 1

            try:
                result = await asyncio.wait_for(selected.ainvoke({}), timeout=timeout)
            except asyncio.TimeoutError:
                print("Timed out waiting for tool response over STDIO.")
                print("Hints: verify DB credentials and that the server supports stdio transport.")
                return 1

            # Print normalized result
            print("Tool:", selected.name)
            print("Result:")
            if isinstance(result, (dict, list)):
                import json
                print(json.dumps(result, indent=2))
            else:
                print(str(result))
            return 0
    except Exception as e:
        # Common failure: BrokenResourceError indicates server process exited or protocol mismatch
        msg = str(e)
        print("STDIO session failed:", msg)
        print("Troubleshooting:")
        print(" - Ensure 'teradata-mcp-server' is on PATH in this env.")
        print(" - Confirm '--mcp_transport stdio' is supported by your server version.")
        print(" - Use 'all' profile as supported by your server.")
        print(" - Double-check DATABASE_URI and DB network access.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Spawn teradata-mcp-server via STDIO and run a tool")
    parser.add_argument("--config", default="tests/mcp_stdio.yml")
    args = parser.parse_args()
    return asyncio.run(run_stdio(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
