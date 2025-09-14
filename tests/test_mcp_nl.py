"""NL→SQL MCP test (tool-only): Ask in natural language and let the server translate & run.

Usage (Windows cmd):
    python tests/test_mcp_nl.py --config tests/mcp_nl.yml

Notes:
- SQL-ignorant client: no SQL crafting or direct DB access.
- Relies on server-exposed tools/workflows/prompts for NL→SQL.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

try:
    from tests.mcp_helpers import (
        load_env_from_config,
        load_yaml_config,
        resolve_mcp_url,
        print_mcp_result,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.mcp_helpers import (
        load_env_from_config,
        load_yaml_config,
        resolve_mcp_url,
        print_mcp_result,
    )


def _build_args_for_tool(tool: Any, question: str, database: Optional[str], extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a minimal, schema-compliant arg dict for the tool.

    - Prefer passing the NL question under a common key: 'question', 'input', 'query', or 'prompt'.
    - Include 'database_name' only if the tool explicitly supports it.
    - If schema is unavailable, default to {'question': question}.
    """
    allowed: set[str] = set()
    try:
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            if hasattr(schema, "model_fields"):  # pydantic v2
                allowed = set(schema.model_fields.keys())
            elif hasattr(schema, "__fields__"):  # pydantic v1
                allowed = set(schema.__fields__.keys())
    except Exception:
        allowed = set()

    args: Dict[str, Any] = {}
    question_keys = ("question", "input", "query", "prompt")
    chosen_key: Optional[str] = None
    if allowed:
        for key in question_keys:
            if key in allowed:
                chosen_key = key
                break
        if chosen_key is None:
            # Fall back to the first available field name
            try:
                chosen_key = next(iter(allowed))
            except StopIteration:
                chosen_key = None
    else:
        chosen_key = "question"

    if chosen_key:
        args[chosen_key] = question

    # Include database under an accepted alias if explicitly allowed
    if database and allowed:
        for db_key in ("database_name", "database", "db", "schema", "databasename"):
            if db_key in allowed:
                args[db_key] = database
                break

    # Include optional extras only when schema allows them
    if extras and allowed:
        for key, val in extras.items():
            if key in allowed:
                args[key] = val

    return args


async def run(url: str, question: str, database: Optional[str], timeout: float, extras: Optional[Dict[str, Any]] = None) -> int:
    client = MultiServerMCPClient({
        "mcp_server": {
            "url": url,
            "transport": "streamable_http",
        }
    })

    async with client.session("mcp_server") as session:
        tools = await load_mcp_tools(session)
        tools_by_name: Dict[str, Any] = {t.name: t for t in tools}

        # Prefer a dedicated NL tool, else a workflow tool
        nl_tool = None
        for candidate in ("nl_query", "text_to_sql", "nl2sql", "ask_db", "ask_database"):
            if candidate in tools_by_name:
                nl_tool = tools_by_name[candidate]
                break

        workflow_tool = tools_by_name.get("rag_executeWorkflow") or tools_by_name.get("rag_Execute_Workflow")

        if nl_tool is None and workflow_tool is None:
            print("No NL→SQL or workflow tool found on the MCP server.")
            print("Ensure the server exposes a tool to accept a natural language question and execute it.")
            return 1

        # Build args that match the tool's accepted schema
        selected_tool = nl_tool or workflow_tool
        args = _build_args_for_tool(selected_tool, question, database, extras)

        # Lightweight debug: show tool name and arg keys being sent (no values)
        try:
            tool_name = getattr(selected_tool, "name", str(selected_tool))
            print(f"Using tool: {tool_name}")
            print(f"Sending args: {', '.join(sorted(args.keys())) or '(none)'}")
        except Exception:
            pass

        try:
            if nl_tool is not None:
                result = await asyncio.wait_for(nl_tool.ainvoke(args), timeout=timeout)
            else:
                result = await asyncio.wait_for(workflow_tool.ainvoke(args), timeout=timeout)
        except asyncio.TimeoutError:
            print("Timed out waiting for NL→SQL response from MCP server.")
            return 1

        print_mcp_result(result)

        return 0


def main() -> int:
    load_env_from_config()
    parser = argparse.ArgumentParser(description="Natural language query via MCP (tool-only)")
    parser.add_argument("--config", default="tests/mcp_nl.yml")
    parser.add_argument("--url", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--k", type=int, default=None, help="Optional RAG top-k if supported by the tool")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    url = resolve_mcp_url(args.url, cfg)
    question = args.question or cfg.get("question")
    database = args.database or cfg.get("database")
    timeout = args.timeout or float(cfg.get("timeout", 120))

    # Optional extras to pass only if tool supports them
    extras: Dict[str, Any] = {}
    k_val = args.k if args.k is not None else cfg.get("k")
    if k_val is not None:
        try:
            extras["k"] = int(k_val)
        except Exception:
            pass

    if not question:
        print("Provide a natural language question via --question or YAML 'question'.")
        return 2

    return asyncio.run(run(url, question, database, timeout, extras))


if __name__ == "__main__":
    raise SystemExit(main())
