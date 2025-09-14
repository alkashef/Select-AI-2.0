"""YAML-driven MCP test (tool-only): get DB schema and samples.

Usage (Windows cmd):
    python tests\test_mcp_yml.py --config tests\mcp_schema_sample.yml

CLI overrides:
    --url, --database, --tables, --limit, --rows

Notes:
- This script is SQL-ignorant. It does not craft SQL or use base_readQuery.
- Requires MCP tools: base_tableList, base_columnDescription, base_tablePreview.
"""

from __future__ import annotations
import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
try:
    from tests.mcp_helpers import (
        load_env_from_config,
        print_mcp_result,
        load_yaml_config,
        resolve_mcp_url,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.mcp_helpers import (
        load_env_from_config,
        print_mcp_result,
        load_yaml_config,
        resolve_mcp_url,
    )
    

def load_config(path: str) -> Dict[str, Any]:
    return load_yaml_config(path)


async def run(url: str, database: str, limit_tables: int, rows_per_table: int, tables: Optional[List[str]], timeout: float) -> int:
    client = MultiServerMCPClient({
        "mcp_server": {
            "url": url,
            "transport": "streamable_http",
        }
    })

    async with client.session("mcp_server") as session:
        tools = await load_mcp_tools(session)
        tools_by_name: Dict[str, Any] = {t.name: t for t in tools}
        # Require dedicated server tools (no SQL fallbacks)
        table_list_tool = tools_by_name.get("base_tableList")
        col_desc_tool = tools_by_name.get("base_columnDescription")
        preview_tool = tools_by_name.get("base_tablePreview")
        if not (table_list_tool and col_desc_tool and preview_tool):
            print("Missing required MCP tools: base_tableList, base_columnDescription, base_tablePreview.")
            return 1

        # Determine tables
        table_names: List[str]
        if tables:
            table_names = tables
            print(f"Using tables from config/CLI: {', '.join(table_names)}")
        else:
            print(f"Listing tables in {database} via base_tableList...")
            tables_res = await asyncio.wait_for(table_list_tool.ainvoke({"database_name": database}), timeout=timeout)
            # Extract names from structured content
            table_names = []
            try:
                from json import loads
                payload = tables_res if isinstance(tables_res, dict) else loads(tables_res) if isinstance(tables_res, str) else None
                rows_list = (payload or {}).get("results", [])
                for row in rows_list:
                    name = row.get("TableName") or row.get("tablename") or row.get("name")
                    if name:
                        table_names.append(name)
            except Exception:
                print_mcp_result(tables_res)
                print("Could not parse table names from base_tableList response.")
                return 1

        # Apply client-side limit after listing (tableList has no limit param)
        table_names = table_names[:limit_tables]

        for tname in table_names:
            print(f"\n=== {database}.{tname} ===")
            # Column description (schema) via tool
            try:
                cols_res = await asyncio.wait_for(col_desc_tool.ainvoke({"database_name": database, "obj_name": tname}), timeout=timeout)
                print("-- Schema --")
                print_mcp_result(cols_res)
            except asyncio.TimeoutError:
                print("Schema query timed out.")

            # Sample rows via preview tool
            try:
                data_res = await asyncio.wait_for(preview_tool.ainvoke({"database_name": database, "table_name": tname}), timeout=timeout)
                print("-- Sample Rows --")
                print_mcp_result(data_res)
            except asyncio.TimeoutError:
                print("Sample query timed out.")

    return 0


async def main() -> int:
    load_env_from_config()
    parser = argparse.ArgumentParser(description="YAML-driven MCP schema+sample test")
    parser.add_argument("--config", default="tests/mcp_schema_sample.yml")
    parser.add_argument("--url", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--tables", default=None, help="Comma-separated list of tables")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    url = resolve_mcp_url(args.url, cfg)
    tables = None
    if args.tables:
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    elif isinstance(cfg.get("tables"), list):
        tables = [str(t) for t in cfg.get("tables")]

    database = args.database or cfg.get("database", os.getenv("TD_NAME", ""))
    limit_tables = args.limit or int(cfg.get("limit_tables", 3))
    rows_per_table = args.rows or int(cfg.get("rows_per_table", 5))

    # Database is required (both for listing and per-table tool calls)
    if not database:
        print("Provide --database (or set in YAML). Tables alone are insufficient.")
        return 2

    return await run(url, database, limit_tables, rows_per_table, tables, args.timeout)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
