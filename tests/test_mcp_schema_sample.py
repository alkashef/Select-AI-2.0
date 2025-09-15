"""HTTP MCP script (tool-only): extract schema and sample rows.

Usage (Windows cmd):
    python -m tests.test_mcp_schema_sample --database BANK_DB --limit 3 --rows 5

Requirements:
- MCP server running separately (HTTP) and `MCP_URL` set or passed via --url.
- No SQL. Uses only MCP tools: base_tableList, base_columnDescription, base_tablePreview.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from tests.mcp_runner import spawn_http_server, stop_server
from tests.mcp_helpers import load_env_from_config, print_mcp_result


async def run(url: str, database: str, limit: int, rows: int, tables_filter: Optional[List[str]], timeout: float) -> int:
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

        table_list_tool = tools_by_name.get("base_tableList")
        col_desc_tool = tools_by_name.get("base_columnDescription")
        preview_tool = tools_by_name.get("base_tablePreview")
        if not (table_list_tool and col_desc_tool and preview_tool):
            print("Missing required MCP tools: base_tableList, base_columnDescription, base_tablePreview.")
            return 1

        # 1) List tables in the database (limit for sanity)
        if tables_filter:
            table_names = tables_filter
            print(f"Using provided tables: {', '.join(table_names)}")
        else:
            print(f"\nListing tables in {database} via base_tableList...")
            tables_res = await asyncio.wait_for(table_list_tool.ainvoke({"database_name": database}), timeout=timeout)
            table_names: List[str] = []
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

            if not table_names:
                print("No tables found or failed to parse results.")
                return 1

            # Apply client-side limit
            table_names = table_names[:limit]
            print(f"Found {len(table_names)} tables (after limiting to {limit}):")
            for t in table_names:
                print(f"- {t}")

        # 2) For each table, fetch column schema and sample data
        for tname in table_names:
            print(f"\n=== {database}.{tname} ===")
            # Columns via tool
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
    parser = argparse.ArgumentParser(description="Extract schema and sample rows via MCP (spawns HTTP server unless --url provided)")
    parser.add_argument("--config", default="tests/mcp_from_app.yml", help="From-app HTTP spawn config")
    parser.add_argument("--url", default=None, help="If provided, skip spawn and use this MCP URL")
    parser.add_argument("--database", default=os.getenv("TD_NAME", ""), help="Database name to inspect")
    parser.add_argument("--limit", type=int, default=3, help="Max number of tables to inspect")
    parser.add_argument("--rows", type=int, default=5, help="Sample rows per table")
    parser.add_argument("--tables", default="", help="Comma-separated list of specific tables to inspect")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds for each tool call")
    args = parser.parse_args()

    if not args.database:
        print("Provide --database to proceed (tables alone are insufficient for tool calls).")
        return 2

    tables_filter = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None

    proc = None
    try:
        if args.url:
            url = args.url
        else:
            proc, url = spawn_http_server(args.config)
        return await run(url, args.database, args.limit, args.rows, tables_filter, args.timeout)
    finally:
        if proc is not None:
            stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
