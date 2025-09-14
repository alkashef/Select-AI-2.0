"""HTTP MCP script: extract schema and sample rows from a database.

Usage (Windows cmd):
  python tests\test_mcp_schema_sample.py --database BANK_DB --limit 3 --rows 5

Requires:
- MCP server running separately (see README) and `MCP_URL` configured.
- Database name via `--database` or `TD_NAME` in config/.env.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, List

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
try:
    from tests.mcp_helpers import load_env_from_config, print_mcp_result
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.mcp_helpers import load_env_from_config, print_mcp_result


async def run(url: str, database: str, limit: int, rows: int, tables_filter: List[str] | None, timeout: float) -> int:
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
            print("Tool 'base_readQuery' not available on the MCP server.")
            return 1
        read_query = tools_by_name["base_readQuery"]

        # 1) List tables in the database (limit for sanity)
        if tables_filter:
            table_names = tables_filter
            print(f"Using provided tables: {', '.join(table_names)}")
        else:
            sql_tables = (
                f"SELECT TOP {limit} TableName, TableKind "
                f"FROM DBC.TablesV WHERE DatabaseName='{database}' ORDER BY TableName"
            )
            print(f"\nListing up to {limit} tables in {database}...")
            tables_res = await asyncio.wait_for(read_query.ainvoke({"sql": sql_tables}), timeout=timeout)
            # Expect a JSON payload with 'results'
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
                # Fallback: just pretty-print and bail
                print_mcp_result(tables_res)
                print("Could not parse table names from response.")
                return 1

            if not table_names:
                print("No tables found or failed to parse results.")
                return 1

            print(f"Found {len(table_names)} tables (showing up to {limit}):")
            for t in table_names:
                print(f"- {t}")

        # 2) For each table, fetch column schema and sample data
        for tname in table_names:
            print(f"\n=== {database}.{tname} ===")
            # Columns
            sql_cols = (
                "SELECT ColumnId, ColumnName, ColumnType, ColumnLength, "
                "DecimalTotalDigits, DecimalFractionalDigits, Nullable, ColumnFormat "
                f"FROM DBC.ColumnsV WHERE DatabaseName='{database}' AND TableName='{tname}' "
                "ORDER BY ColumnId"
            )
            try:
                cols_res = await asyncio.wait_for(read_query.ainvoke({"sql": sql_cols}), timeout=timeout)
                print("-- Schema --")
                print_mcp_result(cols_res)
            except asyncio.TimeoutError:
                print("Schema query timed out.")

            # Sample rows
            sql_sample = f"SELECT TOP {rows} * FROM {database}.{tname}"
            try:
                data_res = await asyncio.wait_for(read_query.ainvoke({"sql": sql_sample}), timeout=timeout)
                print("-- Sample Rows --")
                print_mcp_result(data_res)
            except asyncio.TimeoutError:
                print("Sample query timed out.")

        return 0


async def main() -> int:
    load_env_from_config()
    parser = argparse.ArgumentParser(description="Extract schema and sample rows via MCP HTTP")
    parser.add_argument("--url", default=os.getenv("MCP_URL", "http://localhost:8001/mcp/"), help="MCP server HTTP endpoint")
    parser.add_argument("--database", default=os.getenv("TD_NAME", ""), help="Database name to inspect")
    parser.add_argument("--limit", type=int, default=3, help="Max number of tables to inspect")
    parser.add_argument("--rows", type=int, default=5, help="Sample rows per table")
    parser.add_argument("--tables", default="", help="Comma-separated list of specific tables to inspect")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds for each tool call")
    args = parser.parse_args()

    if not args.database and not args.tables:
        print("Provide --database or --tables to proceed.")
        return 2

    tables_filter = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None
    return await run(args.url, args.database, args.limit, args.rows, tables_filter, args.timeout)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
