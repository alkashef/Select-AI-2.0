"""Slim Teradata helper for manual DB script tests.

Reads connection settings from config/.env via python-dotenv.
Environment variables required:
  - TD_HOST, TD_NAME, TD_USER, TD_PASSWORD
Optional:
  - TD_PORT (default 1025), TD_TEST_QUERY
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
import teradataml as tdml
from teradataml import execute_sql


class TestDBHelper:
    def __init__(self) -> None:
        # Load env from repo-local config/.env regardless of CWD
        env_path = Path(__file__).resolve().parent.parent / "config" / ".env"
        load_dotenv(env_path)
        # Read required values with a couple of friendly aliases
        self.host = os.getenv("TD_HOST") or os.getenv("TERADATA_HOST")
        self.database = os.getenv("TD_NAME") or os.getenv("TD_DATABASE") or os.getenv("TERADATA_DB")
        self.user = os.getenv("TD_USER") or os.getenv("TERADATA_USER")
        self.password = os.getenv("TD_PASSWORD") or os.getenv("TERADATA_PASSWORD")
        self.port = os.getenv("TD_PORT", "1025")
        self.connection = None

    def connect(self) -> None:
        self.connection = tdml.create_context(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
        )

    def disconnect(self) -> None:
        if self.connection:
            tdml.remove_context()
            self.connection = None

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        if not self.connection:
            raise RuntimeError("Call connect() first")
        rs = execute_sql(query)
        cols = [d[0] for d in rs.description] if rs.description else []
        rows = rs.fetchall()
        return [{cols[i]: v for i, v in enumerate(r)} for r in rows]

    def _get_sample_data(self, table: str) -> str:
        db_qual = f"{self.database}." if self.database else ""
        sql = f"SELECT * FROM {db_qual}{table} SAMPLE 3;"
        rs = execute_sql(sql)
        if rs.rowcount == 0:
            return "No data available"
        cols = [c[0] for c in rs.description]
        out: List[str] = ["columns: " + ", ".join(cols)]
        for i, row in enumerate(rs.fetchall()):
            vals = ["NULL" if v is None else str(v) for v in row]
            out.append(f"row{i+1}: {', '.join(vals)}")
        return "\n".join(out)

    def get_schema(self) -> str:
        if not self.connection:
            raise RuntimeError("Call connect() first")
        target_db = (self.database or "").strip()
        sql = (
            "SELECT t.tablename, c.columnname, c.columntype "
            "FROM dbc.tablesv t JOIN dbc.columnsv c ON t.tablename = c.tablename AND t.databasename = c.databasename "
            f"WHERE t.databasename = '{target_db}' AND t.TableKind IN ('T','V') ORDER BY t.tablename, c.columnid"
        )
        td_map = {"CV": "String", "D": "Numeric", "CF": "Numeric", "I": "Integer", "F": "Float"}
        rs = execute_sql(sql)
        if rs.rowcount == 0:
            return "(empty schema)"
        schema: Dict[str, List[str]] = {}
        for row in rs.fetchall():
            table, col, typ = row[0], row[1], row[2]
            schema.setdefault(table, []).append(f"{col} ({td_map.get(typ.strip(), typ.strip())})")
        parts: List[str] = []
        for table, cols in schema.items():
            section = f"\n\nTable: {table}\nColumns:\n  - " + "\n  - ".join(cols)
            sample = self._get_sample_data(table)
            section += f"\n\nSample data:\n{sample}" if sample else "\n\nSample data: No data available"
            parts.append(section)
        return "\n".join(parts)
