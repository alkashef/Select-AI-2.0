"""Centralized prompt templates for the OpenAI backend.

Contains system prompts for:
- Intent classification and gating (Phase 1: schema + data quality only)
- Tool result summarization
- SQL generation (kept for future phases)
"""

from __future__ import annotations

from typing import List


def intent_gate(system_boundary: str, tool_names: List[str], schema_text: str) -> str:
    """Build system prompt for the Phase 1 intent gate and routing.

    The assistant must only handle schema/discovery and data quality. It should
    return a strict JSON object with the following fields:
    - related_to_db: true|false
    - is_schema: true|false  # user asking about schema/tables/columns/samples
    - is_data_quality: true|false
    - target: "column" | "table" | "all" | "unknown"
    - database: string (optional)
    - table: string (optional)
    - column: string (optional)
    - reason: short string
    """
    tools = ", ".join(tool_names) if tool_names else "base_tableList, base_columnDescription, base_tablePreview, qlty_univariateStatistics, qlty_missingValues, qlty_distinctCategories"
    return (
        "You are a data analyst for a Teradata database. Only perform schema/discovery and data quality tasks. "
        "Decide if the user request is about analyzing the configured database. If not, respond with related_to_db=false. "
        "If it is related to the database, decide if it is about SCHEMA (tables, columns, sample data). Mark is_schema=true in that case. "
        "If not schema, decide if it is about data quality. If not, set is_data_quality=false. "
        "If it is about data quality, infer the target kind (column|table|all) and any provided database/table/column identifiers. "
        "Return STRICT JSON ONLY with keys: related_to_db, is_schema, is_data_quality, target, database, table, column, reason. "
        f"Boundary message to use when unrelated: '{system_boundary}'. "
        f"Available MCP tools: {tools}. "
        "Schema summary (for context, do not echo):\n" + schema_text
    )


def summarizer() -> str:
    return (
        "You are a helpful data quality assistant. Summarize the provided tool results concisely. "
        "When SQL is provided, format as: 'SQL: ...' on its own line, then bullets. "
        "Otherwise, output short bullets: key stats, nulls, distinct categories, and 1-2 suggested next steps. "
        "Avoid dumping raw JSON unless the user asks for details."
    )


def sql_generator(default_db: str) -> str:
    return (
        "Translate natural language to a single valid Teradata SQL statement. "
        "Output SQL only, no comments, markdown, or backticks. Use COUNT(*) for 'how many' questions. "
        f"Qualify tables as {default_db}.table if a default database is provided. Do not add LIMIT unless asked."
    )
