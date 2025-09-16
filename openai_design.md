# OpenAI + MCP Integration Design

Goal: Add a new backend `ai/openai.py` that implements `AI.generate_reply` and orchestrates between the user and the Teradata MCP server (via streamable HTTP). The user chats with the UI; the backend decides when/how to call MCP tools (read-only) to answer database questions. This is a demo/PoC (dev-only), so we favor visibility for debugging over strict caps.

## Architecture Overview

- UI: Streamlit chat sends the full message history to `AI.generate_reply(messages, context)`.
- Backend: `ai.openai.AI_OpenAI` (new) orchestrates:
  - Uses OpenAI Chat Completions to plan and respond.
  - Detects DB intent in-text (LLM-led, with simple heuristic fallback).
  - Calls MCP tools (streamable HTTP) for read-only DB introspection/query.
  - Produces concise answers including both SQL (when used) and a summary.
- MCP Client: `langchain-mcp-adapters` creates an HTTP session to `MCP_URL`. Dev-only: optional spawn using a `tests/mcp_http.yml`-like config.
- MCP Server: `teradata-mcp-server` in streamable-http mode connects to the DB via `DATABASE_URI`.

## Data Flow

1. User message enters `generate_reply`.
2. LLM determines if MCP tools are needed (schema lookup, table preview, or SQL read).
3. If needed:
   - Ensure MCP session (connect to `MCP_URL`; spawn in dev if enabled).
   - Invoke tools (e.g., `base_tableList`, `base_columnDescription`, `base_readQuery`).
   - Normalize outputs.
4. Construct a response including SQL (if used) and a readable summary.
5. Return final text.

## Responsibilities

- AI backend (OpenAI):
  - Prompting: minimal system prompt with tool-use policy.
  - Decision: choose which tools to call and when.
  - Execution: run tools over HTTP; manage timeouts/retries.
  - Summarization: transform raw JSON into user-friendly text.
- MCP client/runner:
  - Prefer URL `MCP_URL` (prod). Optional spawn (dev-only) via a wrapper around `tests.mcp_runner.spawn_http_server`.
  - Cleanly stop spawned processes.

## Configuration

- `config/.env`:
  - `OPENAI_API_KEY`, `OPENAI_MODEL` (e.g., `gpt-4o`, `gpt-4o-mini`).
  - `MCP_URL` (http://localhost:8001/mcp/).
  - `MCP_SPAWN` (0/1) for dev-only spawn.
  - `DATABASE_URI` (for spawn mode only).
  - `MCP_TIMEOUT` per tool call (60–120s typical).
  - `TD_NAME` default database when user doesn’t specify.

## Factory Integration

- Add `ai/openai.py` defining `AI_OpenAI` (subclass of `AI`).
- Update `ai/factory.py` to support `AI_BACKEND=openai`.
- Keep `ai/gpt.py` as the LLM-only backend.

## Tooling Interface (Read-only)

- `base_tableList(database_name)`
- `base_columnDescription(database_name, obj_name)`
- `base_tablePreview(database_name, table_name)` (optional)
- `base_readQuery(sql)`
- Data quality (if exposed by the server):
  - `qlty_univariateStatistics(database_name, table_name, column_name)`
  - `qlty_missingValues(database_name, table_name)`
  - `qlty_distinctCategories(database_name, table_name, column_name)`
- Client: `langchain_mcp_adapters` with `MultiServerMCPClient` and `load_mcp_tools`.

## Session Strategy

- Prefer MCP via `MCP_URL` (streamable HTTP) and reuse session.
- Dev-only: if `MCP_SPAWN=1`, spawn on first use and stop on shutdown/teardown.

## Prompting Strategy

- Only call MCP tools when necessary.
- Ask clarifying questions if key constraints are missing (DB name, filters). Use `TD_NAME` as default DB.
- Provide concise summaries; do not impose client-side row caps unless user prompted (let MCP decide).
- When `base_readQuery` is used, include both SQL and a short summary.

## Error Handling

- LLM/API errors: retry with backoff (similar to `ai.gpt`).
- MCP connection errors: surface clear messages; in spawn mode, attempt start and show tail on failure.
- Tool timeouts: provide partials if possible and guidance to narrow scope.
- No write operations (read-only policy enforced in prompts and tool selection).

## Telemetry & Logging

- Events: `ai_openai.init`, `ai_openai.call`, `ai_openai.tool.call`, `ai_openai.tool.error`.
- Add log lines with prefixes:
  - `[LLM=>MCP]` tool name and payload (sanitize; include SQL if present).
  - `[MCP=>LLM]` response payload (for debugging, avoid truncation unless outputs are extremely large; never log secrets).
- Never log credentials; mask passwords in URIs; cap payload size.

## Security & Secrets

- No hardcoded keys. Load from `config/.env`.
- Never echo sensitive connection strings; mask before logging.

## Open Items

1. Confirm dev-only spawn vs prod URL-only (default assumed).
2. Confirm logging truncation limit for `[MCP=>LLM]` (e.g., 8–16 KB).
3. Confirm `TD_NAME` as the canonical env var for default DB.

## MVP Scope

- One-shot replies (no token streaming to UI initially).
- Read-only tools.
- URL-based MCP by default; optional dev spawn.
- LLM-led intent detection with lightweight keyword fallback.
