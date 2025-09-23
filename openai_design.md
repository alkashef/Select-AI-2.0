# OpenAI + MCP Integration Design

Goal: Add a new backend `ai/openai.py` that implements `AI.generate_reply` and orchestrates between the user and the Teradata MCP server (via streamable HTTP). The user chats with the UI; the backend decides when/how to call MCP tools (read-only) to answer database questions. This is a dev-first PoC, so we favor fast feedback and visible logging over strict caps.

## MCP Capabilities: What It Can/Can’t Do

- Transport: `streamable-http` (recommended on Windows) and `stdio` (used by tests). HTTP fits shared/remote servers; stdio fits local spawn.
- Scope (current PoC): read-only tools; generally one-shot interactions per call path.
- Core tools available (profiles determine availability):
  - Base: `base_tableList`, `base_columnDescription`, `base_tablePreview`, `base_tableDDL`, `base_readQuery`.
  - Data Quality: `qlty_univariateStatistics`, `qlty_missingValues`, `qlty_distinctCategories`.
  - Admin/DBA and others exist but are out of scope for analyst flows.
- Behavior in our PoC integration (LLM_MCP_INTEGRATION.md): planner picks one action; responses should include SQL (if used) + brief summary.
- Security: AuthN/AuthZ enforced at DB level; request context adds query banding; HTTP mode supports per-user auth patterns; profiles limit tool exposure.
- Not in scope for MVP: write operations (DML/DDL), multi-step server-side workflows, long-running jobs. We’ll keep client orchestration simple.

## Agent/Roles Architecture (Phase 1)

For Phase 1, we intentionally narrow the scope to one expert: Data Quality. The SQL Expert and BI Expert are deferred to later phases. We still keep a lightweight Data Analyst as the user-facing orchestrator.

1) Data Analyst (primary interface)
- Responsibilities: greet the user, set expectations, maintain conversation, route intents (schema/discovery, data_quality, smalltalk), and stitch results into friendly answers.
- Startup behavior: on app start, fetch and cache metadata for `TD_NAME` (tables and columns) via `base_tableList` + `base_columnDescription` per table. Cache in memory (and optionally a small JSON on disk) for quick reuse.
- Off-domain conversations: respond with a gentle boundary message, e.g., “I’m a data analyst specialized in analyzing the data in your XYZ database.”
- Routing: classify user intent into [schema/discovery, data_quality, smalltalk]. SQL/BI requests should be acknowledged and deferred as “coming soon”.

2) Data Quality Expert (Phase 1 focus)
- Purpose: describe data quality via descriptive stats, missing values, and categorical distributions.
- Tools: call `qlty_univariateStatistics`, `qlty_missingValues`, `qlty_distinctCategories`, and optionally `base_tablePreview` for small samples.
- Output: a short summary (central tendency, dispersion, missingness, category counts) and actionable next steps.

Intent Router
- Implementation: single LLM classification call (labels: schema, data_quality, smalltalk) with a tiny heuristic fallback (e.g., keywords: “missing”, “nulls”, “distinct”, “stats”, “schema”, “columns”, “tables”).
- Memory: include cached metadata summary in the router context to improve accuracy.

Execution Diagram (Phase 1)
```
User → Data Analyst → (Router)
  ├─ schema → MCP base_tableList/columnDescription/preview → Answer
  ├─ data_quality → DQ Expert → MCP qlty_* (+optional preview) → Summary → Answer
  └─ smalltalk → Boundary message → Answer
```

## Implementation Plan (Phase 1 MVP)

- Form factor: start with plain Python functions and small classes inside `ai/openai.py` (or a sibling `agents/` module later) to keep iteration fast.
- Metadata cache: a simple dict `{ database: { tables: [...], columns: {table:[...] } } }` populated at startup. Reuse for DQ prompts and schema answers.
- Tool calls: via `langchain_mcp_adapters` HTTP client to `MCP_URL`. Reuse a single session, 60–120s timeout per call.
- Prompting: role-specific system prompts with explicit read-only policy and DQ output schema guidelines.
- Responses: concise user-facing answers; include small samples when helpful (via `base_tablePreview`), avoid large JSON dumps.

## What LangChain Adds (and When)

- Pros:
  - Tool abstraction with schema exposure to the LLM (reduces glue code).
  - Agent executors, routers, and memory primitives we can adopt later.
  - Built-in retry, tracing, and callbacks.
- Cons:
  - Additional framework complexity; our current MCP usage is already through `langchain-mcp-adapters` for the client side.
- MVP choice: keep role logic as functions; optionally wrap into a simple LangChain agent later (particularly once SQL/BI flows are introduced).

## Can CrewAI Help?

- Strengths: multi-agent orchestration, role definitions, task delegation, shared memory, and planning.
- Use cases here: coordinating SQL Expert and BI/DQ experts with clear handoffs and artifacts (SQL, KPI spec, viz spec).
- Tradeoffs: new dependency and learning curve; can overcomplicate MVP.
- Recommendation: Future phase. Once single-process routing stabilizes and SQL/BI are added, we can model specialists as CrewAI agents with explicit tasks and shared context.

## Configuration

- `config/.env`:
  - `OPENAI_API_KEY`, `OPENAI_MODEL` (e.g., `gpt-4o`, `gpt-4o-mini`).
  - `MCP_URL` (e.g., `http://localhost:8001/mcp/`).
  - `MCP_SPAWN` (0/1) for dev-only spawn.
  - `DATABASE_URI` (for spawn mode only).
  - `MCP_TIMEOUT` per tool call (60–120s typical).
  - `TD_NAME` default database when user doesn’t specify.

## Factory Integration

- Add `ai/openai.py` defining `AI_OpenAI` (subclass of `AI`).
- Update `ai/factory.py` to support `AI_BACKEND=openai`.
- Keep `ai/gpt.py` as the LLM-only backend.

## Prompting Strategy (Role-Specific)

- Data Analyst: greet, set expectations, route, keep answers concise; inject the boundary message when off-topic. For SQL/BI requests, acknowledge and state they’re coming soon.
- Data Quality: summarize stats with 2–4 bullet takeaways and next-step suggestions; request missing parameters (db/table/column) if not provided.

## Error Handling

- LLM/API errors: retry with backoff (similar to `ai.gpt`).
- MCP connection errors: surface clear messages; in spawn mode, attempt start and show tail on failure.
- Tool timeouts: provide partials if possible and guidance to narrow scope.
- Enforce read-only policy in prompts; validate SQL strings before execution (no keywords like DROP/DELETE/UPDATE/INSERT/ALTER).

## Telemetry & Logging

- Events: `ai_openai.init`, `ai_openai.call`, `ai_openai.tool.call`, `ai_openai.tool.error`.
- Log prefixes:
  - `[LLM=>MCP]` tool name and payload (sanitize; include SQL if present; mask secrets).
  - `[MCP=>LLM]` response payload (truncate very large outputs, e.g., >8–16 KB).

## MVP Roadmap

Phase 1 (this sprint)
- Implement Data Analyst + Data Quality Expert + metadata cache.
- Enable router to handle schema/discovery and data quality flows.
- Use `qlty_*` tools (+optional `base_tablePreview`), no SQL/BI yet.

Phase 2
- Add SQL Expert and BI Expert; introduce `base_readQuery` aggregates and KPI flows.

Phase 3
- Consider CrewAI for explicit multi-agent orchestration.
- Optional LangChain Agent wrappers and richer memory.

## Open Items

1. Confirm dev-only spawn vs prod URL-only (default assumed).
2. Confirm logging truncation limit for `[MCP=>LLM]` (e.g., 8–16 KB).
3. Confirm `TD_NAME` as the canonical env var for default DB.

## MVP Scope

- One-shot replies (no token streaming to UI initially).
- Read-only tools.
- URL-based MCP by default; optional dev spawn.
- LLM-led intent detection with lightweight keyword fallback.
