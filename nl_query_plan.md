# Natural Language → SQL → Answer (MCP) — Test Plan

Goal
- Send a natural-language question to the MCP server and let the server:
  1) Interpret the NL question
  2) Generate SQL (server-side)
  3) Execute against the configured DB
  4) Return an answer (and optionally the generated SQL and evidence)

Constraints
- Tool-only: No client-side SQL crafting or direct DB access
- HTTP transport: Use the MCP HTTP endpoint
- Config-driven: Use YAML for URL, DB defaults, and options when possible

Assumptions (to verify in docs)
- The server exposes either:
  - A specialized NL→SQL tool (e.g., `nl_query`, `sql_generate_and_run`, etc.), or
  - A workflow executor (e.g., `rag_executeWorkflow`) configured for NL→SQL, or
  - A prompt-based interface (e.g., `_testMyServer`) that chains tools internally.
- Typical response format includes a `results` payload and possibly metadata like `sql` or `explanation`.

High-Level Flow
1) Load environment and YAML
   - load `config/.env` for defaults
   - read a YAML file (e.g., `tests/mcp_schema_sample.yml` or a new `tests/mcp_nl.yml`) for:
     - `mcp_url`: MCP HTTP endpoint
     - `question`: natural language query
     - optional flags: `return_sql`, `max_rows`, `timeout`
2) Connect to MCP server (HTTP)
   - create `MultiServerMCPClient` with `transport = streamable_http`
   - open session for "mcp_server"
   - discover tools via `load_mcp_tools(session)`
3) Select NL→SQL mechanism (server-owned)
   - Prefer a dedicated tool if present, e.g.: `nl_query`, `text_to_sql`, or similar
   - Else, use a workflow tool if present, e.g.: `rag_executeWorkflow` with `question`
   - Else, use a server-defined prompt (e.g., `_testMyServer`) if it supports instructing the server to translate & run
   - If none are available, exit with a clear message
4) Invoke the tool/workflow
   - Build arguments strictly as defined by the server (no SQL on client)
   - Required: `question` (the NL string)
   - Optional (if supported): `database_name`, `limit_rows`, `return_sql`
   - Call `tool.ainvoke(args)` with a reasonable timeout
5) Parse and print results
   - Pretty-print the returned payload
   - If present, show `answer`, `rows`, and `generated_sql` (for transparency)
   - Do not assume schema; rely on server’s structure (use generic normalization)
6) Exit code policy
   - 0 on success (valid response object)
   - 1 on missing tools, invocation error, or unparsable response

Pseudocode
```
load_env_from_config()
cfg = load_yaml_config(nl_yaml_path)
url = resolve_mcp_url(cli_url, cfg)
question = cli_question or cfg.get("question")
if not question:
    print("Provide NL question via --question or YAML")
    exit(2)

client = MultiServerMCPClient({
  "mcp_server": {"url": url, "transport": "streamable_http"}
})
async with client.session("mcp_server") as session:
    tools = await load_mcp_tools(session)
    tools_by_name = {t.name: t for t in tools}

    # NL→SQL selection (prefer dedicated, then workflow, then prompt)
    nl_tool = tools_by_name.get("nl_query") or tools_by_name.get("text_to_sql")
    workflow_tool = tools_by_name.get("rag_executeWorkflow")

    if nl_tool:
        args = {"question": question}
        # Optionally pass database_name, return_sql, limit_rows if supported
        result = await asyncio.wait_for(nl_tool.ainvoke(args), timeout)
    elif workflow_tool:
        args = {"question": question}
        # Optionally include k or workflow-specific params if documented
        result = await asyncio.wait_for(workflow_tool.ainvoke(args), timeout)
    else:
        # As a last resort, try a prompt-based path if the server supports it
        print("No NL→SQL tool/workflow available on MCP server")
        exit(1)

    # Normalize and print
    print_mcp_result(result)
    exit(0)
```

Validation
- Smoke test with a simple question like "How many accounts were opened last month?"
- Ensure server returns either:
  - a direct numeric answer, and/or
  - a table snippet with aggregated result, and
  - optionally the generated SQL (for transparency-only)

Notes
- Keep the client SQL-ignorant at all times
- If the server lacks NL→SQL support, skip with a clear message rather than failing other tests
- If needed, create `tests/mcp_nl.yml` to store `mcp_url`, default `question`, and knobs like `limit_rows`
