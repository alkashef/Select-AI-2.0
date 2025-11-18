# Select AI

Select AI is a Streamlit-based AI assistant for business intelligence that lets business users and executives ask natural-language questions against enterprise data and receive instant answers and visualizations. It connects to Teradata Vantage via a local MCP (Model Context Protocol) server and supports interactive chart generation, test tooling, and pre-built agent backends.

## Why this project

- Natural-language access to enterprise data (replaces many static dashboards)
- Pluggable AI backends and agent implementations
- Tight integration with Teradata Vantage using the Teradata MCP server
- Tools for starting the MCP server and running protocol-based integration tests

## Quick links

- Source: `src/` (main app in `src/app.py`)
- Start MCP server helper: `scripts/start_mcp_server.py`
- Integration test runner: `tests/run_mcp_tests.py`
- Dependencies: `requirements.txt`

## Architecture diagram

Below is a flow diagram showing how user input is processed end-to-end.

![Dashboard](assets/langgraph_graph.png)

## Getting started

These instructions assume you have a working Conda installation on Windows (PowerShell). Adjust commands for other platforms or shells.

1. Clone the repository

2. Create and activate a Conda environment and install dependencies

```powershell
conda create --name next-bi python=3.12 -y
conda activate next-bi
pip install -r requirements.txt
```

3. Required environment variables

- `TD_HOST` — Hostname or IP address for the Teradata server (example: `mcp-db.example.com`).
- `TD_NAME` — Database name/schema to connect to (example: `BANK_DB`).
- `TD_USER` — Username for Teradata authentication (example: `demo_user`).
- `TD_PASSWORD` — Password for Teradata authentication (do NOT commit this value).
- `DATABASE_URI` — Teradata connection string (required by MCP server and tests). Example:

  `teradata://username:password@host:1025/database`

- `OPENAI_API_KEY` (or other model provider keys) — if you intend to use OpenAI-based backends.
- Optional MCP-related env vars used by helper scripts: `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`

Create a `config/.env` file if you prefer storing local environment settings; `scripts/start_mcp_server.py` attempts to load `config/.env` automatically.

4. Run the Streamlit app (UI)

```powershell
streamlit run src/app.py
```

The app will initialize an event loop and try to create the multi-agent AI backend. If the AI backend or MCP connection is not configured, the app will show an initialization error and a refresh will be required after fixing env vars.

5. Start the MCP server (helper script)

Use the convenience script to start a Teradata MCP server process. Example (HTTP transport, custom port and profile):

```powershell
python scripts/start_mcp_server.py --http --port 8001 --profile dba --database-uri "teradata://user:pass@host:1025/BANK_DB"
```

If you want stdio transport (default for testing), omit `--http`/`--sse` and ensure `DATABASE_URI` is present in your environment.

6. Run integration tests against a running MCP server

The test runner starts a server (stdio mode) and runs cases in `tests/core_test_cases.json` (or other files).

```powershell
python tests/run_mcp_tests.py "uv run teradata-mcp-server --mcp_transport stdio --mcp_port 1025 --profile all --database_uri teradata://user:pass@host:1025/BANK_DB" tests/core_test_cases.json
```

Notes:
- The test runner requires `DATABASE_URI` and network access to the Teradata instance used for tests.
- Use `--verbose` with the test runner to see more output.

## Project structure

- `src/` — Streamlit app and application logic (main entry: `src/app.py`)
- `scripts/` — helper scripts (MCP server startup, utilities)
- `tests/` — test runners and test case JSON files
- `config/` — environment and configuration samples (e.g., `.env`)
- `requirements.txt` — Python dependencies

## Agents

This project uses a small multi-agent architecture. Each agent has a focused responsibility:

- Manager Agent (`src/agents/manager_agent.py`)
  - Orchestrates the conversation and decides which tool/agent should handle a user request.
  - Reads a system prompt (`config/manager_agent_system_prompt.txt`) and combines user input plus intermediate responses to make a JSON-style decision (e.g. `teradata`, `plot`, or `done`).
  - Routes work to the Teradata or Plot agents and appends manager messages to the session history.

- Teradata Agent (`src/agents/teradata_agent.py`)
  - Integrates with the Teradata MCP server via `mcp_use` and exposes Teradata tools to the agent via a LangChain adapter.
  - Receives manager requests and runs database queries, returning textual results or structured data that the manager will use.
  - System prompt is templated with database and charts paths (see `config/teradata_agent_system_prompt.txt`).

- Plot Agent (`src/agents/plot_agent.py`)
  - Generates visualizations by executing Python code snippets via a Python REPL tool (`PythonAstREPLTool`) so plots can be created and saved under `charts/`.
  - Uses a template system prompt (`config/plot_agent_system_prompt.txt`) to instruct chart generation and where to write images.
  - When it runs, it sets `is_plot=True` in the multi-agent state so the Streamlit UI will display the generated chart image.

### Short contract (inputs / outputs)

- Input: multi-agent state with `user_query` and optional partial responses (`td_agent_response`, `plot_agent_response`).
- Output: updated multi-agent state containing `response`, `explanation`, `manager_decision`, and, for plots, `is_plot` and `plot_agent_response`.
- Errors: agents append error text to the state and the app surface shows initialization or runtime errors.

### Edge cases handled

- Missing `DATABASE_URI` or MCP connectivity: `scripts/start_mcp_server.py` and test runners validate environment and exit with helpful errors.
- Plot file cleanup: the Streamlit app attempts to load the newest image from `charts/` and will try to delete temporary files after loading.

## Troubleshooting & Where to get help

- Check runtime logs in the `logs/` directory for recent runs.
- Create an issue in the repository (GitHub Issues) with reproduction steps, environment, and relevant logs.
- Use the `scripts/start_mcp_server.py` health-check and `--debug` flags to surface server startup problems.

## Security and secrets

- Do not commit secrets (API keys, passwords) to the repository. Use `config/.env` (added to `.gitignore`) or your CI secret storage.
- The repository uses environment variables for database and AI provider credentials.
