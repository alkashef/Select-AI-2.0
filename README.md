# Select AI 2.0
Natural Language to SQL Chat UI with pluggable AI backend.

## Overview

Select AI is a natural language to SQL assistant that enables users to query databases using plain English. The app provides a Streamlit chat UI and a simple AI backend interface so you can swap models without changing the UI. The default backend uses OpenAI Chat Completions and reads config from `config/.env`.

Key features:

- Natural language to SQL translation (model-dependent)
- Streamlit chat interface with lightweight styling
- Pluggable AI backend via `ai/base.py` + `ai/factory.py`
- Optional logging to `log.txt` controlled by `config/.env`

## Project Status

- Default AI backend: OpenAI Chat Completions via `openai>=1.30`.
- Integration tests: gated by `RUN_REAL_OPENAI_TESTS=1` in `config/.env`.
- Database: manual validation script under `tests/test_db.py` using `teradataml` and `config/.env` TD_* variables.
- New: optional Teradata MCP Server support added. We use a separate terminal/server (HTTP) flow with small client scripts under `tests/`.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.11+
- **AI Backend**: OpenAI Chat Completions (via `openai>=1.30`) – default
- **Testing**: pytest – unit + optional integration
- **Configuration**: python-dotenv – `config/.env`

## Important Files

- Core Application: `app.py` – Streamlit UI and chat loop
- AI Interface: `ai/base.py` – abstract contract `AI`
- AI Factory: `ai/factory.py` – selects and constructs the backend
- OpenAI Backend: `ai/gpt.py` – `AI_GPT` implementation
- Configuration: `config.py`, `config/.env`, `config/.env.example`
- Logging: `logger.py` (writes to `log.txt` when enabled)

## Setup for Development

#### Conda Environment

1. Create/activate a Conda env (or use your preferred venv):

   ```cmd
   C:\Users\<you>\AppData\Local\anaconda3\Scripts\activate.bat select-ai
   python -m pip install -r requirements.txt
   ```

GPU-specific setup is not required for the default OpenAI backend.

### Mac GPUs (Apple Silicon or Metal-compatible Intel)
1. Ensure PyTorch 2.0+ is installed:
   ```bash
   pip install --upgrade torch
   ```

When using other local models, add guidance as needed in a separate section.

If your app needs a database, document its setup in a dedicated section.

## How to Test

Run tests to verify the application components are working correctly:

Run unit tests:

```cmd
pytest -q
```

Run the real OpenAI integration test (optional, costs may apply):

```cmd
set RUN_REAL_OPENAI_TESTS=1
pytest -q tests\test_ai_gpt.py
```

DB connectivity check without MCP (manual script):

```cmd
python tests\test_db.py
```

Requirements:
- Set `TD_HOST`, `TD_NAME`, `TD_USER`, `TD_PASSWORD` in `config/.env` (or environment).
- Optional: `TD_PORT` (default `1025`) and `TD_TEST_QUERY` to override the sample query.
- The script prints a schema snapshot and query results. It’s a plain script, not a pytest test.

## How to Run

The application can be run in two modes:

Web UI (Streamlit):

```cmd
python -m streamlit run app.py
```

Batch/CLI modes are not provided in this minimal UI. Add separate scripts as needed under `scripts/`.

## Teradata MCP Server (Optional)

We use a separate terminal/server (HTTP) flow. Start the MCP server in one terminal, then run client scripts from another.

1) Install the server and adapters (into your active env):

```cmd
pip install --upgrade pip
pip install teradata-mcp-server langchain-mcp-adapters
teradata-mcp-server --version
```

2) In a new terminal, set DB env and start the server in HTTP mode:

```cmd
set TD_USER=your_user
set TD_PASSWORD=your_password
set TD_HOST=your_host
set TD_NAME=your_database
set TD_PORT=1025
set DATABASE_URI=teradata://%TD_USER%:%TD_PASSWORD%@%TD_HOST%:%TD_PORT%/%TD_NAME%
teradata-mcp-server --mcp_transport streamable-http --mcp_port 8001 --profile all
```

You should see a line like: http://localhost:8001/mcp/

3) In your project terminal, configure the client URL in `config/.env` (or set at runtime):

```cmd
set MCP_URL=http://localhost:8001/mcp/
```

4) Run the HTTP client smoke tests:

List tools:
```cmd
python tests\test_mcp_list_tools.py
```

Diagnostics (tools/prompts/resources):
```cmd
python tests\test_mcp_diag.py
```

Run a SQL via base_readQuery:
```cmd
python tests\test_mcp_read_query.py --sql "SELECT CURRENT_DATE"
```

Notes:
- Server must run in a separate terminal with `DATABASE_URI` set there. The clients only need `MCP_URL`.
- If your password contains '&', ensure you quote/escape when constructing `DATABASE_URI`.
- For production, consider `LOGMECH`, `ENCRYPTDATA=ON`, and `CHARSET=UTF8` connection options.

Troubleshooting:
- If the client times out, verify the server terminal shows requests and that `MCP_URL` matches.
- Confirm DB host/port connectivity from the server machine.
