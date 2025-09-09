## Pinkman

A lightweight Streamlit chat UI wired to pluggable AI backends. Fast to spin up, easy to tweak, and great for demos or prototyping agent logic.

## What you get

- Minimal chat interface (Streamlit)
- AI abstraction layer with a factory selector
- OpenAI GPT backend with retries and timeouts
- Simple UTC file logger for chats and events
- Tests (optional live integration)

## Project layout

- `app.py` – Streamlit UI that routes messages to the AI backend
- `ai/` – AI abstraction and implementations
   - `base.py` – Abstract `AI` contract
   - `factory.py` – `get_ai()` selects backend from env
   - `gpt.py` – OpenAI GPT backend (chat completions)
- `config.py` – Loads `config/.env`, exposes settings and OpenAI client
- `logger.py` – Append-only logger with UTC timestamps
- `assets/styles.css` – Chat bubble styles
- `tests/test_ai_gpt.py` – Opt-in integration test for real OpenAI client
- `scripts/cleanup.py` – Repo cleanup tool (caches, logs, prunes empties)

## Setup

1) Python 3.11+ recommended. Create a virtual env and install deps:

    - Windows (cmd):
       - py -m venv .venv
       - .venv\Scripts\activate
       - pip install -r requirements.txt

2) Create env file:

    - Copy `config/.env.example` to `config/.env` and fill in:
       - OPENAI_API_KEY
       - GPT_MODEL (e.g., gpt-4o or gpt-4o-mini)
       - Optional: OPENAI_TIMEOUT, OPENAI_BASE_URL, OPENAI_ORG, OPENAI_PROJECT
       - Logging: LOG_ENABLED=true, LOG_FILE=log.txt

## Run the app

- streamlit run app.py

The chat appears in your browser. Type a message and the AI replies. Errors render as an AI bubble so the flow isn’t broken.

## Running tests

- Unit/integration (real client, opt-in):
   - Set environment variables before running:
      - RUN_REAL_OPENAI_TESTS=1
      - OPENAI_API_KEY and GPT_MODEL
   - Then: pytest -q

If env vars aren’t set, the integration test is skipped.

## Cleanup script

The cleanup tool removes caches and optional ignored files, and can prune empty directories.

- python scripts/cleanup.py -y --gitignore --prune-empty

Flags:
- -y/--yes: don’t prompt
- -g/--gitignore: also remove items matched by .gitignore patterns
- --prune-empty: remove now-empty directories
- -v/--verbose: show actions

## Notes

- `config/.env` is ignored by Git. Never commit secrets. Use `config/.env.example` for reference.
- `AI_BACKEND` defaults to `gpt`. Extend the factory to add more backends.
- Logging is off unless LOG_ENABLED=true.
