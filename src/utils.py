from pathlib import Path
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory

import os
from typing import Optional

from modules.config import Config
from multi_agents import MultiAgent
from agents import TeradataAgent, ManagerAgent, PlotAgent

def get_openai_config(base_dir: Optional[Path] = None) -> dict:
    """Load OpenAI settings and return a ready client + settings.

    Parameters
    ----------
    base_dir:
        Optional base directory to isolate .env loading (tests may pass a
        temporary directory). When provided the loader reads `config/.env`
        from that directory instead of the process environment.

    Returns
    -------
    dict
        Mapping with keys {"api_key", "model", "client"}. The
        ``client`` is an instantiated OpenAI client object.

    Raises
    ------
    RuntimeError
        If the required OPENAI_API_KEY is not set in the environment.
    """
    # Ensure .env is loaded and exists (reuses Config side-effect to load)
    Config.load(base_dir=base_dir)

    # Import locally to avoid hard dependency for non-OpenAI flows
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or config/.env")

    model = os.getenv("GPT_MODEL", "gpt-4o").strip() or "gpt-4o"

    # Optional timeout to avoid hanging calls
    try:
        timeout = int(os.getenv("OPENAI_TIMEOUT", "20").strip())
    except ValueError:
        timeout = 20

    # Optional advanced settings
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    organization = os.getenv("OPENAI_ORG", "").strip() or None
    project = os.getenv("OPENAI_PROJECT", "").strip() or None

    client_kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url
    if organization:
        client_kwargs["organization"] = organization
    if project:
        client_kwargs["project"] = project

    client = OpenAI(**client_kwargs)
    return {"api_key": api_key, "model": model, "client": client}

def get_google_genai_config(base_dir: Optional[Path] = None) -> dict:
    """Load Google (Generative AI) settings and return a ready client.

    Parameters
    ----------
    base_dir:
        Optional base directory used to locate `config/.env` for isolated
        runs. If None the process environment is consulted.

    Returns
    -------
    dict
        Mapping with keys {"api_key", "model", "client"} where
        ``client`` is an instantiated client object.

    Raises
    ------
    RuntimeError
        If the required GOOGLE_API_KEY is not present in the environment.
    """
    # Ensure .env is loaded and exists (reuses Config side-effect to load)
    Config.load(base_dir=base_dir)

    # Import locally to avoid hard dependency for non-OpenAI flows
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or config/.env")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    # Optional timeout to avoid hanging calls
    try:
        timeout = int(os.getenv("GOOGLE_TIMEOUT", "20").strip())
    except ValueError:
        timeout = 20

    client_kwargs = {"api_key": api_key, "timeout": timeout}

    client = OpenAI(**client_kwargs)
    return {"api_key": api_key, "model": model, "client": client}

def get_ai_backend(base_dir: Optional[Path] = None) -> str:
    """Return the configured AI backend name.

    Parameters
    ----------
    base_dir:
        When provided, the function reads `config/.env` from this
        directory and ignores the global process environment. Useful
        for isolated tests.

    Returns
    -------
    str
        Lower-cased backend name (e.g. 'gpt' or 'gemini'). Defaults to 'gpt'.

    Raises
    ------
    FileNotFoundError
        If ``base_dir`` is provided but the expected .env file is missing.
    """
    if base_dir is not None:
        env_path = base_dir / "config" / ".env"
        if not env_path.exists():
            raise FileNotFoundError(f"Config file not found: {env_path}")
        values = dotenv_values(dotenv_path=env_path)
        val = (values.get("AI_BACKEND", "") or "").strip().lower()
        return val or "gpt"

    # Default behavior: consult the current process environment only.
    # Other components (e.g., logger, OpenAI config) will load .env as needed.
    return os.getenv("AI_BACKEND", "gpt").strip().lower() or "gpt"

async def get_multi_agent() -> MultiAgent:
    """Construct and return a configured MultiAgent instance.

    The function chooses an LLM implementation based on the configured
    backend and initializes the three agents (Teradata, Plot, Manager)
    with a shared conversational memory.

    Returns
    -------
    MultiAgent
        A fully constructed MultiAgent ready to be used by the application.

    Raises
    ------
    ValueError
        If an unknown AI backend is configured.
    """
    backend = get_ai_backend()

    llm = None
    if backend == "gpt":
        cfg = get_openai_config()
        llm = ChatOpenAI(
            model=cfg["model"],
            api_key=cfg["api_key"]
        )
    elif backend == "gemini":
        cfg = get_google_genai_config()
        llm = ChatGoogleGenerativeAI(
            model=cfg["model"],
            api_key=cfg["api_key"]
        )
    else:
        raise ValueError(f"Unknown AI backend: {backend}")

    memory = ConversationBufferWindowMemory(
        k=15,
        output_key="output",
        return_messages=True,
        memory_key="chat_history",
    )
    teradata_agent = await TeradataAgent.create(llm, memory)
    plot_agent = await PlotAgent.create(llm, memory)
    manager_agent = await ManagerAgent.create(llm, memory)

    multi_agent = MultiAgent(memory, manager_agent, plot_agent, teradata_agent)

    return multi_agent