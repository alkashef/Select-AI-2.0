from pathlib import Path

BASE_DIR = Path(__file__).parents[1]

CONFIG_PATH = BASE_DIR / "config"

ENV_PATH = CONFIG_PATH / ".env"

PLOT_AGENT_SYSTEM_PROMPT_PATH = CONFIG_PATH / "plot_agent_system_prompt.txt"

TERADATA_AGENT_SYSTEM_PROMPT_PATH = CONFIG_PATH / "teradata_agent_system_prompt.txt"

MANAGER_AGENT_SYSTEM_PROMPT_PATH = CONFIG_PATH / "manager_agent_system_prompt.txt"

ASSETS_PATH = BASE_DIR / "assets"

TERADATA_LOGO_PATH = ASSETS_PATH / "td_new_trans.png"

LANGGRAPH_GRPAH_IMAGE_PATH = ASSETS_PATH / "langgraph_graph.png"

CHARTS_PATH = BASE_DIR / "charts"