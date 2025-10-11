from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

ENV_PATH = BASE_DIR / ".env"

SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"

ASSETS_PATH = BASE_DIR / "assets"

TERADATA_LOGO_PATH = ASSETS_PATH / "td_new_trans.png"

CHARTS_PATH = BASE_DIR / "charts"