"""RAG config helper for teradata-mcp-server.

This script helps you find where the MCP server expects `rag_config.yml`,
show the active config, and optionally install our project template into the
installed package location.

Usage (Windows cmd):
  python scripts\rag_config.py --show-path
  python scripts\rag_config.py --show-config
  python scripts\rag_config.py --install-from config\rag_config.yml

Notes:
 - The MCP server loads the config relative to its package path:
   src/teradata_mcp_server/config/rag_config.yml
 - This script locates the `teradata_mcp_server` package and operates
   on the `config/rag_config.yml` file in that installation.
 - Requires write permissions to site-packages to install/overwrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil

try:
    import teradata_mcp_server  # type: ignore
except Exception as e:
    print("teradata-mcp-server is not installed in this environment.")
    print("Install it first: pip install teradata-mcp-server")
    raise SystemExit(1)


def get_package_config_path() -> Path:
    pkg_path = Path(teradata_mcp_server.__file__).resolve().parent
    # package root/src/.../teradata_mcp_server
    config_path = pkg_path / "config" / "rag_config.yml"
    return config_path


def show_path() -> None:
    cfg_path = get_package_config_path()
    print(str(cfg_path))


def show_config() -> int:
    cfg_path = get_package_config_path()
    if not cfg_path.exists():
        print(f"Config not found at: {cfg_path}")
        return 1
    print(cfg_path.read_text(encoding="utf-8"))
    return 0


def install_from(src: Path) -> int:
    if not src.exists():
        print(f"Source not found: {src}")
        return 1
    dst = get_package_config_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Installed RAG config -> {dst}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage teradata-mcp-server RAG config")
    parser.add_argument("--show-path", action="store_true", help="Print path to active rag_config.yml in package")
    parser.add_argument("--show-config", action="store_true", help="Print contents of active rag_config.yml")
    parser.add_argument("--install-from", type=str, default=None, help="Path to project rag_config.yml to install into package")
    args = parser.parse_args()

    if args.show_path:
        show_path()
        return 0
    if args.show_config:
        return show_config()
    if args.install_from:
        return install_from(Path(args.install_from))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
