#!/usr/bin/env python3
"""
Teradata MCP Server Startup Script

This script provides a convenient way to start the Teradata MCP server
with various transport modes and profiles from the command line.

Usage:
    python scripts/start_mcp_server.py [options]

Examples:
    # Start with default settings (stdio mode)
    python scripts/start_mcp_server.py

    # Start HTTP server on port 8001
    python scripts/start_mcp_server.py --http --port 8001

    # Start with specific profile
    python scripts/start_mcp_server.py --http --profile dba

    # Start with custom database URI
    python scripts/start_mcp_server.py --http --database-uri "teradata://user:pass@host:1025/db"

    # Start with debug logging
    python scripts/start_mcp_server.py --http --debug
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import time
import urllib.request
import urllib.error


def load_env_file():
    """Load environment variables from config/.env if it exists."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / "config" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment from {env_path}")
        else:
            print(f"ℹ No .env file found at {env_path}")
    except ImportError:
        print("ℹ python-dotenv not available; using system environment only")


def check_database_uri() -> Optional[str]:
    """Check if DATABASE_URI is available and validate basic format."""
    database_uri = os.getenv("DATABASE_URI")
    if not database_uri:
        print("⚠ WARNING: DATABASE_URI not set in environment or .env file")
        print("  The MCP server requires a Teradata connection string like:")
        print("  export DATABASE_URI='teradata://username:password@host:1025/database'")
        return None
    
    if not database_uri.startswith("teradata://"):
        print(f"⚠ WARNING: DATABASE_URI should start with 'teradata://', got: {database_uri[:20]}...")
    
    return database_uri


def find_mcp_server_command() -> Optional[str]:
    """Find the teradata-mcp-server command in the system."""
    commands_to_try = [
        "teradata-mcp-server",  # If installed globally
        "uvx teradata-mcp-server",  # Using uvx
        "python -m teradata_mcp_server",  # Module execution
    ]
    
    for cmd in commands_to_try:
        try:
            # Test if command is available
            result = subprocess.run(
                f"{cmd} --help", 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ Found MCP server command: {cmd}")
                return cmd
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            continue
    
    print("✗ Could not find teradata-mcp-server command")
    print("  Try installing with: pip install teradata-mcp-server")
    print("  Or using uvx: uvx teradata-mcp-server --help")
    return None


def build_command(args) -> List[str]:
    """Build the command line arguments for the MCP server."""
    base_cmd = find_mcp_server_command()
    if not base_cmd:
        return []
    
    # Split command if it contains spaces (e.g., "uvx teradata-mcp-server")
    cmd_parts = base_cmd.split()
    
    # Add transport mode
    if args.http:
        cmd_parts.extend(["--mcp_transport", "streamable-http"])
        cmd_parts.extend(["--mcp_port", str(args.port)])
        if args.host:
            cmd_parts.extend(["--mcp_host", args.host])
    elif args.sse:
        cmd_parts.extend(["--mcp_transport", "sse"])
        cmd_parts.extend(["--mcp_port", str(args.port)])
        if args.host:
            cmd_parts.extend(["--mcp_host", args.host])
    # stdio is default, no need to specify
    
    # Add profile if specified
    if args.profile:
        cmd_parts.extend(["--profile", args.profile])
    
    # Add database URI if specified
    if args.database_uri:
        cmd_parts.extend(["--database_uri", args.database_uri])
    
    # Add logging level
    if args.debug:
        cmd_parts.extend(["--logging_level", "DEBUG"])
    elif args.verbose:
        cmd_parts.extend(["--logging_level", "INFO"])
    
    return cmd_parts


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start Teradata MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start with stdio (default)
  %(prog)s --http                            # Start HTTP server on port 8001  
  %(prog)s --http --port 8002                # Start HTTP server on custom port
  %(prog)s --http --profile dba              # Start with DBA tools profile
  %(prog)s --sse --port 8003                 # Start SSE server
  %(prog)s --debug                           # Enable debug logging
        """
    )
    
    # Transport mode options
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--http", 
        action="store_true", 
        help="Use streamable-http transport mode (for web clients)"
    )
    transport_group.add_argument(
        "--sse", 
        action="store_true", 
        help="Use server-sent events transport mode"
    )
    
    # Server configuration
    parser.add_argument(
        "--port", 
        type=int, 
        default=8001, 
        help="Port for HTTP/SSE server (default: 8001)"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host for HTTP/SSE server (default: localhost)"
    )
    parser.add_argument(
        "--profile", 
        choices=["all", "tester", "dba", "dataScientist", "base"],
        help="MCP tools profile to load"
    )
    parser.add_argument(
        "--database-uri", 
        help="Teradata database connection URI (overrides environment)"
    )
    
    # Logging options
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    logging_group.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable info logging"
    )

    # Health check options (only meaningful for HTTP/SSE) - default enabled
    parser.add_argument(
        "--no-health-check", dest="health_check", action="store_false", default=True,
        help="Disable post-start health check ping (default: enabled)"
    )
    parser.add_argument(
        "--health-timeout", type=int, default=20,
        help="Health check overall timeout seconds (default: 20)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load environment
    load_env_file()

    # If no transport flag provided, fall back to environment MCP_TRANSPORT
    if not args.http and not args.sse:
        mcp_transport_env = os.getenv("MCP_TRANSPORT", "").strip().lower()
        if mcp_transport_env == "http":
            args.http = True
            print("ℹ Using HTTP transport from MCP_TRANSPORT env")
        elif mcp_transport_env == "sse":
            args.sse = True
            print("ℹ Using SSE transport from MCP_TRANSPORT env")
        elif mcp_transport_env == "stdio":
            # Explicit stdio declaration; nothing to set
            print("ℹ Using stdio transport from MCP_TRANSPORT env")
        elif mcp_transport_env:
            print(f"⚠ Unrecognized MCP_TRANSPORT value '{mcp_transport_env}', defaulting to stdio")

    # Allow env overrides for host/port only if user did not specify flags
    # (Keep CLI precedence highest.)
    if (args.http or args.sse):
        # Only override if user kept defaults
        if args.host == parser.get_default('host'):
            env_host = os.getenv("MCP_HOST")
            if env_host:
                args.host = env_host
                print("ℹ Host overridden from MCP_HOST env")
        if args.port == parser.get_default('port'):
            env_port = os.getenv("MCP_PORT")
            if env_port and env_port.isdigit():
                args.port = int(env_port)
                print("ℹ Port overridden from MCP_PORT env")
            elif env_port:
                print(f"⚠ Ignoring non-numeric MCP_PORT value '{env_port}'")
    
    # Check database connection
    if not args.database_uri:
        database_uri = check_database_uri()
        if not database_uri:
            print("\n✗ Cannot start MCP server without DATABASE_URI")
            print("  Set DATABASE_URI in environment or use --database-uri option")
            return 1
    
    # Build command
    cmd_parts = build_command(args)
    if not cmd_parts:
        return 1
    
    # Display startup info
    print(f"\n🚀 Starting Teradata MCP Server...")
    print(f"   Command: {' '.join(cmd_parts)}")
    
    if args.http or args.sse:
        transport = "streamable-http" if args.http else "sse"
        server_url = f"http://{args.host}:{args.port}/mcp/"
        print(f"   Transport: {transport}")
        print(f"   Server URL: {server_url}")
        # Extra explicit line for tooling / copy-paste convenience
        print(f"   Copy: MCP_SERVER_URL={server_url}")
    else:
        print(f"   Transport: stdio (default)")
    
    if args.profile:
        print(f"   Profile: {args.profile}")
    
    print(f"   Database: {os.getenv('DATABASE_URI', 'from --database-uri')[:50]}...")
    print()
    
    try:
        # If HTTP/SSE with health check enabled, start asynchronously and poll
        if (args.http or args.sse) and args.health_check:
            print("   Health: performing startup probe...")
            process = subprocess.Popen(cmd_parts)
            # If binding to 0.0.0.0 / ::, probe via localhost to avoid WinError 10049
            probe_host = args.host
            if probe_host in {"0.0.0.0", "::"}:
                probe_host = "127.0.0.1"
                print(f"   Health: substituting probe host '{args.host}' -> '{probe_host}'")
            base_http = f"http://{probe_host}:{args.port}"
            base_paths = ["/mcp/health", "/mcp/", "/health", "/"]
            probe_urls = [base_http + p for p in base_paths]
            deadline = time.time() + args.health_timeout
            last_error = None
            success = False
            attempt = 0
            headers = {"Accept": "text/event-stream,application/json;q=0.9,*/*;q=0.1"}
            opener = urllib.request.build_opener()
            while time.time() < deadline and process.poll() is None and not success:
                for probe_url in probe_urls:
                    attempt += 1
                    req = urllib.request.Request(probe_url, headers=headers, method="GET")
                    try:
                        with opener.open(req, timeout=3) as resp:
                            code = resp.getcode()
                            if 200 <= code < 500:
                                print(f"   Health: OK ({code}) at {probe_url} [attempt {attempt}]")
                                success = True
                                break
                    except urllib.error.HTTPError as he:
                        last_error = he
                        # Accept HTTPError codes below 500 as readiness (e.g., 404 Not Found acceptable)
                        if 400 <= he.code < 500:
                            print(f"   Health: OK ({he.code}) at {probe_url} [attempt {attempt}] (acceptable HTTPError)")
                            success = True
                            break
                    except urllib.error.URLError as ue:
                        last_error = ue
                if not success:
                    time.sleep(0.75)
            if not success:
                if process.poll() is not None:
                    print("✗ Server process exited prematurely during health check")
                    return process.returncode or 1
                print(f"⚠ Health check did not confirm readiness within {args.health_timeout}s after {attempt} attempts")
                if last_error:
                    print(f"   Last error: {last_error}")
                    # Show failing probe URLs once for diagnostics
                    print("   Probed URLs:")
                    for u in probe_urls:
                        print(f"     - {u}")
            else:
                print("   Health: server ready")
            print("\n⏳ Press Ctrl+C to stop the server.")
            process.wait()
            return process.returncode or 0
        else:
            # Simple blocking run
            result = subprocess.run(cmd_parts, check=False)
            return result.returncode
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
