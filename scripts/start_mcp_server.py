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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load environment
    load_env_file()
    
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
        print(f"   Transport: {transport}")
        print(f"   Server URL: http://{args.host}:{args.port}/mcp/")
    else:
        print(f"   Transport: stdio (default)")
    
    if args.profile:
        print(f"   Profile: {args.profile}")
    
    print(f"   Database: {os.getenv('DATABASE_URI', 'from --database-uri')[:50]}...")
    print()
    
    try:
        # Start the server
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