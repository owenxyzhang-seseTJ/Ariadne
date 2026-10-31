"""Optional MCP transport using the official Python SDK, no custom protocol fork."""
from __future__ import annotations
import argparse
from typing import Any

from .agent import AgentWorkspace, TOOLS
from .capabilities import capabilities


def create_server(root, legacy_executables=None):
    try:
        # Official v2 server; v1 remains supported for existing installations.
        try:
            from mcp.server import MCPServer
        except ImportError:
            from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError as error:
        raise RuntimeError('Install the optional transport with pip install ".[mcp]"') from error
    workspace = AgentWorkspace(root, legacy_executables)
    server = MCPServer("Ariadne-Xtal")

    @server.tool()
    def crystal_capabilities() -> dict[str, Any]:
        """Return exact native capability boundaries and the available action names."""
        return {"capabilities": capabilities(), "tools": TOOLS}

    @server.tool()
    def crystal_execute(request: dict[str, Any]) -> dict[str, Any]:
        """Execute a declared crystallographic action and return its traceable result.

        Input files must be workspace-relative. Actions: inspect, refine,
        solve (P1; known composition), maps, absolute, legacy. Each scientific
        operation writes a new run directory. No Python/shell execution tool.
        """
        return workspace.dispatch(request)

    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ariadne-Xtal MCP stdio server")
    parser.add_argument("--root", default=".")
    parser.add_argument("--shelxt")
    parser.add_argument("--shelxl")
    args = parser.parse_args(argv)
    legacy = {key: getattr(args, key) for key in ("shelxt", "shelxl") if getattr(args, key)}
    create_server(args.root, legacy).run(transport="stdio")


if __name__ == "__main__":
    main()
