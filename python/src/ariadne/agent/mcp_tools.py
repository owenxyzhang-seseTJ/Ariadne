"""Optional MCP bridge. CLI/numerical kernels do not require an MCP runtime."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from . import engine


class Reply(BaseModel):
    data: dict[str, Any] = Field(description='Structured agent state or operation result')


def register(mcp):
    @mcp.tool()
    def agent_capabilities() -> Reply:
        return Reply(data=engine.capabilities())

    @mcp.tool()
    def agent_init(request: dict[str, Any], session: str) -> Reply:
        return Reply(data=engine.initialize(request, session))

    @mcp.tool()
    def agent_step(session: str) -> Reply:
        """Run at most one bounded scientific action; call again after inspecting state."""
        return Reply(data=engine.run(session, max_steps=1))

    @mcp.tool()
    def agent_status(session: str) -> Reply:
        return Reply(data=engine.status(session))

    @mcp.tool()
    def agent_cancel(session: str) -> Reply:
        return Reply(data=engine.cancel(session))

    @mcp.tool()
    def agent_resume(session: str) -> Reply:
        return Reply(data=engine.resume(session))

    @mcp.tool()
    def agent_retry(session: str, node: str) -> Reply:
        return Reply(data=engine.retry(session, node))

    @mcp.tool()
    def agent_export(session: str, destination: str, include_inputs: bool = False) -> Reply:
        return Reply(data=engine.export_session(session, destination, include_inputs=include_inputs))
