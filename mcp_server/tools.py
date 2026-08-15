"""
Tool definitions.

Two generic stubs showing the shape a tool takes. They are registered onto the
server in `server.py` via `register_tools()`.

To add a tool: write the function here, decorate it inside `register_tools`,
done. Nothing in `server.py` needs to change.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Tool response models ──────────────────────────────────────────────


class EchoResult(BaseModel):
    """What `echo` returns."""

    message: str = Field(..., description="The message that was sent back.")
    length: int = Field(..., description="Character count of the message.")


class AddResult(BaseModel):
    """What `add` returns."""

    a: float
    b: float
    total: float = Field(..., description="a + b")


# ── Registration ──────────────────────────────────────────────────────


def register_tools(mcp: FastMCP) -> FastMCP:
    """
    Attach every tool in this module to `mcp`.

    Takes the server as an argument instead of importing it, so that
    `server.py` stays the single place that owns the server instance and there
    is no circular import between the two files.
    """

    @mcp.tool()
    def echo(message: str) -> EchoResult:
        """
        Echo a message back.

        This docstring is what the model reads when deciding whether to call
        this tool, so in a real tool it should describe *when to use it*, not
        how it works.
        """
        logger.info("echo(%r)", message)
        return EchoResult(message=message, length=len(message))

    @mcp.tool()
    def add(a: float, b: float) -> AddResult:
        """
        Add two numbers together.

        Second stub, included to show that parameter types are read straight off
        the annotations — FastMCP builds the tool's JSON schema from them.
        """
        logger.info("add(%s, %s)", a, b)
        return AddResult(a=a, b=b, total=a + b)

    return mcp
