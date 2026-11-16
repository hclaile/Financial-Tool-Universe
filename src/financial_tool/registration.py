"""Register every public financial tool with one FastMCP manager."""

from mcp.server.fastmcp import FastMCP

from .registry import ToolRegistry
from .auth import AuthMiddleware
from .cache import CacheManager
from .rate_limit import RateLimiter


def create_tool_server() -> FastMCP:
    """Build the shared registry used by the public MCP server."""
    mcp = FastMCP("financial-tool-universe")
    registry = ToolRegistry.from_catalog()
    registry.register_all(mcp)

    AuthMiddleware.attach(mcp)
    CacheManager.attach(mcp)
    RateLimiter.attach(mcp)
    return mcp
