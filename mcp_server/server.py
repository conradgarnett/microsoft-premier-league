"""MCP server over the shared match tools.

    python -m mcp_server                 # stdio (Agent Framework MCPStdioTool, Copilot, Claude, ...)
    python -m mcp_server --http 8001     # Streamable HTTP at http://localhost:8001/mcp
"""

from __future__ import annotations

import functools
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agents.tools import MatchTools
from api.store import MatchRepo, make_store


def _clean_errors(fn):
    """Unknown ids are caller errors: report them as tool errors instead of crashes."""

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except (KeyError, ValueError, StopIteration) as e:
            raise ToolError(f"not found or invalid argument: {e}") from e

    return wrapper


def build_server(tools: MatchTools | None = None) -> MCPServer:
    tools = tools or MatchTools(MatchRepo(make_store()))
    server = MCPServer(
        name="tactical-replay",
        instructions="Synthetic football match data. Every result carries event ids (evN) - cite them as evidence.",
    )

    @server.tool()
    @_clean_errors
    def list_matches() -> list[str]:
        """Ids of the matches available in the store."""
        return tools.repo.store.list_matches()

    @server.tool()
    @_clean_errors
    def get_match_summary(match_id: str) -> dict[str, Any]:
        """Teams, final score, goals and team metrics (possession, xG, PPDA, field tilt, defensive line)."""
        return tools.get_match_summary(match_id)

    @server.tool()
    @_clean_errors
    def get_events(match_id: str, start_t: float = 0, end_t: float = 1e9, types: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Match events between start_t and end_t seconds. types filters e.g. ["pass","shot"]. Max 200."""
        return tools.get_events(match_id, start_t, end_t, types, limit)

    @server.tool()
    @_clean_errors
    def get_key_moments(match_id: str, top: int = 10) -> list[dict[str, Any]]:
        """Key moments (goals, big chances, saves) ranked by importance, with xG and evidence ids."""
        return tools.get_key_moments(match_id, top)

    @server.tool()
    @_clean_errors
    def get_player_stats(match_id: str, player_id: str) -> dict[str, Any]:
        """One player's passing, shooting, defensive and physical (distance, sprints, top speed) stats."""
        return tools.get_player_stats(match_id, player_id)

    @server.tool()
    @_clean_errors
    def get_skeleton_window(match_id: str, window_id: str, player_id: str | None = None, stride: int = 5) -> dict[str, Any]:
        """Biomechanics for a replay window: strike angles, scanning, and optionally one player's keypoints + joint angles."""
        return tools.get_skeleton_window(match_id, window_id, player_id, stride)

    @server.tool()
    @_clean_errors
    def get_pressure_map(match_id: str, team_id: str | None = None) -> dict[str, Any]:
        """Grid counts of pressure events per team (each team attacking to the right)."""
        return tools.get_pressure_map(match_id, team_id)

    @server.tool()
    @_clean_errors
    def get_data_quality(match_id: str) -> dict[str, Any]:
        """Data provenance: synthetic or video, player/ball coverage, how many events were detected from video."""
        return tools.get_data_quality(match_id)

    return server
