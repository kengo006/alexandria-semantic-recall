"""MCP server (stdio): exposes local semantic recall as a tool that read-only
subagents can call.

Why MCP and not a CLI: in the alexandria role architecture the Searcher subagent
runs with Read/Grep/Glob only — no shell — so it cannot invoke a CLI. An MCP tool
attaches to the subagent's toolchain and is callable directly. The same applies to
any read-only agent setup.

Start:  uv run python server.py
Wire-up: register this server in your MCP config (.mcp.json or equivalent) so your
search agent inherits the `semantic_search` tool. See README "Wiring it into agents".
"""
from __future__ import annotations

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_core import search, info  # noqa: E402

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("semantic-search")


@mcp.tool()
def semantic_search(query: str, k: int = 8) -> list[dict]:
    """Semantic recall: retrieve text-layer passages by *meaning*, catching what
    keyword grep misses (paraphrase and cross-lingual matches).

    Returns top-k fragments, each with file (relative path), page_start/page_end
    (for going back to the source PDF), score (cosine; higher = closer), and text
    (the fragment itself).
    ⚠ Recall layer only: verbatim quotes, page numbers, and emphasis must be
    verified against the source PDF. Never cite these fragments directly.
    """
    return search(query, k=k)


@mcp.tool()
def semantic_search_info() -> dict:
    """Report current index state: backend, model, chunk count, file count, build time."""
    return info()


if __name__ == "__main__":
    mcp.run()
