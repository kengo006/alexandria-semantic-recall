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
from search_core import search, info, deep_search, degraded_status  # noqa: E402

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
def semantic_search_deep(query: str, k: int = 8) -> list[dict]:
    """Deep recall: retrieve wide, widen each hit into its surrounding passage,
    then rerank with a cross-encoder and return the best few.

    **Use this deliberately, not by default.** It is for one situation: the passage
    you need is in the corpus but sits below the plain-recall cutoff - the twentieth
    hit rather than the third. When plain recall is already precise,
    reranking can scatter it just as easily as it can help.

    ⚠ **score here is a reranker logit, not a cosine.** It orders the results and
    nothing else: there is no value at which it becomes "relevant", and scores from
    different queries are not comparable. recall_score keeps the original cosine
    alongside it.
    ⚠ Recall layer only, exactly as with semantic_search: verbatim quotes, page
    numbers and emphasis are verified against the source PDF, never cited from here.

    ⚠ **First call downloads ~2.13 GiB** (the ONNX graph plus its external weight
    file) and is slow; later calls reuse the cached model. That is ten times the
    embedding model this project fetches by default - worth knowing before you
    invoke this rather than after. Requires the LanceDB backend. Point
    SEMANTIC_RECALL_RERANKER at a smaller cross-encoder if that is too much.
    """
    return deep_search(query, top_k=k)


@mcp.tool()
def semantic_search_info() -> dict:
    """Report current index state: backend, model, chunk count, file count, build time,
    and the degradation registry's self-report.

    ⚠ Read the degradation block before treating any *miss* as evidence. status
    "unknown" means the registry could not be read - which is not the same as
    "no degraded files", and must not be rounded down to zero.
    """
    out = info()
    out["degraded"] = degraded_status()
    return out


if __name__ == "__main__":
    mcp.run()
