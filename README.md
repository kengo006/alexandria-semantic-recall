# alexandria-semantic-recall

**Local semantic recall for [alexandria](https://github.com/kengo006/alexandria) — the runnable implementation of the reference recipe in its [optional-integrations §2](https://github.com/kengo006/alexandria/blob/main/optional-integrations.md).**

If you want general-purpose semantic search over an Obsidian vault, use [Smart Connections](https://smartconnections.app/) — it is mature, zero-setup, and built for exactly that. This project exists for a narrower job: serving a **citation-integrity workflow**, where semantic search is allowed to *find* passages but never to *quote* them. Every fragment it returns carries a file path and page numbers so a downstream agent (or you) can walk back to the source PDF and verify. Recall-only, by design and by contract.

- **100% local**: ONNX embeddings via [fastembed](https://github.com/qdrant/fastembed) (no PyTorch), vectors in [LanceDB](https://github.com/lancedb/lancedb) on disk. No API calls, no cost, works offline.
- **Cross-lingual**: the default multilingual model lets a query in one language recall passages in another — the one thing keyword grep can never do.
- **Modest hardware is enough**: the upstream production instance indexes over 450,000 chunks from more than 500 source texts on a 16 GB laptop, CPU-only; incremental updates run in about 90 seconds.
- **MCP, not just CLI**: read-only subagents often have no shell. As an MCP server this attaches `semantic_search` directly to their toolchain.

## The contract

This implements the interface alexandria documents for its Searcher role:

```
search(query, k) -> [{file, page_start, page_end, score, text}, ...]
```

- Fragments are **pointers**: `file` is relative to your corpus root, pages map back to the source PDF.
- **Recall-only**: fragments are never citation sources. Quotes, page numbers, and emphasis are verified against the PDF.
- **Degrades gracefully**: agents treat semantic recall as a bonus; grep remains the backstop (a stale index means *semantic silence never proves absence*).

## Quickstart (5 minutes, synthetic example included)

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone https://github.com/kengo006/alexandria-semantic-recall
cd alexandria-semantic-recall

# 1) build an index over the bundled synthetic corpus (first run downloads the model, ~0.22 GB)
uv run python build_index.py "examples/corpus"

# 2) query it from the CLI — note the paraphrase match: no keyword overlap required
uv run python search_core.py "can spoken testimony be trusted as evidence" -k 3

# 3) start the MCP server (stdio)
uv run python server.py
```

## Your own corpus

The input is a **text layer**: a folder tree of `.txt` files, one per source PDF, with page boundaries preserved in either of two formats:

- born-digital extraction (e.g. `pdftotext -layout`): form-feed `\f` page breaks
- scanned/OCR sources: explicit markers `===== page N =====`

How you produce the text layer is up to you — alexandria's optional-integrations §1 describes the conventions. Then:

```bash
set CORPUS_ROOT=C:\path\to\your\text-layer     # or export on unix; or pass the path as an argument
uv run python build_index.py                   # numpy index (index/)
uv run python migrate_to_lance.py              # -> LanceDB (lance_db/), the serving backend
uv run python build_lance_index.py             # ANN index for fast queries
```

Or all three steps as one guarded unit:

```bash
uv run python rebuild.py
```

`rebuild.py` exists because of a real incident: the server prefers `lance_db/`, so refreshing only the numpy index silently serves stale data. The three steps are bound together and verified at the end.

### Keeping it fresh

After adding, changing, moving, or deleting corpus files:

```bash
uv run python incremental_update.py --dry-run   # see the delta first
uv run python incremental_update.py             # embed only what changed (~90s vs hours for a full rebuild)
```

Not real-time — run it after ingestion as a maintenance habit. The citation layer is the second safety net: quotes are verified against PDFs regardless of index state.

## Wiring it into agents

Register the server in your MCP config (`.mcp.json` or your client's equivalent):

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/alexandria-semantic-recall",
               "python", "/path/to/alexandria-semantic-recall/server.py"]
    }
  }
}
```

Then give your search agent one standing instruction — this is the discipline that makes the tool safe:

> Use `semantic_search` to *discover* candidate passages, then go back to the source PDF (via the returned file + pages) and read the verbatim text there. Fragments locate; only the source speaks.

alexandria's Searcher role ships with this wiring already described (search path D).

## Design notes

- **Chunks are small (~500 chars, sentence-aligned)** because the default model's max_seq is ~128 tokens, and because the consumer wants precise pointers, not summaries. Page numbers are computed from character offsets against the page marks.
- **Default model** `paraphrase-multilingual-MiniLM-L12-v2` (0.22 GB, 384-dim, CPU): genuinely cross-lingual. In an upstream head-to-head against a larger 8192-token model, recall@5 tied — the small model stays because it is ~1.8× faster, half the vector size, and safe on 16 GB RAM. Swap via `--model` if your needs differ; longer-context models need smaller embedding batches.
- **Embedding batch defaults to 32**: measured on a 16 GB machine, larger batches with long-sequence models can spike RAM to the ceiling. Tune upward only with monitoring.
- **fastembed is version-pinned**: a silent pooling change in a future version would make new query vectors inconsistent with an existing index. Unpin deliberately, and rebuild when you do. (On load you may see fastembed's own warning that this model "now uses mean pooling" — it is informational; the pin keeps query-side and index-side pooling consistent, which is what matters.)
- **Dual backend**: LanceDB serves (text + vectors on disk, top-k reads, resident RAM roughly constant — migrating saved ~640 MB at 410k chunks); numpy remains a fallback (delete `lance_db/` to fall back). The `search(query,k)` API is identical on both.
- **Incremental ids** start at 500,000,000 so they never collide with full-build ids; the query side never reads ids.

## Status

Extracted from a production system (2026) where it serves a five-role academic writing workflow — see [alexandria](https://github.com/kengo006/alexandria) for the architecture it plugs into. Issues and adaptations welcome. [MIT](LICENSE).
