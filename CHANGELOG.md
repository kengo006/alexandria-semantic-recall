# Changelog

This repository is a reference implementation extracted from a live production
instance; releases are curated snapshots.

## v0.2 — 2026-07-20 · Token-based chunking (correctness fix)

- **`index_lib.py`: chunks are now sized in tokens (≤120), using the embedding
  model's own tokenizer.** The v0.1 character windows (~500 chars) silently
  overflowed the model's 128-token truncation on CJK text — over half of every
  CJK chunk never entered the index, defeating the cross-lingual headline claim.
  Western text lost only ~3%, which is why the defect stayed invisible.
  The minimum-size floor moved to tokens for the same reason (a character floor
  filters CJK chunks out). Character chunking remains only as a fallback when
  the `tokenizers` package is unavailable.
- **Rebuild required** to benefit: existing indexes keep working, but chunks
  embedded under v0.1 carry the truncation. `python rebuild.py <corpus>` once.
- README design notes updated to match; this file added (per-release summaries
  are now maintained).

## v0.1 — 2026-07-14 · Initial public release

- Local, CPU-only semantic recall over a folder of page-marked .txt files:
  fastembed (ONNX) + LanceDB, cross-lingual multilingual MiniLM by default.
- `build_index.py` / `migrate_to_lance.py` / `build_lance_index.py` /
  `rebuild.py` one-shot pipeline; `incremental_update.py` delta updates
  (~90 s instead of a full rebuild); `search_core.py` + MCP `server.py`
  exposing `semantic_search` / `semantic_search_info` to read-only agents.
- Version-pinned fastembed; dual backend (LanceDB serving, numpy fallback);
  synthetic example corpus. 16 files.
