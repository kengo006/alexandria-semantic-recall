# Changelog

This repository is a reference implementation extracted from a live production
instance; releases are curated snapshots.

## v0.5 — 2026-09-28 · Every category, by name

The degradation self-report was behind the contract it implements. alexandria's
registry has had four categories since its v3.6, and the contract asks the index to
report each by name; this server still read the older two-key form and returned one
count — the shape that contract warns against, because the categories prescribe
different, sometimes opposite, actions. alexandria v4.0 said so in its own
`optional-integrations.md`. The upstream instance had made this change on 2026-08-23;
it missed the v0.4 cut, and the comparison meant to catch such gaps started its count
from the wrong date.

- 🔴 **Breaking: `degraded_files` and the flat `non_citable` count are gone.** The
  `degraded` block now carries `counts` and `meaning`, each keyed by category. A
  caller that read `degraded_files` now gets nothing instead of a number that meant
  less than it looked.
- **Categories are found by structure, not by name**: any top-level array, or an
  object with a `files` array. A category this server has never heard of is still
  reported, and listed in `categories_without_meaning`. Upstream, a reader that knew
  one key said nothing through two new categories; a list of known names would do
  the same the day a fifth appears.
- **Each category's instruction**: the registry's own wording first (`meaning` on an
  object-shaped category, or a top-level `<category>_meaning` beside an array), then
  this server's line for the four documented categories.
- **Declared counts are checked per category** (`counts`, v0.4's `count`, or a
  `count` on the category itself); a disagreement goes into `count_mismatch`, with
  the measured value in `counts`.
- **Still `"unknown"`, never `0`** — and now also when the file is not an object, is
  not UTF-8, or holds no category at all.
- The v0.4 two-key form (`files`, `non_citable`) is still read, and marked in
  `registry_form`.
- `_test_degraded.py` (new): thirteen known-answer cases, one of them the example
  registry this repository ships. Fault-injected twice: with categories read from a
  list of known names, the two unknown-category cases fail; with the empty-registry
  guard removed, that case fails. A case whose check raises is now a failed case, not
  a crashed test.
- `examples/degraded.json` moves to the four-category form; the README section is
  rewritten, and both of its JSON examples were compared against the shipped file and
  the server's actual output.

## v0.4 — 2026-08-11 · What the two backends disagreed about

A line-by-line read of every file in this repository, prompted by nothing going
wrong. Four defects, none of which any mechanical scan here had reported.

- 🔴 **The two backends did not hold the same corpus.** Working files (basename
  starting with `_`) were excluded by `migrate_to_lance.py` and
  `incremental_update.py` but **not** by `build_index.py` — so the numpy index
  carried chunks the LanceDB table had dropped, while the README stated that "the
  `search(query,k)` API is identical on both". The API was; the corpus was not.
  The rule now lives once, in `index_lib.is_workfile()`, and all three callers use
  it. **It had no test** — which is how it drifted — and now it has one
  (`_test_incremental.py` step E, asserting both directions).
- 🔴 **The character-chunking fallback was silent.** `index_lib` falls back to
  character windows when `tokenizers` is unavailable — the exact path its own
  docstring documents as losing over half of every CJK chunk. For a tool whose
  headline is cross-lingual recall, sliding into that quietly is worse than
  stopping: it now prints a warning naming the cause. And `tokenizers` is
  **declared** rather than relied on transitively through `fastembed`.
- 🔴 **`pyproject.toml` still said `version = "0.1.0"`** — two releases stale.
- **`search_core`'s nprobes comment** claimed "100/512 partitions". `num_partitions`
  is `min(512, max(16, sqrt(N)))`, so that ratio holds only past ~262k chunks; the
  comment stated one instance's configuration as a general fact.
- README Status now points at this file.

## v0.3 — 2026-08-02 · Deep recall, and an honest "unknown"

Two additions, both optional, both off unless you turn them on.

*(Documentation follow-up, same day: the registry's schema is now written down
with a runnable `examples/degraded.json`, the deep-recall section carries four
measured queries — including the signal we expected to find and did not, and the
structural limit that reranking widens the ordering but never the net — and download
sizes are stated in one unit throughout.)*

- **New tool `semantic_search_deep`** — recall wide, widen each hit into its
  surrounding passage, rerank with a cross-encoder, return the best few. It exists
  for one situation: the passage you need is in the corpus but sits below the
  plain-recall cutoff. It is **not** a better default — when plain recall is
  already precise, reranking can scatter it just as easily as it can help.
  Chunks are sized for recall, not for judgement, which is why the passage is
  widened before the reranker ever sees it.
  ⚠ **`score` is a reranker logit, not a cosine.** It orders results and nothing
  else: no threshold makes it "relevant", and scores from different queries are not
  comparable. The original cosine is returned alongside it as `recall_score` so the
  two views stay separable. Reranker is lazy-loaded and configurable via
  `SEMANTIC_RECALL_RERANKER`; requires the LanceDB backend.
- **New `degraded_status()`, surfaced through `semantic_search_info`** — reads an
  optional degradation registry (`SEMANTIC_RECALL_DEGRADED`) listing corpus files
  whose text layer is known-unreliable, so a caller can tell when a *miss* means
  nothing.
  🔑 **It returns `"unknown"`, never `0`, when the registry cannot be read.** To a
  caller deciding whether a search miss is evidence, "zero degraded files" and
  "I could not read the registry" are opposite statements — the first says a
  negative result is trustworthy, the second says nothing at all. Rounding a failed
  read down to zero makes the caller assert exactly where it should have held back.
  A declared count that disagrees with the actual length is reported as
  `count_mismatch` rather than silently resolved.
  ⚠ **The default reranker pulls ~2.3 GB on first use** - ten times the embedding
  model this project fetches by default - and that number is now stated in the
  tool docstring, the README and beside the constant rather than left to be
  discovered mid-query. Provenance is disclosed with it: the upstream model is
  Apache-2.0 and heavily used, while the ONNX conversion repository declares no
  licence of its own and its organisation is not HF-verified. Weights and a
  tokenizer only; no executable code.
  `_ensure_reranker_external_data()` now swallows **only** "this repository has no
  external-data file". A bare except there would have cancelled the function's
  reason for existing - a timeout or a full disk would surface exactly where the
  call was meant to prevent it.

  A second key, `non_citable`, is reported separately and deliberately **not**
  merged into the degraded count: `files` governs whether a *negative* conclusion
  is trustworthy, `non_citable` governs whether a *positive* quote may be used.
  Folding them together inflates the first with unrelated reasons.

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
