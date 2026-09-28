# alexandria-semantic-recall

**Local semantic recall for [alexandria](https://github.com/kengo006/alexandria) — the runnable implementation of the reference recipe in its [optional-integrations §2](https://github.com/kengo006/alexandria/blob/main/optional-integrations.md).**

<!-- zh-intro -->

[中文簡介](#中文簡介)

<!-- /zh-intro -->

If you want general-purpose semantic search over an Obsidian vault, use [Smart Connections](https://smartconnections.app/) — it is mature, zero-setup, and built for exactly that. This project exists for a narrower job: serving a **citation-integrity workflow**, where semantic search is allowed to *find* passages but never to *quote* them. Every fragment it returns carries a file path and page numbers so a downstream agent (or you) can walk back to the source PDF and verify. Recall-only, by design and by contract.

- **100% local**: ONNX embeddings via [fastembed](https://github.com/qdrant/fastembed) (no PyTorch), vectors in [LanceDB](https://github.com/lancedb/lancedb) on disk. No API calls, no cost, works offline.
- **Cross-lingual**: the default multilingual model lets a query in one language recall passages in another — the one thing keyword grep can never do.
- **Modest hardware is enough**: the upstream production instance indexes over 450,000 chunks from more than 500 source texts on a 16 GB laptop, CPU-only; incremental updates run in about 90 seconds.
- **MCP, not just CLI**: read-only subagents often have no shell. As an MCP server this attaches `semantic_search` directly to their toolchain.

<!-- zh-intro -->

## 中文簡介

**[alexandria](https://github.com/kengo006/alexandria) 的本地語意召回：它在 [optional-integrations §2](https://github.com/kengo006/alexandria/blob/main/optional-integrations.md) 所寫的參考做法的可執行實作。**

如果你要的是對 Obsidian 知識庫做通用的語意搜尋，請用 [Smart Connections](https://smartconnections.app/)：它成熟、免設定，本來就是為此而做。這個專案只做一件更窄的事：服務一套**引文完整性**的工作流程；在這套流程裡，語意搜尋可以*找到*段落，但永遠不能*引用*它們。它回傳的每個片段都附有檔案路徑與頁碼，讓下游的代理（或你自己）能回到原始 PDF 核對。只負責召回，既是設計，也是約定。

- **100% 本地**：透過 [fastembed](https://github.com/qdrant/fastembed) 做 ONNX 嵌入（不需要 PyTorch），向量以 [LanceDB](https://github.com/lancedb/lancedb) 存在磁碟上。不呼叫任何 API、不花錢、可離線運作。
- **跨語言**：預設的多語模型能讓一種語言的查詢召回另一種語言的段落，這是關鍵字 grep 永遠做不到的事。
- **一般硬體就夠**：上游的正式執行個體在一台 16 GB 筆電上只用 CPU，索引了來自 500 多份原始文本的 45 萬多個片段；增量更新大約 90 秒。
- **是 MCP，不只是 CLI**：唯讀的子代理常常沒有 shell。做成 MCP 伺服器，就能把 `semantic_search` 直接接進它們的工具鏈。

<!-- /zh-intro -->

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

## Optional: deep recall, and knowing when a miss means nothing

Two capabilities that stay off unless configured.

**`semantic_search_deep`** recalls wide, widens each hit into its surrounding
passage, reranks with a cross-encoder, and returns the best few. Reach for it
deliberately: it is for the case where the passage you need is in the corpus but
sits below the plain-recall cutoff. When plain recall is already precise,
reranking can scatter it.

**What it does — four measured queries.** On a production corpus of a few hundred
scholarly texts, four queries fixed before the run — two question-shaped, one
term-shaped, one keyword-shaped — each recalled thirty candidates and were
reranked. In **all four**, plain recall's top hit was not the reranker's first
choice: the passage the reranker put first had been sitting at plain rank 6, 8, 5
and 16, and its top three drew on passages as deep as rank 24. Cosine spread
across the thirty ranged from **0.0404 to 0.1251**, a factor of three; the
reranker's logits spread across 7.6 to 9.7.

🔑 **The signal we expected to find, and did not.** The natural hypothesis is that
bunched cosines are that signal: if the top thirty sit inside a few hundredths,
the ordering among them is close to arbitrary and there is something to fix; if
the top hit stands clearly apart, there is not. **These four do not support it.**
The tightest spread did displace plain rank 1 the furthest, to rank 25 — but the
widest spread, whose top two stood apart from the pack by sixteen times the
median gap, still pushed plain rank 1 down to rank 13, while the two middling
spreads moved it only to rank 6. So there is no cheap pre-check on offer here:
whether reranking earns its download is not visible in the recall scores.

**Who that changes things for.** The reranker's first choice came from plain ranks
6, 8, 5 and 16 — in four out of four, never from the plain top three. So if your
workflow reads the first few recall hits and stops, this replaced the answer every
time; if you already read all thirty, it mostly reorders what you were going to read
anyway. And note the hard edge, which is structural rather than measured: reranking
only ever reorders the recall window. A passage that plain recall put at rank 45 is
exactly as invisible to `semantic_search_deep` as it was to `semantic_search` — this
widens the *ordering*, never the *net*.

⚠ **Not measured: whether the new order is better.** Reading the passages, the
reranked top hits were on-question where the top cosine hits were adjacent to the
topic — but that is a reading of four queries, not a benchmark.

⚠ **The first call downloads ~2.3 GB** — the ONNX graph plus its external weight
file — and is slow; later calls reuse the cached model. That is ten times the
0.22 GB embedding model above, which is why it is stated here rather than left to
be discovered mid-query. Set `SEMANTIC_RECALL_RERANKER` to a smaller cross-encoder
if that is too much. Requires the LanceDB backend.

**Provenance of the default** (verified 2026-08-02, re-check before relying on it):
the default is an ONNX conversion of
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3),
which is Apache-2.0 and very widely used. The conversion repository it is
actually fetched from —
[`onnx-community/bge-reranker-v2-m3-ONNX`](https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX)
— **declares no licence of its own**, and its organisation is not HF-verified.
What ships is weights, a tokenizer and configs — no executable code. None of that
is disqualifying, and none of it should be discovered by a reader after the
download has started.

If the undeclared licence is a blocker in your setting, the closest alternative
that is both multilingual and licensed is
[`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) (MIT,
about half the size, published by the upstream organisation itself). The
English-only cross-encoders bundled with fastembed are smaller still, but this
project's whole point is cross-lingual recall, so they are not a like-for-like
swap.

⚠ **The `score` it returns is a reranker logit, not a cosine.** It orders results
and nothing else — there is no value at which it becomes "relevant", and scores
from different queries are not comparable. The original cosine comes back as
`recall_score`.

**The degradation registry.** Point `SEMANTIC_RECALL_DEGRADED` at a JSON file
listing corpus files whose text layer you know to be unreliable, and
`semantic_search_info` will report it. The design decision worth copying even if
you never use this file:

> When the registry cannot be read, the report says **`"unknown"` — never `0`.**

To anyone deciding whether a search *miss* is evidence, "zero degraded files" and
"I could not read the registry" are opposite claims. The first says a negative
result is trustworthy; the second says nothing at all. Defaulting a failed read to
zero makes the caller assert precisely where it should have held back.

A second key, `non_citable`, is reported separately and is deliberately not added
to the degraded count. They answer different questions: `files` governs whether a
*negative* conclusion is trustworthy; `non_citable` governs whether a *positive*
quote may be used at all. Merging them inflates the first with unrelated reasons.

**The file it expects** (a runnable copy is at `examples/degraded.json`):

```json
{
  "generated": "2026-08-02T12:00:00Z",
  "count": 2,
  "files": [
    "examples/corpus/monuments_and_memory.txt",
    "examples/corpus/oral_history_methods.txt"
  ],
  "non_citable": [
    "examples/corpus/oral_history_methods.txt"
  ]
}
```

| key | required | meaning |
|---|---|---|
| `files` | **yes** | array of paths whose text layer is unreliable — grep on these can return zero for material that is present. Anything else makes the report `"unknown"`. |
| `count` | no | your generator's own count. If it disagrees with the length of `files`, the report says so in `count_mismatch` instead of quietly picking one. |
| `non_citable` | no | array (or an object with a `files` array) of paths that are recallable but have no page-bearing original to quote from. |
| `generated` | no | passed through untouched, so a consumer can see how stale the registry is. |

Paths are yours to interpret: this server reports counts and shape, it does not
resolve them against your corpus.

## Design notes

- **Chunks are sized in tokens (≤120, sentence-aligned), using the embedding model's own tokenizer** — not in characters. The char↔token ratio varies ~3× across languages (Latin prose ≈3.9 chars/token, CJK ≈1.4), so the v0.1 500-character windows silently overflowed the model's 128-token truncation on CJK text: **over half of every CJK chunk never entered the index**, while Western text lost only ~3% — invisible unless you test cross-lingually, which is exactly this tool's headline claim. Measured: embedding a full chunk vs. only its first 128 tokens returns cosine 1.0000. The size floor is in tokens too (a character floor filters CJK chunks out). Chunks stay small because the consumer wants precise pointers, not summaries; page numbers are computed from character offsets against the page marks. Character chunking survives only as a fallback when the `tokenizers` package is unavailable.
- **Default model** `paraphrase-multilingual-MiniLM-L12-v2` (0.22 GB, 384-dim, CPU): genuinely cross-lingual. In an upstream head-to-head against a larger 8192-token model, recall@5 tied — the small model stays because it is ~1.8× faster, half the vector size, and safe on 16 GB RAM. Swap via `--model` if your needs differ; longer-context models need smaller embedding batches.
- **Embedding batch defaults to 32**: measured on a 16 GB machine, larger batches with long-sequence models can spike RAM to the ceiling. Tune upward only with monitoring.
- **fastembed is version-pinned**: a silent pooling change in a future version would make new query vectors inconsistent with an existing index. Unpin deliberately, and rebuild when you do. (On load you may see fastembed's own warning that this model "now uses mean pooling" — it is informational; the pin keeps query-side and index-side pooling consistent, which is what matters.)
- **Dual backend**: LanceDB serves (text + vectors on disk, top-k reads, resident RAM roughly constant — migrating saved ~640 MB at 410k chunks); numpy remains a fallback (delete `lance_db/` to fall back). The `search(query,k)` API is identical on both.
- **Incremental ids** start at 500,000,000 so they never collide with full-build ids; the query side never reads ids.

## Status

Extracted from a production system (2026) where it serves a six-role academic writing workflow — see [alexandria](https://github.com/kengo006/alexandria) for the architecture it plugs into. Release history, and the reasoning behind each change, is in [CHANGELOG.md](CHANGELOG.md). Issues and adaptations welcome. [MIT](LICENSE).
