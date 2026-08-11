"""Index loading + querying. Dual backend, shared by the CLI and the MCP server.

Backend detection: if `lance_db/` exists -> LanceDB (text + vectors stay on disk,
queries fetch only top-k, resident RAM does not grow with corpus size);
otherwise -> numpy (embeddings.npy mmap + chunks.jsonl fully loaded; fallback).

The API `search(query, k)` and its return shape (score/file/page_start/page_end/text)
are backend-independent and stable — this is the contract alexandria's
optional-integrations §2 documents.

CLI:  uv run python search_core.py "how commemoration shapes collective memory" -k 8
"""
from __future__ import annotations

import sys, os, json, argparse, functools

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.environ.get("SEMANTIC_RECALL_INDEX", os.path.join(HERE, "index"))
LANCE_DIR = os.environ.get("SEMANTIC_RECALL_LANCE", os.path.join(HERE, "lance_db"))
MODEL_CACHE = os.path.join(HERE, "model_cache")  # fixed cache dir: %TEMP% gets wiped by disk cleanup

# Optional deep-dive reranker: a cross-encoder, ONNX only (no torch in this project).
# Loaded lazily - nothing is downloaded unless semantic_search_deep is actually called.
#
# ⚠ The default pulls ~2.3 GB on first use (2,288,835,892 bytes measured
# 2026-08-02: the ONNX graph plus its external weight file). That is ten times
# the embedding model this project downloads by default, so it is stated here
# rather than left to be discovered.
# Provenance, verified 2026-08-02: the upstream model (BAAI/bge-reranker-v2-m3) is
# Apache-2.0 and heavily used; the ONNX conversion repository below declares **no
# licence of its own** and its organisation is not an HF-verified one. Weights and
# a tokenizer only - no executable code ships with it. Set SEMANTIC_RECALL_RERANKER
# to point elsewhere if any of that is unacceptable in your setting.
RERANKER = os.environ.get("SEMANTIC_RECALL_RERANKER", "onnx-community/bge-reranker-v2-m3-ONNX")

# Optional degradation registry: a JSON file listing corpus files whose text layer is
# known-unreliable. Point at your own with SEMANTIC_RECALL_DEGRADED. Absent = "unknown",
# never "none" - see degraded_status().
DEGRADED_REGISTRY = os.environ.get("SEMANTIC_RECALL_DEGRADED", "")


def _use_lance() -> bool:
    return os.path.isdir(LANCE_DIR)


def info() -> dict:
    """Index metadata (model/count/...), backend-independent (reads the active backend's meta.json).
    Deliberately uncached: incremental updates to meta become visible in the same
    session (pairs with _lance_table being uncached)."""
    p = os.path.join(LANCE_DIR if _use_lance() else INDEX_DIR, "meta.json")
    return json.load(open(p, encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _model(name: str):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=name, cache_dir=MODEL_CACHE)


@functools.lru_cache(maxsize=1)
def _lance_db():
    import lancedb
    return lancedb.connect(LANCE_DIR)


def _lance_table():
    # Table handle is NOT cached: each open sees the latest committed version, so
    # incremental updates become searchable in the same session without a restart.
    # The connection IS cached (connect is the expensive part; open_table only reads
    # the manifest and is cheap).
    return _lance_db().open_table("chunks")


@functools.lru_cache(maxsize=1)
def _load(index_dir: str = INDEX_DIR):
    """numpy backend loader (fallback / tests). The lance path never calls this,
    so chunk text does not stay resident."""
    emb = np.load(os.path.join(index_dir, "embeddings.npy"), mmap_mode="r")
    chunks = [json.loads(l) for l in open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8")]
    meta = json.load(open(os.path.join(index_dir, "meta.json"), encoding="utf-8"))
    return emb, chunks, meta


def _embed_query(query: str) -> np.ndarray:
    q = np.asarray(next(iter(_model(info()["model"]).embed([query]))), dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-9)


def search(query: str, k: int = 8) -> list[dict]:
    q = _embed_query(query)
    if _use_lance():
        # nprobes=100 + refine_factor=20 (re-rank on raw vectors) matched exact numpy
        # recall in production; warm query ~55ms there. ⚠ num_partitions is
        # min(512, max(16, sqrt(N))) (build_lance_index), so 100 is a full scan on a
        # small table and a fifth of them only at ~262k+ chunks — the ratio is not a
        # constant, and neither is the latency.
        rows = (_lance_table().search(q).metric("cosine")
                .nprobes(100).refine_factor(20).limit(k).to_list())
        return [{
            "score": round(1.0 - float(r["_distance"]), 4),   # cosine distance -> similarity
            "file": r["file"],
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "text": r["text"],
        } for r in rows]
    # --- numpy fallback ---
    emb, chunks, _ = _load(INDEX_DIR)
    scores = emb @ q
    k = min(k, len(chunks))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    out = []
    for i in top:
        c = chunks[int(i)]
        out.append({
            "score": round(float(scores[i]), 4),
            "file": c["file"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "text": c["text"],
        })
    return out


def degraded_status() -> dict:
    """Report the degradation registry - and report **"unknown", never 0, when it cannot be read.**

    This is the only genuinely important design decision in this function. To a
    caller deciding whether a search miss means anything, "the registry lists zero
    degraded files" and "I could not read the registry" are *opposite* statements:
    the first says a negative result is trustworthy, the second says nothing at all.
    Defaulting a failed read to 0 makes the caller assert exactly when it should
    have held back. Same family as an empty-sample check that passes because it ran
    over nothing.

    Not cached: the registry is a living file, regenerated after each repair batch,
    and a cached count would hand out stale numbers.
    """
    if not DEGRADED_REGISTRY:
        return {"status": "unknown",
                "reason": "no registry configured (set SEMANTIC_RECALL_DEGRADED)"}
    try:
        with open(DEGRADED_REGISTRY, encoding="utf-8") as fh:
            d = json.load(fh)
    except FileNotFoundError:
        return {"status": "unknown", "reason": "registry not found at %s" % DEGRADED_REGISTRY}
    except (json.JSONDecodeError, OSError) as e:
        return {"status": "unknown", "reason": "%s: %r" % (type(e).__name__, e)}

    files = d.get("files")
    if not isinstance(files, list):
        return {"status": "unknown", "reason": "malformed registry: 'files' is not an array"}

    out = {
        "status": "ok",
        "degraded_files": len(files),
        "generated": d.get("generated"),
        "meaning": "grep is unreliable on these files (0 hits != absent); semantic recall still works",
    }
    # A declared count that disagrees with the actual length means the generator
    # drifted. Surface both rather than silently picking one.
    declared = d.get("count")
    if declared is not None and declared != len(files):
        out["count_mismatch"] = {"declared": declared, "actual": len(files)}

    # A second, independent key: files that are recallable but not citable - sources
    # with no page-bearing original behind them. The two keys answer different
    # questions and must not be merged: "files" governs whether a *negative*
    # conclusion is trustworthy; "non_citable" governs whether a *positive* quote may
    # be used. Folding one into the other inflates the first with unrelated reasons.
    nc = d.get("non_citable")
    if isinstance(nc, list):
        out["non_citable"] = len(nc)
    elif isinstance(nc, dict):
        out["non_citable"] = len(nc.get("files", []))
    return out


@functools.lru_cache(maxsize=1)
def _reranker():
    """Lazily load the cross-encoder. Nothing is fetched until deep_search runs."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from fastembed.common.model_description import ModelSource
    try:
        TextCrossEncoder.add_custom_model(model=RERANKER, model_file="onnx/model.onnx",
                                          sources=ModelSource(hf=RERANKER))
    except Exception:
        pass  # already registered; re-registering raises and is harmless
    _ensure_reranker_external_data()
    return TextCrossEncoder(model_name=RERANKER, cache_dir=MODEL_CACHE)


def _ensure_reranker_external_data():
    """Large ONNX models ship their weights beside the graph as external data, and
    fastembed does not fetch that file. Without it the model loads and then fails -
    so fetch it once, up front, rather than discovering it mid-query.

    ⚠ The exception handling is deliberately narrow. A bare ``except Exception:
    pass`` here would cancel the function's entire reason for existing: a network
    timeout, a full disk or a rate limit would be swallowed, and you would discover
    the failure exactly where this call was meant to prevent it — mid-query. Only
    "this repository has no external-data file" is a benign outcome; everything
    else must surface.
    """
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(RERANKER, "onnx/model.onnx_data", cache_dir=MODEL_CACHE)
    except Exception as e:
        # Benign: smaller rerankers keep their weights inside the graph, so the
        # side file genuinely does not exist. Identify it by name rather than by
        # class, because the hub library moves these exceptions between versions.
        if type(e).__name__ in ("EntryNotFoundError", "RepositoryNotFoundError"):
            return
        raise


def _expand_to_paragraph(tbl, hit: dict, radius: int = 1) -> str:
    """Widen a recalled chunk into its surrounding passage before reranking.

    Chunks are sized for recall, not for judgement: a cross-encoder scoring a
    120-token fragment is reading less than the author wrote. Filter by id range
    first (cheap), then drop foreign files in Python - which also stops the window
    from walking across a file boundary.
    """
    lo, hi = hit["id"] - radius, hit["id"] + radius
    cands = (tbl.search().where("id >= %d and id <= %d" % (lo, hi))
             .select(["id", "file", "text"]).limit(2 * radius + 1).to_list())
    same = sorted([c for c in cands if c["file"] == hit["file"]], key=lambda x: x["id"])
    return " ".join(c["text"] for c in same) if same else hit["text"]


def deep_search(query: str, top_n: int = 30, top_k: int = 8, radius: int = 1) -> list[dict]:
    """Recall wide, expand to passages, rerank, return the top few.

    **This does not replace search().** It exists for one situation: the passage you
    need is in the corpus but sits below the recall cutoff - the twentieth hit rather
    than the third. Reranking pulls it up. When plain recall is already precise,
    reranking can just as easily scatter it, so reach for this deliberately.

    **score is a reranker logit, not a cosine.** It orders results and nothing more:
    there is no threshold at which it becomes "relevant", and comparing scores across
    queries is meaningless. recall_score (the original cosine) is kept alongside it so
    the two views stay separable.
    """
    if not _use_lance():
        raise RuntimeError("deep_search requires the LanceDB backend")
    tbl = _lance_table()
    q = _embed_query(query)
    rows = (tbl.search(q).metric("cosine").nprobes(100).refine_factor(20)
            .limit(top_n).to_list())
    hits = [{"id": int(r["id"]), "file": r["file"],
             "page_start": r["page_start"], "page_end": r["page_end"],
             "text": r["text"], "recall_score": round(1.0 - float(r["_distance"]), 4)}
            for r in rows]
    if not hits:
        return []
    paras = [_expand_to_paragraph(tbl, h, radius) for h in hits]
    scores = list(_reranker().rerank(query, paras))
    order = sorted(range(len(hits)), key=lambda i: -scores[i])[:top_k]
    return [{
        "score": round(float(scores[i]), 4),        # reranker logit - ordering only
        "file": hits[i]["file"],
        "page_start": hits[i]["page_start"],
        "page_end": hits[i]["page_end"],
        "text": hits[i]["text"],
        "recall_score": hits[i]["recall_score"],    # original cosine, kept for contrast
    } for i in order]


def _fmt(r: dict) -> str:
    pg = f"p.{r['page_start']}" if r["page_start"] == r["page_end"] else f"p.{r['page_start']}-{r['page_end']}"
    snip = r["text"] if len(r["text"]) <= 300 else r["text"][:300] + "…"
    return f"[{r['score']:.3f}] {r['file']}  ({pg})\n    {snip}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()
    m = info()
    print(f"# backend: {'LanceDB' if _use_lance() else 'numpy'} · {m['count']} chunks / {m['n_files']} files · model {m['model']}\n")
    for r in search(args.query, args.k):
        print(_fmt(r))
        print()


if __name__ == "__main__":
    main()
