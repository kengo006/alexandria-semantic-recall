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
        # nprobes=100 (scan 100/512 partitions) + refine_factor=20 (re-rank on raw
        # vectors) matches exact numpy recall in production; warm query ~55ms.
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
