"""incremental_update.py — incremental index update (embed only added/changed files · LanceDB upsert).

Pain point: every time the corpus gains or loses a few files, a full rebuild
re-embeds everything (hours at ~400k chunks). This script walks the delta instead:
scan corpus × manifest ->
    added/changed  -> chunk_text + embed + tbl.add
    removed/changed -> tbl.delete("file = ...")
Measured in production: +11/-1 files in ~90s vs ~2.8h full rebuild (~111×).
The ANN index does not need rebuilding every time.

File-change semantics (your index-maintenance routine depends on these):
    ● Deleted source: file disappears from corpus = delta.removed -> all its chunks
      deleted -> no longer recalled.
    ● Moved/renamed (path change): old path removed + new path added (content
      re-embedded once) -> chunks' file column updates to the new path.
    ⚠ Not real-time: between changing the corpus and running this script, the index
      is stale (deleted files still recalled, moved files under old paths). So
      "run after ingestion/changes" is a maintenance discipline; the citation layer
      has a second safety net (agents must verify quotes against the source PDF).

State layer lance_db/manifest.json:
    {"model","dim","files":{rel:{mtime,size}}, "next_id", "updated", ...}
    First run without a manifest -> bootstrap: baseline comes from the live table's
    distinct files (NOT the current corpus — the table reflects the last full build
    and may lag the corpus); mtime/size filled from current corpus values; files in
    the corpus but not the table skip the baseline and are picked up as `added` in
    the same run.

LanceDB only (the active query backend). The numpy index/ is not updated
incrementally (dense arrays don't delete rows well); it stays a full-rebuild
fallback layer — the two re-converge after the next full rebuild.

id: incremental chunks start at ID_BASE (500M) and grow (stored in
    manifest.next_id) — never colliding with full-build ids (<1e6) or each other.
    The query side never uses id (search returns file/page/text), so ids only need
    to be unique.

ANN: newly added rows not yet in IVF_PQ are still found by exact brute-force scan
    (recall does not drop; that batch is just slightly slower). The index is rebuilt
    when additions accumulate past --reindex-threshold (default 20000) or with
    --reindex.

Safety: --dry-run reports the delta without writing; the table version is printed
    before writes (rollback = tbl.checkout(v).restore()).

Usage:
    uv run python incremental_update.py                 # incremental update
    uv run python incremental_update.py --dry-run       # report delta only
    uv run python incremental_update.py --reindex       # update, then rebuild ANN
    uv run python incremental_update.py "<corpus path>" # explicit corpus (else CORPUS_ROOT)
"""
from __future__ import annotations

import sys, os, json, glob, argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from index_lib import chunk_text  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LANCE = os.environ.get("SEMANTIC_RECALL_LANCE", os.path.join(HERE, "lance_db"))
TABLE = "chunks"
MODEL_CACHE = os.path.join(HERE, "model_cache")
DEFAULT_CORPUS = os.environ.get("CORPUS_ROOT")
ID_BASE = 500_000_000       # incremental id floor, far above full-build ids (<1e6)
ADD_BATCH = 5000
EMBED_BATCH = 32            # 16 GB-RAM lesson: big batches spike RAM; 32 is measured-safe


def _now_str() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def scan_corpus(root: str) -> dict:
    """{rel: {mtime,size}}, excluding `_`-prefixed working files (consistent with
    migrate_to_lance). rel uses os.path.relpath = platform separators, same format
    as the table's file column."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        if os.path.basename(p).startswith("_"):
            continue
        st = os.stat(p)
        out[os.path.relpath(p, root)] = {"mtime": round(st.st_mtime, 2), "size": st.st_size}
    return out


def table_files(tbl) -> set:
    """Distinct file set of the live table (scans only the file column, not text/vector)."""
    n = tbl.count_rows()
    rows = tbl.search().select(["file"]).limit(n).to_list()
    return {r["file"] for r in rows}


def bootstrap_baseline(tbl, root: str) -> dict:
    """Baseline = files actually in the table (not the corpus); mtime/size filled
    from current corpus values; missing files recorded as None (treated as removed
    next time)."""
    base = {}
    for rel in table_files(tbl):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            st = os.stat(p)
            base[rel] = {"mtime": round(st.st_mtime, 2), "size": st.st_size}
        else:
            base[rel] = {"mtime": None, "size": None}
    return base


def compute_delta(now: dict, prev: dict):
    now_s, prev_s = set(now), set(prev)
    added = sorted(now_s - prev_s)
    removed = sorted(prev_s - now_s)
    changed = sorted(r for r in (now_s & prev_s)
                     if now[r]["mtime"] != prev[r].get("mtime")
                     or now[r]["size"] != prev[r].get("size"))
    return added, removed, changed


def embed_files(root: str, rels: list, model):
    """Chunk + embed (L2-normalized) the given files -> rows (without id)."""
    chunks = []
    for rel in rels:
        text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
        chunks.extend(chunk_text(text, rel))
    if not chunks:
        return []
    vecs = [np.asarray(v, dtype=np.float32) for v in model.embed([c.text for c in chunks], batch_size=EMBED_BATCH)]
    emb = np.vstack(vecs)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return [{"file": c.file, "page_start": int(c.page_start), "page_end": int(c.page_end),
             "text": c.text, "vector": emb[i].tolist()} for i, c in enumerate(chunks)]


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def run(lance_dir: str, corpus: str, *, dry_run=False, reindex=False, reindex_threshold=20000) -> dict:
    import lancedb
    if not os.path.isdir(lance_dir):
        print(f"[abort] no {lance_dir}; run rebuild.py for a full build first.")
        sys.exit(2)
    root = os.path.abspath(corpus)
    manifest_p = os.path.join(lance_dir, "manifest.json")
    meta_p = os.path.join(lance_dir, "meta.json")
    tbl = lancedb.connect(lance_dir).open_table(TABLE)
    meta = json.load(open(meta_p, encoding="utf-8"))

    now = scan_corpus(root)
    man = json.load(open(manifest_p, encoding="utf-8")) if os.path.exists(manifest_p) else None
    if man is None:
        prev = bootstrap_baseline(tbl, root)
        next_id = ID_BASE
        print(f"[bootstrap] no manifest -> baseline from table: {len(prev)} files (corpus currently {len(now)})")
    else:
        prev = man["files"]
        next_id = man.get("next_id", ID_BASE)

    added, removed, changed = compute_delta(now, prev)
    count_before = tbl.count_rows()
    print(f"[delta] +{len(added)} added · -{len(removed)} removed · ~{len(changed)} changed · live {count_before} chunks / {len(prev)} files")
    for tag, lst in (("added", added), ("removed", removed), ("changed", changed)):
        for rel in lst[:20]:
            print(f"        [{tag}] {rel}")
        if len(lst) > 20:
            print(f"        [{tag}] … {len(lst)} total")

    if not (added or removed or changed):
        print("[no change] index already in sync with corpus.")
        if man is None and not dry_run:      # first run still writes the baseline manifest
            _write_manifest(manifest_p, meta, now, next_id, bootstrap=True)
            print("[bootstrap] baseline manifest written.")
        return {"added": 0, "removed": 0, "changed": 0, "added_rows": 0,
                "count_before": count_before, "count_after": count_before}

    if dry_run:
        print("[dry-run] report only, nothing written.")
        return {"added": len(added), "removed": len(removed), "changed": len(changed),
                "added_rows": None, "count_before": count_before, "count_after": count_before}

    v0 = tbl.version
    print(f"[safety] table version before writes = {v0} (rollback = tbl.checkout({v0}).restore())")

    # 1) delete (removed ∪ changed) — delete first, changed files get re-added fresh
    for rel in removed + changed:
        tbl.delete("file = " + _sql_str(rel))

    # 2) embed + add (added ∪ changed)
    added_rows = 0
    to_embed = added + changed
    if to_embed:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=meta["model"], cache_dir=MODEL_CACHE)
        rows = embed_files(root, to_embed, model)
        for i, r in enumerate(rows):
            r["id"] = next_id + i
        next_id += len(rows)
        for i in range(0, len(rows), ADD_BATCH):
            tbl.add(rows[i:i + ADD_BATCH])
        added_rows = len(rows)
        print(f"[add] {added_rows} chunks (from {len(to_embed)} files) · id …~{next_id-1}")

    # 3) ANN rebuild (threshold or forced)
    reindexed = False
    count_now = tbl.count_rows()
    if (reindex or added_rows >= reindex_threshold) and count_now >= 4096:
        import time as _t
        t = _t.time()
        parts = min(512, max(16, int(count_now ** 0.5)))   # KMeans needs centroids << rows
        tbl.create_index(metric="cosine", vector_column_name="vector",
                         index_type="IVF_PQ", num_partitions=parts, num_sub_vectors=96, replace=True)
        reindexed = True
        print(f"[reindex] IVF_PQ rebuilt · {_t.time()-t:.0f}s (num_partitions={parts})")
    elif reindex and count_now < 4096:
        print(f"[reindex] skipped: table too small for ANN ({count_now} rows); exact scan is already fast")
    elif added_rows:
        print(f"[reindex] skipped ({added_rows} added < threshold {reindex_threshold}); unindexed rows are covered by exact brute-force scan — recall does not drop")

    # 4) persist state: manifest (files = current corpus, now in sync) + meta (count/n_files)
    _write_manifest(manifest_p, meta, now, next_id, bootstrap=(man is None))
    count_after = tbl.count_rows()
    meta.update({"count": count_after, "n_files": len(now),
                 "last_incremental": _now_str(), "backend": "lancedb"})
    json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[done] {count_before} -> {count_after} chunks · {len(prev)} -> {len(now)} files"
          f"{' · ANN rebuilt' if reindexed else ''}")
    return {"added": len(added), "removed": len(removed), "changed": len(changed),
            "added_rows": added_rows, "count_before": count_before, "count_after": count_after}


def _write_manifest(path, meta, files, next_id, *, bootstrap):
    obj = {"model": meta["model"], "dim": int(meta["dim"]),
           "files": files, "next_id": next_id, "updated": _now_str()}
    if bootstrap:
        obj["bootstrapped"] = _now_str()
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default=DEFAULT_CORPUS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--reindex-threshold", type=int, default=20000)
    args = ap.parse_args()
    if not args.corpus:
        sys.exit("no corpus given: pass a folder or set CORPUS_ROOT")
    run(LANCE, args.corpus, dry_run=args.dry_run,
        reindex=args.reindex, reindex_threshold=args.reindex_threshold)


if __name__ == "__main__":
    main()
