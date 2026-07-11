"""Migrate the numpy index (index/embeddings.npy + chunks.jsonl) into LanceDB.

Why: the numpy backend loads every chunk's text into RAM (~800 MB resident at
~410k chunks — painful on a 16 GB machine). LanceDB keeps text and vectors on
disk and fetches only top-k per query, so resident RAM barely grows with corpus
size.

Design: streaming read (chunks.jsonl line by line, embeddings via mmap slices)
-> batched writes, so the migration itself cannot spike RAM. Files whose basename
starts with `_` are excluded (working files in the corpus root would otherwise
leak noise chunks into the index).

This is step 2 of every full rebuild (see rebuild.py), not a one-off.

Usage:  uv run python migrate_to_lance.py            # reads index/ -> writes lance_db/
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = os.environ.get("SEMANTIC_RECALL_INDEX", os.path.join(HERE, "index"))
LANCE = os.environ.get("SEMANTIC_RECALL_LANCE", os.path.join(HERE, "lance_db"))
TABLE = "chunks"
BATCH = 5000


def main():
    import lancedb
    import pyarrow as pa

    meta = json.load(open(os.path.join(IDX, "meta.json"), encoding="utf-8"))
    dim = int(meta["dim"])
    emb = np.load(os.path.join(IDX, "embeddings.npy"), mmap_mode="r")
    n = emb.shape[0]
    assert emb.shape[1] == dim, f"dim mismatch {emb.shape[1]} vs {dim}"
    print(f"[source] {n} chunks / dim={dim} / model={meta['model']}")

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("file", pa.string()),
        pa.field("page_start", pa.int32()),
        pa.field("page_end", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])
    db = lancedb.connect(LANCE)
    tbl = db.create_table(TABLE, schema=schema, mode="overwrite")

    t0 = time.time()
    buf = []
    written = skipped = 0
    with open(os.path.join(IDX, "chunks.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            c = json.loads(line)
            if os.path.basename(c["file"]).startswith("_"):   # exclude working-file noise
                skipped += 1
                continue
            buf.append({
                "id": int(c["id"]),
                "file": c["file"],
                "page_start": int(c["page_start"]),
                "page_end": int(c["page_end"]),
                "text": c["text"],
                "vector": emb[i].astype(np.float32).tolist(),
            })
            if len(buf) >= BATCH:
                tbl.add(buf)
                written += len(buf)
                buf = []
                if written % 50000 == 0:
                    print(f"   {written}/{n}  ({written/(time.time()-t0):.0f}/s)")
    if buf:
        tbl.add(buf)
        written += len(buf)

    dt = time.time() - t0
    # Store a meta copy so the query side gets the model name without touching index/.
    with open(os.path.join(LANCE, "meta.json"), "w", encoding="utf-8") as mf:
        json.dump({**meta, "count": written, "excluded_workfiles": skipped,
                   "migrated_from": "index/", "backend": "lancedb"}, mf, ensure_ascii=False, indent=2)
    print(f"[done] wrote {written} chunks (excluded {skipped} working-file chunks) · {dt:.0f}s · {written/dt:.0f}/s")
    print(f"[LanceDB] {LANCE}  table={TABLE}  rows={tbl.count_rows()}")


if __name__ == "__main__":
    main()
