"""Regression test: after removing the table/info caches from search_core, commits
made to the table by another handle become visible in the same process (no server
restart needed). Temp dir · monkeypatched LANCE_DIR · never touches production.
Run:  uv run --project . python _test_refresh.py"""
import sys, os, tempfile, shutil, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lancedb, pyarrow as pa
import search_core as sc
from index_lib import chunk_text
from fastembed import TextEmbedding

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384


def make(topic):
    return f"===== page 1 =====\n{(topic + '. ') * 40}\n"


def emb_rows(model, topic, f, base):
    cks = chunk_text(make(topic), f)
    vs = [np.asarray(v, dtype=np.float32) for v in model.embed([c.text for c in cks], batch_size=16)]
    e = np.vstack(vs)
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    return [{"id": base + i, "file": c.file, "page_start": int(c.page_start),
             "page_end": int(c.page_end), "text": c.text, "vector": e[i].tolist()}
            for i, c in enumerate(cks)]


def wmeta(lance, count, n):
    json.dump({"model": MODEL, "dim": DIM, "count": count, "n_files": n, "backend": "lancedb"},
              open(os.path.join(lance, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)


def main():
    d = tempfile.mkdtemp(prefix="refresh_")
    lance = os.path.join(d, "lance_db")
    try:
        model = TextEmbedding(model_name=MODEL, cache_dir=sc.MODEL_CACHE)
        sc.LANCE_DIR = lance                     # monkeypatch -> search_core points at temp lance
        try:
            sc._lance_db.cache_clear()
        except Exception:
            pass

        schema = pa.schema([pa.field("id", pa.int64()), pa.field("file", pa.string()),
                            pa.field("page_start", pa.int32()), pa.field("page_end", pa.int32()),
                            pa.field("text", pa.string()), pa.field("vector", pa.list_(pa.float32(), DIM))])
        tbl = lancedb.connect(lance).create_table("chunks", schema=schema, mode="overwrite")
        tbl.add(emb_rows(model, "public monuments and official commemoration", os.path.join("archives", "monuments.txt"), 0))
        wmeta(lance, tbl.count_rows(), 1)

        c1 = sc.info()["count"]
        r1 = [x["file"] for x in sc.search("how communities remember online", 3)]
        print(f"[initial] info.count={c1} · search 'communities remember' hits={r1}")

        # Same process: another handle commits an increment (simulating another
        # process / a previous incremental write).
        tbl2 = lancedb.connect(lance).open_table("chunks")
        tbl2.add(emb_rows(model, "digital memory platforms change how communities remember",
                          os.path.join("archives", "digital.txt"), 500000000))
        wmeta(lance, tbl2.count_rows(), 2)

        c2 = sc.info()["count"]                  # info uncached -> must see the new count
        r2 = [x["file"] for x in sc.search("how communities remember online", 3)]   # table reopened per query -> must see the new file
        print(f"[after increment · same process · no restart] info.count={c2} · hits={r2}")

        assert c2 > c1, f"info still caching old count ({c1}->{c2}) = fix regressed"
        assert any("digital" in f for f in r2), f"same-process increment invisible: {r2} = fix regressed"
        print(f"\n✓ same-process info {c1}->{c2}, new content 'digital' immediately searchable (no server restart)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
