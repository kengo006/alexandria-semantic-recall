"""End-to-end unit test for incremental_update (temp dir, never touches production).
Build a small table (files 1,2 · simulating post-migration, no manifest) ->
A add file 3 -> B change file 1 -> C delete file 3 -> D idempotency.
Each step verifies delta / count / recall.
Run:  uv run --project . python _test_incremental.py

Note: a LanceDB table handle is bound to the version at open time; iu.run() writes
through its own connection, so verification always uses a fresh open_table (inside
hits()) and counts from run()'s return value — never a stale handle.
"""
import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lancedb, pyarrow as pa
import incremental_update as iu
from fastembed import TextEmbedding

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def make_txt(topic):
    return (f"===== page 1 =====\n{(topic + '. ') * 40}\n"
            f"===== page 2 =====\n{(topic + ' — extended examples and discussion. ') * 30}\n")


def qvec(model, q):
    v = np.asarray(next(iter(model.embed([q]))), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def hits(lance, model, q, k=5):
    tbl = lancedb.connect(lance).open_table("chunks")   # fresh handle = latest version
    return [r["file"] for r in tbl.search(qvec(model, q)).metric("cosine").limit(k).to_list()]


def main():
    d = tempfile.mkdtemp(prefix="iu_test_")
    corpus, lance = os.path.join(d, "corpus"), os.path.join(d, "lance_db")
    os.makedirs(lance)
    try:
        model = TextEmbedding(model_name=MODEL, cache_dir=iu.MODEL_CACHE)
        f1 = os.path.join("archives", "oral_history.txt")
        f2 = os.path.join("archives", "public_monuments.txt")
        f3 = os.path.join("archives", "digital_memory.txt")
        w(os.path.join(corpus, f1), make_txt("oral history interviews preserve testimony of witnesses"))
        w(os.path.join(corpus, f2), make_txt("public monuments shape official commemoration of the past"))

        schema = pa.schema([pa.field("id", pa.int64()), pa.field("file", pa.string()),
                            pa.field("page_start", pa.int32()), pa.field("page_end", pa.int32()),
                            pa.field("text", pa.string()), pa.field("vector", pa.list_(pa.float32(), DIM))])
        tbl = lancedb.connect(lance).create_table("chunks", schema=schema, mode="overwrite")
        rows = iu.embed_files(corpus, [f1, f2], model)
        for i, r in enumerate(rows):
            r["id"] = i
        tbl.add(rows)
        json.dump({"model": MODEL, "dim": DIM, "count": len(rows), "n_files": 2, "backend": "lancedb"},
                  open(os.path.join(lance, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
        base = tbl.count_rows()
        print(f"[setup] {base} chunks / 2 files (no manifest · simulating post-migration)\n")

        # A: add file 3
        w(os.path.join(corpus, f3), make_txt("digital memory platforms change how communities remember"))
        r = iu.run(lance, corpus)
        assert r["added"] == 1 and r["removed"] == 0 and r["changed"] == 0, r
        assert r["count_after"] > base, r
        assert os.path.exists(os.path.join(lance, "manifest.json")), "manifest not created"
        assert any("digital_memory" in h for h in hits(lance, model, "how communities remember online")), "new file not recalled"
        print(f"[A add] OK · manifest created · {base}->{r['count_after']} · recall includes digital_memory ✓\n")

        # B: change file 1
        w(os.path.join(corpus, f1), make_txt("oral history now with an added epistemology section on testimony and memory reliability, greatly expanded"))
        r = iu.run(lance, corpus)
        assert r["changed"] == 1 and r["added"] == 0, r
        assert any("oral_history" in h for h in hits(lance, model, "epistemology of testimony and reliability")), "changed content not recalled"
        print(f"[B change] OK · ~1 file · {r['count_before']}->{r['count_after']} · new content recallable ✓\n")

        # C: delete file 3
        os.remove(os.path.join(corpus, f3))
        r = iu.run(lance, corpus)
        assert r["removed"] == 1, r
        assert r["count_after"] < r["count_before"], r
        assert not any("digital_memory" in h for h in hits(lance, model, "how communities remember online")), "still recalled after delete"
        print(f"[C delete] OK · -1 file · {r['count_before']}->{r['count_after']} · digital_memory gone ✓\n")

        # D: idempotency
        r = iu.run(lance, corpus)
        assert r["added"] == 0 and r["removed"] == 0 and r["changed"] == 0, r
        print("[D idempotent] OK · second run sees no changes ✓\n")
        print("=== all tests passed ===")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
