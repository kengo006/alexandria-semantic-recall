"""Build the IVF_PQ ANN vector index on the LanceDB chunks table.

Exact flat scan over ~410k chunks costs ~1.2s per query; the IVF_PQ approximate
index brings that to tens of ms. The recall cost is compensated at query time by
nprobes + refine_factor (see search_core).

Usage:  uv run python build_lance_index.py
"""
from __future__ import annotations
import sys, os, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LANCE = os.environ.get("SEMANTIC_RECALL_LANCE", os.path.join(HERE, "lance_db"))


def main():
    import lancedb
    db = lancedb.connect(LANCE)
    tbl = db.open_table("chunks")
    n = tbl.count_rows()
    print(f"rows = {n}")
    if n < 4096:
        # KMeans needs centroids << rows, and at this size an exact scan is already
        # fast — an ANN index would add nothing. Queries brute-force just fine.
        print("small table: skipping ANN index (exact scan is already fast)")
        return
    t = time.time()
    # num_partitions ~ sqrt(N), capped at 512; num_sub_vectors=96 (384/96 = 4 dims per sub-vector)
    parts = min(512, max(16, int(n ** 0.5)))
    tbl.create_index(metric="cosine", vector_column_name="vector",
                     index_type="IVF_PQ", num_partitions=parts, num_sub_vectors=96, replace=True)
    print(f"IVF_PQ index built in {time.time()-t:.0f}s (num_partitions={parts})")
    print("indices:", [ix.name for ix in tbl.list_indices()])


if __name__ == "__main__":
    main()
