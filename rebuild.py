"""Full index rebuild — three steps as one unit, preventing dual-backend drift.

Background (a real production incident): the server is dual-backend and prefers
`lance_db/`. If you run only build_index.py (refreshing the numpy `index/`)
without migrating to LanceDB, the server silently serves stale data. So a rebuild
always binds three steps:
  1) build_index.py       corpus -> index/    (embedding; the slow step)
  2) migrate_to_lance.py  index/  -> lance_db/ (overwrite; safe to re-run)
  3) build_lance_index.py IVF_PQ ANN index
  4) verify backend/count/built via search_core.info()

Usage:  uv run python rebuild.py ["<corpus path>"] [--skip-embed]
        Corpus defaults to the CORPUS_ROOT env var. Sub-steps reuse this venv's
        python (sys.executable — no nested uv).
"""
from __future__ import annotations
import sys, os, time, shutil, subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # venv python provided by uv run; sub-steps share all deps
DEFAULT_CORPUS = os.environ.get("CORPUS_ROOT")
INDEX = os.environ.get("SEMANTIC_RECALL_INDEX", os.path.join(HERE, "index"))
INDEX_PREV = os.path.join(HERE, "index_prev")


def _backup_prev_index() -> None:
    """Before rebuilding, keep one generation of the current index/ -> index_prev/
    (a numpy rollback layer; overwrites the previous generation)."""
    if os.path.isdir(INDEX):
        if os.path.isdir(INDEX_PREV):
            shutil.rmtree(INDEX_PREV)
        os.replace(INDEX, INDEX_PREV)
        print("[backup] index/ -> index_prev/ (previous-generation numpy backup for rollback)", flush=True)


def step(title: str, script: str, *args: str) -> None:
    print(f"\n===== {title} =====", flush=True)
    t = time.time()
    r = subprocess.run([PY, os.path.join(HERE, script), *args])
    if r.returncode != 0:
        print(f"[abort] {script} failed exit {r.returncode} — rebuild incomplete, server still serves the previous version (no swap happened).", flush=True)
        sys.exit(r.returncode)
    print(f"[ok] {script} · {time.time()-t:.0f}s", flush=True)


def main():
    argv = sys.argv[1:]
    skip_embed = "--skip-embed" in argv          # corpus unchanged: only re-migrate to LanceDB
    pos = [a for a in argv if not a.startswith("--")]
    corpus = pos[0] if pos else DEFAULT_CORPUS
    t0 = time.time()
    if skip_embed:
        print("[skip] step1 embedding (--skip-embed: reusing existing index/, re-migrating LanceDB only)", flush=True)
    else:
        if not corpus:
            print("[abort] no corpus given: pass a folder or set CORPUS_ROOT")
            sys.exit(2)
        if not os.path.isdir(corpus):
            print(f"[abort] corpus does not exist: {corpus}")
            sys.exit(2)
        _backup_prev_index()
        step("1/3 embed -> index/ (numpy)", "build_index.py", corpus)
    step("2/3 migrate -> lance_db/", "migrate_to_lance.py")
    step("3/3 IVF_PQ ANN index", "build_lance_index.py")

    # 4) verify
    print("\n===== verify =====", flush=True)
    sys.path.insert(0, HERE)
    import search_core as sc
    m = sc.info()
    ok = sc._use_lance() and m.get("backend") == "lancedb"
    print(f"backend={m.get('backend')} · count={m.get('count')} · built={m.get('built')}", flush=True)
    print(f"\n{'[done]' if ok else '[warning: not on the LanceDB backend]'} full rebuild, three steps as one · total {time.time()-t0:.0f}s", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
