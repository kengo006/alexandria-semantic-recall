"""Build the index: walk a corpus folder -> chunk -> fastembed embeddings -> index/.

Usage:
    uv run python build_index.py "<corpus_folder>"
    uv run python build_index.py "<corpus_folder>" --out index --model <hf_name>

The corpus folder is your text layer: one .txt per source PDF, page markers
preserved (see index_lib). CORPUS_ROOT env is used when no argument is given.

Output (index/):
    embeddings.npy   float32 (N, dim), L2-normalized -> cosine = dot product
    chunks.jsonl     one chunk per line {id,file,page_start,page_end,text}
    meta.json        {model,dim,count,corpus_root,built,...}
"""
from __future__ import annotations

import sys, os, json, glob, time, argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from index_lib import chunk_text  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows narrow-codepage console guard
except Exception:
    pass

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Model cache pinned inside the project (the default %TEMP%\fastembed_cache gets
# wiped by disk cleanup -> instability).
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")


def load_corpus(root: str):
    files = sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))
    all_chunks = []
    for p in files:
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except Exception as e:
            print(f"  skipped (read failed) {p}: {e}")
            continue
        rel = os.path.relpath(p, root)
        all_chunks.extend(chunk_text(text, rel))
    return files, all_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default=os.environ.get("CORPUS_ROOT"),
                    help="corpus folder (recursive *.txt); defaults to CORPUS_ROOT env")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "index"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=32)  # 16 GB-RAM lesson: big batches spike RAM; 32 is measured-safe
    args = ap.parse_args()
    if not args.corpus:
        sys.exit("no corpus given: pass a folder or set CORPUS_ROOT")

    root = os.path.abspath(args.corpus)
    print(f"[1/4] scanning corpus: {root}")
    files, chunks = load_corpus(root)
    print(f"       {len(files)} files -> {len(chunks)} chunks")
    if not chunks:
        print("       nothing to index, aborting.")
        sys.exit(1)

    print(f"[2/4] loading embedding model: {args.model} (cache {MODEL_CACHE})")
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=args.model, cache_dir=MODEL_CACHE)

    print(f"[3/4] embedding {len(chunks)} chunks ...")
    t0 = time.time()
    texts = [c.text for c in chunks]
    vecs = []
    done = 0
    for v in model.embed(texts, batch_size=args.batch):
        vecs.append(np.asarray(v, dtype=np.float32))
        done += 1
        if done % 1000 == 0:
            dt = time.time() - t0
            print(f"       {done}/{len(chunks)}  ({done/dt:.0f} chunks/s)")
    emb = np.vstack(vecs)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)  # L2 normalize
    dt = time.time() - t0
    print(f"       done: {emb.shape}  in {dt:.0f}s  ({len(chunks)/dt:.0f} chunks/s)")

    print(f"[4/4] writing index -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "embeddings.npy"), emb)
    with open(os.path.join(args.out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps(c.to_dict(i), ensure_ascii=False) + "\n")
    meta = {
        "model": args.model,
        "dim": int(emb.shape[1]),
        "count": int(emb.shape[0]),
        "n_files": len(files),
        "corpus_root": root,
        "built": datetime.now().astimezone().isoformat(timespec="seconds"),
        "build_seconds": round(dt, 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("       OK:", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
