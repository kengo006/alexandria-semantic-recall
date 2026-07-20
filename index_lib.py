"""Chunking + page mapping.

The text layer (one .txt per source PDF) can carry page boundaries in two formats:
  - born-digital : pdftotext -layout output, page break = form feed \\f (0x0C);
                   page number = count of preceding \\f + 1
  - scanned/OCR  : explicit markers  ===== page N =====
This module absorbs both, cutting the text into small, sentence-aligned chunks,
each carrying {relative file path, start page, end page}.

**Chunks are sized in TOKENS, not characters (v0.2 — a correctness fix).**

v0.1 cut ~500-character windows and presented that as a fit for the default
model's max_seq ≈ 128 tokens. That reasoning only holds for Western scripts:
the char↔token ratio varies roughly 3× across languages (Latin-script prose
≈ 3.9 chars/token, CJK ≈ 1.4). A 500-character CJK chunk is ~355 tokens — and
the model's tokenizer silently truncates at 128, so **over half of every CJK
chunk never entered the index at all**. Western text lost only ~3%, which is
why the defect stayed invisible. Measured directly: embedding a full chunk vs
only its first 128 tokens returns cosine 1.0000 (an unrelated control pair
scores ~0.08) — everything past the truncation point contributes nothing.

For a recall layer whose headline claim is *cross-lingual* search, that is the
one place this bug must not live. The fix cuts by token count using the
**embedding model's own tokenizer** (no estimation, no second rule set):
TARGET_TOKENS = 120 keeps every chunk under the 128 limit with margin for
special tokens. The minimum-size floor moved to tokens too — a character
floor (min_chars=200) filters out almost every CJK chunk, since 120 CJK
tokens is only ~150 characters.

Rejected alternative: raising max_length to 512. Attention cost is O(n²),
index build time balloons, and for a majority-Western corpus it is pure waste.
Small chunks remain the right shape here: the consumer is a read-only search
agent that wants a *precise pointer* to a passage — it then goes back to the
source PDF for context. Recall-only, never a citation source (see alexandria,
optional-integrations §2).
"""
from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, asdict

FF = "\f"
_OCR_PAGE = re.compile(r"=====\s*page\s+(\d+)\s*=====")
_BOUNDARY = re.compile(r"[.!?。！？；;\n]")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE = os.path.join(HERE, "model_cache")   # same pin as build_index.py
TARGET_TOKENS = 120      # < the tokenizer's truncation.max_length=128, margin for [CLS]/[SEP]
OVERLAP_TOKENS = 24      # ≈ the v0.1 overlap ratio (100/500)
MIN_TOKENS = 30          # the floor must be tokens too (see module docstring)


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """Load the SAME tokenizer the embedding model uses (not an estimate,
    not a second rule set). Looks for the tokenizer.json that fastembed
    downloaded into MODEL_CACHE; returns None if unavailable (callers fall
    back to character chunking so the tool never hard-stops)."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    for root, _dirs, files in os.walk(MODEL_CACHE):
        if "tokenizer.json" in files:
            tok = Tokenizer.from_file(os.path.join(root, "tokenizer.json"))
            # Chunking must see offsets for ALL tokens, so disable the built-in
            # 128 truncation here. The embedding stage keeps its own truncating
            # tokenizer — our chunks are already <128 tokens, so nothing is cut.
            try:
                tok.no_truncation()
            except Exception:
                pass
            return tok
    return None


@dataclass
class Chunk:
    file: str          # path relative to corpus root
    page_start: int
    page_end: int
    text: str

    def to_dict(self, idx: int) -> dict:
        d = asdict(self)
        d["id"] = idx
        return d


def _page_marks(text: str) -> list[tuple[int, int]]:
    """Return [(char_offset, page_no), ...] in increasing offset order, marking each page start."""
    ocr = list(_OCR_PAGE.finditer(text))
    if ocr:
        marks = [(0, 1)]
        for m in ocr:
            marks.append((m.end(), int(m.group(1))))
        return marks
    # born-digital: accumulate page number over \f
    marks = [(0, 1)]
    page = 1
    start = 0
    while True:
        i = text.find(FF, start)
        if i == -1:
            break
        page += 1
        marks.append((i + 1, page))
        start = i + 1
    return marks


def _page_at(marks: list[tuple[int, int]], offset: int) -> int:
    """Binary-search the page containing offset."""
    lo, hi, ans = 0, len(marks) - 1, marks[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if marks[mid][0] <= offset:
            ans = marks[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def chunk_text(text: str, file_rel: str, target: int = 500,
               overlap: int = 100, min_chars: int = 200, slack: int = 160) -> list[Chunk]:
    """Chunk `text`; window tails snap to the nearest sentence boundary.
    Page numbers are back-computed from original-text offsets.

    **Default path = token chunking** (TARGET_TOKENS; valid across languages —
    see the module docstring). The `target`/`overlap`/`min_chars` character
    parameters only apply on the fallback path when no tokenizer is available
    (kept for API compatibility, and so the tool degrades instead of stopping).
    """
    tok = _tokenizer()
    if tok is None:
        return _chunk_by_chars(text, file_rel, target, overlap, min_chars, slack)
    return _chunk_by_tokens(text, file_rel, slack)


def _chunk_by_tokens(text: str, file_rel: str, slack: int) -> list[Chunk]:
    """Token chunking: take offsets from the embedding model's own tokenizer,
    ≤TARGET_TOKENS per chunk, window tail still snaps to a sentence boundary
    (within `slack` chars) without exceeding the token budget. Page numbers are
    still back-computed from original-text character offsets (mechanism unchanged)."""
    tok = _tokenizer()
    marks = _page_marks(text)
    enc = tok.encode(text)
    offs = [o for o in enc.offsets if o[1] > o[0]]     # drop special tokens' (0,0)
    if not offs:
        return []
    chunks: list[Chunk] = []
    n_tok = len(offs)
    t = 0
    while t < n_tok:
        t_end = min(t + TARGET_TOKENS, n_tok)
        i = offs[t][0]                                  # chunk start (char)
        end = offs[t_end - 1][1]                        # chunk end (char)
        if t_end < n_tok:
            # Sentence alignment: look for a boundary within `slack` chars past
            # `end` — but never past the next token's start, so the chunk stays
            # under the 128-token limit.
            m = _BOUNDARY.search(text, end, min(end + slack, offs[t_end][1]))
            if m:
                end = m.end()
        seg = text[i:end].replace(FF, " ")
        clean = re.sub(r"[ \t]+", " ", seg).strip()
        # Size floor in TOKENS (a char floor filters CJK out — see MIN_TOKENS note)
        if (t_end - t) >= MIN_TOKENS or (not chunks and clean):
            chunks.append(Chunk(
                file=file_rel,
                page_start=_page_at(marks, i),
                page_end=_page_at(marks, max(i, end - 1)),
                text=clean,
            ))
        if t_end >= n_tok:
            break
        t = max(t_end - OVERLAP_TOKENS, t + 1)
    return chunks


def _chunk_by_chars(text: str, file_rel: str, target: int, overlap: int,
                    min_chars: int, slack: int) -> list[Chunk]:
    """The v0.1 character chunking = FALLBACK ONLY (tokenizer unavailable).
    ⚠ On CJK text this produces >128-token chunks whose tails never enter the
    index (see module docstring); it exists so the tool degrades instead of
    hard-stopping when the tokenizers package is missing."""
    marks = _page_marks(text)
    chunks: list[Chunk] = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + target, n)
        if end < n:  # extend to a sentence boundary so we don't cut mid-sentence
            m = _BOUNDARY.search(text, end, min(end + slack, n))
            if m:
                end = m.end()
        seg = text[i:end].replace(FF, " ")
        clean = re.sub(r"[ \t]+", " ", seg).strip()
        if len(clean) >= min_chars or (not chunks and clean):
            chunks.append(Chunk(
                file=file_rel,
                page_start=_page_at(marks, i),
                page_end=_page_at(marks, max(i, end - 1)),
                text=clean,
            ))
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks
