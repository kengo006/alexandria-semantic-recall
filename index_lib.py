"""Chunking + page mapping.

The text layer (one .txt per source PDF) can carry page boundaries in two formats:
  - born-digital : pdftotext -layout output, page break = form feed \\f (0x0C);
                   page number = count of preceding \\f + 1
  - scanned/OCR  : explicit markers  ===== page N =====
This module absorbs both, cutting the text into ~500-character, sentence-aligned
chunks, each carrying {relative file path, start page, end page}.

Small chunks are deliberate: the default embedding model has max_seq ≈ 128 tokens,
and the consumer is a read-only search agent that wants a *precise pointer* to a
passage — it then goes back to the source PDF for context. Recall-only, never a
citation source (see alexandria, optional-integrations §2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

FF = "\f"
_OCR_PAGE = re.compile(r"=====\s*page\s+(\d+)\s*=====")
_BOUNDARY = re.compile(r"[.!?。！？；;\n]")


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
    """Sliding-window chunking; window tail snaps to the nearest sentence boundary
    within `slack`. Page numbers are back-computed from original-text offsets."""
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
