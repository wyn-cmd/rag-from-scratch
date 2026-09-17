"""Split text into retrievable chunks.

Paragraph boundaries come first because they usually line up with topic changes.
Anything too long for one chunk gets broken on sentence boundaries, and a single
sentence longer than the limit is hard wrapped so no chunk can ever exceed it.
"""

import re
from typing import Iterable, List

from .types import Chunk, Document

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split_paragraphs(text: str) -> List[str]:
    return [part.strip() for part in _PARAGRAPH.split(text) if part.strip()]


def split_sentences(paragraph: str) -> List[str]:
    return [part.strip() for part in _SENTENCE.split(paragraph) if part.strip()]


def _hard_wrap(text: str, size: int) -> List[str]:
    words = text.split()
    if not words:
        # no whitespace at all, so cut it and accept the ugly break
        return [text[i:i + size] for i in range(0, len(text), size)]

    out: List[str] = []
    current = ""
    for word in words:
        if len(word) > size:
            if current:
                out.append(current)
                current = ""
            out.extend(word[i:i + size] for i in range(0, len(word), size))
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) <= size:
            current = candidate
        else:
            out.append(current)
            current = word
    if current:
        out.append(current)
    return out


def _pieces(text: str, max_chars: int) -> List[str]:
    pieces: List[str] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in split_sentences(paragraph) or [paragraph]:
            pieces.extend(_hard_wrap(sentence, max_chars))
    return pieces


def _tail_from(chunk: str, overlap: int) -> str:
    """Take the last `overlap` characters, starting at a word boundary."""
    if overlap <= 0:
        return ""
    tail = chunk[-overlap:]
    if " " in tail[1:]:
        tail = tail.split(" ", 1)[1]
    return tail.strip()


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120, min_chars: int = 80) -> List[str]:
    """Split text into chunks of at most `max_chars`, with `overlap` carried over.

    `min_chars` is a merge rule rather than a filter: a final chunk shorter than
    it gets folded into the previous one so nothing is dropped and there is no
    orphan fragment to retrieve.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    current = ""
    for piece in _pieces(text, max_chars):
        if not current:
            current = piece
            continue
        candidate = f"{current} {piece}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        # Carry the tail of the finished chunk into the next one, but only as much
        # of it as still fits next to the piece. Without this the overlap silently
        # vanishes whenever a piece is close to the limit, which is the kind of
        # failure that looks like the model simply missing things.
        room = max_chars - len(piece) - 1
        tail = _tail_from(current, min(overlap, room)) if room > 0 else ""
        merged = f"{tail} {piece}".strip() if tail else piece
        current = merged if len(merged) <= max_chars else piece

    if current:
        if chunks and len(current) < min_chars and len(f"{chunks[-1]} {current}") <= max_chars:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)

    return chunks


def chunk_document(document: Document, max_chars: int = 800, overlap: int = 120,
                   min_chars: int = 80) -> List[Chunk]:
    """Chunk one document, recording roughly where each chunk came from."""
    chunks: List[Chunk] = []
    cursor = 0
    for index, text in enumerate(chunk_text(document.text, max_chars, overlap, min_chars)):
        start = document.text.find(text[:40], cursor)
        if start < 0:
            start = cursor
        cursor = start + 1
        chunks.append(Chunk(text=text, source=document.source, index=index,
                            start=start, metadata=dict(document.metadata)))
    return chunks


def chunk_documents(documents: Iterable[Document], max_chars: int = 800, overlap: int = 120,
                    min_chars: int = 80) -> List[Chunk]:
    out: List[Chunk] = []
    for document in documents:
        out.extend(chunk_document(document, max_chars, overlap, min_chars))
    return out
