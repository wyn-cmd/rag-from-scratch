# Cosine similarity search over chunk vectors, with JSON persistence.

import json
import math
import os
import tempfile
from typing import Iterable, List, Optional, Sequence

from .types import Chunk, Retrieved


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, computed properly so unnormalised vectors still work."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = math.fsum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    # In memory store of chunks and their vectors.
    #
    # Brute force cosine over every chunk. That is the whole index: no ANN, no
    # quantisation. Comfortable to a few thousand chunks, and the place to swap in
    # something smarter when it stops being comfortable.

    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim
        self._chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def add(self, chunks: Iterable[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        chunks = list(chunks)
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for vector in vectors:
            if self.dim is None:
                self.dim = len(vector)
            elif len(vector) != self.dim:
                raise ValueError(
                    f"vector of length {len(vector)} does not match store dim {self.dim}. "
                    "Documents and queries must use the same embedder."
                )
        self._chunks.extend(chunks)
        self._vectors.extend([list(vector) for vector in vectors])

    def search(self, query_vector: Sequence[float], top_k: Optional[int] = 4,
               min_score: float = 0.0) -> List[Retrieved]:
        # Best `top_k` chunks above `min_score`, highest first.
    #
    # Pass `top_k=None` to get everything that clears the floor, which is
    # useful when inspecting an index rather than answering a question.
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive, or None for everything")
        scored = [
            Retrieved(chunk=chunk, score=cosine(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored = [item for item in scored if item.score >= min_score]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored if top_k is None else scored[:top_k]

    # -- clear ------------------------------------------------------------
    #
    # Reset the index.
    def clear(self) -> None:
        self._chunks = []
        self._vectors = []
        self.dim = None

    def remove_source(self, source: str) -> int:
        # Drop every chunk from one source. Returns how many were removed.
        #
        # Re-adding a changed file without this first would leave the old
        # chunks in the store alongside the new ones: the stale text keeps
        # matching queries and the index only grows, never corrects itself.
        keep_chunks: List[Chunk] = []
        keep_vectors: List[List[float]] = []
        removed = 0
        for chunk, vector in zip(self._chunks, self._vectors):
            if chunk.source == source:
                removed += 1
            else:
                keep_chunks.append(chunk)
                keep_vectors.append(vector)
        self._chunks = keep_chunks
        self._vectors = keep_vectors
        return removed

    def save(self, path: str) -> None:
        # Save index to a JSON file, using an atomic rename to prevent
        # truncation on failure.
        payload = {
            "version": 1,
            "dim": self.dim,
            "entries": [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "index": chunk.index,
                    "start": chunk.start,
                    "metadata": chunk.metadata,
                    "vector": vector,
                }
                for chunk, vector in zip(self._chunks, self._vectors)
            ],
        }
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        # Written to a temp file in the same directory and swapped into place with
        # os.replace, which POSIX and Windows both guarantee is atomic. Without this
        # a save interrupted partway (killed process, full disk) leaves a truncated
        # JSON file where the index used to be, and the next load fails on an index
        # that looked fine seconds earlier.
        fd, temp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

    def stats(self) -> dict:
        # Return a report on the current state of the index.
        return {
            "source_count": len({c.source for c in self._chunks}),
            "chunk_count": len(self._chunks),
            "avg_chunk_length": sum(len(c.text) for c in self._chunks) / len(self._chunks) if self._chunks else 0.0,
            "embedding_dimension": self.dim,
        }

    # -- load -------------------------------------------------------------

    #
    # Reload index from JSON file.
    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        store = cls(dim=payload.get("dim"))
        for entry in payload.get("entries", []):
            chunk = Chunk(
                text=entry["text"],
                source=entry["source"],
                index=entry["index"],
                start=entry.get("start", 0),
                metadata=entry.get("metadata", {}),
            )
            store.add([chunk], [entry["vector"]])
        return store
