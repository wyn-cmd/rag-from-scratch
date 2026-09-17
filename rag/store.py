"""Cosine similarity search over chunk vectors, with JSON persistence."""

import json
import math
import os
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
    """In memory store of chunks and their vectors.

    Brute force cosine over every chunk. That is the whole index: no ANN, no
    quantisation. Comfortable to a few thousand chunks, and the place to swap in
    something smarter when it stops being comfortable.
    """

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

    def search(self, query_vector: Sequence[float], top_k: int = 4,
               min_score: float = 0.0) -> List[Retrieved]:
        """Best `top_k` chunks above `min_score`, highest first."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = [
            Retrieved(chunk=chunk, score=cosine(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored = [item for item in scored if item.score >= min_score]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._chunks = []
        self._vectors = []
        self.dim = None

    def save(self, path: str) -> None:
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
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

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
