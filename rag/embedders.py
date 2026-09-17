"""Turn text into vectors.

Two embedders, and the difference between them matters:

    HashingEmbedder             no dependencies, matches on vocabulary
    SentenceTransformerEmbedder real embeddings, matches on meaning

Both expose `dim` and `embed(texts) -> list[list[float]]`, so the store does not
care which one produced the vectors. Whichever you choose, the same instance has
to embed both the documents and the queries, otherwise the similarity scores are
comparing two different coordinate systems.
"""

import hashlib
import math
import re
from typing import List, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "if", "in", "into", "is", "it", "its", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they", "this", "to", "was",
    "were", "will", "with",
}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stopwords dropped, plus adjacent word pairs."""
    words = [w for w in _TOKEN.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class Embedder:
    """Base class documenting the interface: a `dim` plus `embed(texts)`."""

    dim: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Hash tokens into fixed buckets and normalise. Behaves like bag of words.

    Cheap, deterministic and dependency free, which makes it good for tests and
    for running the pipeline without a multi gigabyte download. It matches shared
    vocabulary only: ask a question with none of the document's words in it and
    the score collapses, which is the limitation real embeddings exist to solve.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            # signed hashing keeps colliding tokens from always reinforcing
            sign = 1.0 if (value >> 17) & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(math.fsum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vector(text or "") for text in texts]


class SentenceTransformerEmbedder(Embedder):
    """Real sentence embeddings through sentence-transformers.

    Defaults to all-MiniLM-L6-v2: 384 dimensions, small enough to run on a laptop
    CPU and good enough for retrieval over prose. Imported lazily so the package
    works without torch installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32,
                 normalize: bool = True) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install the optional stack with: pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        size = model.get_sentence_embedding_dimension()
        return int(size)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._load()
        vectors = model.encode(list(texts), batch_size=self.batch_size,
                               normalize_embeddings=self.normalize,
                               show_progress_bar=False)
        return [[float(value) for value in vector] for vector in vectors]
