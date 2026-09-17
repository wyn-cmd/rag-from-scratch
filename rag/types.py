"""Small containers shared by the pipeline stages."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """A source of text before it is split up."""

    text: str
    source: str = "document"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """One retrievable slice of a document."""

    text: str
    source: str
    index: int
    start: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        # stable across runs so a saved index still lines up
        return f"{self.source}#{self.index}"

    def label(self) -> str:
        return f"{self.source} (chunk {self.index})"


@dataclass
class Retrieved:
    """A chunk plus the score it earned against a query."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source


def unique_sources(items: List[Retrieved]) -> List[str]:
    """Source names in the order they were retrieved, without repeats."""
    seen: List[str] = []
    for item in items:
        if item.source not in seen:
            seen.append(item.source)
    return seen


def as_documents(items: List[Any]) -> List[Document]:
    """Accept strings or Documents and return Documents.

    Convenience so callers can pass a plain list of strings without wrapping each
    one, which is what most experiments actually start with.
    """
    documents: List[Document] = []
    for position, item in enumerate(items):
        if isinstance(item, Document):
            documents.append(item)
        elif isinstance(item, str):
            documents.append(Document(text=item, source=f"document-{position}"))
        else:
            raise TypeError(f"expected str or Document, got {type(item).__name__}")
    return documents
