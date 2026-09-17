"""Wire the stages together: chunk, embed, store, retrieve, prompt, generate."""

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .chunking import chunk_documents
from .embedders import Embedder, HashingEmbedder
from .generators import ExtractiveGenerator, Generator
from .loaders import load_directory, load_text, load_url
from .store import VectorStore
from .types import Chunk, Document, Retrieved, as_documents

PROMPT_RULES = (
    "Answer the question using only the context below.\n"
    "Cite the source label in square brackets after each claim you make.\n"
    "If the context does not contain the answer, say so plainly instead of guessing."
)


@dataclass
class Answer:
    """What `ask()` returns."""

    text: str
    sources: List[str] = field(default_factory=list)
    retrieved: List[Retrieved] = field(default_factory=list)
    prompt: str = ""

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class RagPipeline:
    """A retrieval augmented generation pipeline with swappable stage objects."""

    def __init__(self, embedder: Optional[Embedder] = None,
                 generator: Optional[Generator] = None,
                 store: Optional[VectorStore] = None,
                 chunk_size: int = 800, chunk_overlap: int = 120,
                 min_chunk_chars: int = 80, top_k: int = 4,
                 min_score: float = 0.0) -> None:
        # explicit None checks: an empty VectorStore is falsy, so `x or default`
        # would quietly discard a caller supplied store
        self.embedder = embedder if embedder is not None else HashingEmbedder()
        # the extractive generator keeps ask() usable with no model installed
        self.generator = generator if generator is not None else ExtractiveGenerator()
        self.store = store if store is not None else VectorStore()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.top_k = top_k
        self.min_score = min_score

    def __len__(self) -> int:
        return len(self.store)

    # -- indexing ---------------------------------------------------------

    def add_documents(self, documents: Iterable[Document]) -> int:
        """Chunk and embed documents. Returns the number of chunks added."""
        chunks: List[Chunk] = chunk_documents(
            list(documents), max_chars=self.chunk_size,
            overlap=self.chunk_overlap, min_chars=self.min_chunk_chars,
        )
        if not chunks:
            return 0
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def add_texts(self, texts: Sequence[str], sources: Optional[Sequence[str]] = None) -> int:
        if sources is not None and len(sources) != len(texts):
            raise ValueError("sources must be the same length as texts")
        documents = as_documents(list(texts))
        if sources is not None:
            for document, source in zip(documents, sources):
                document.source = source
        return self.add_documents(documents)

    def index_file(self, path: str) -> int:
        return self.add_documents([load_text(path)])

    def index_directory(self, path: str, glob: str = "*.md", recursive: bool = False) -> int:
        return self.add_documents(load_directory(path, glob=glob, recursive=recursive))

    def index_url(self, url: str) -> int:
        return self.add_documents([load_url(url)])

    # -- querying ---------------------------------------------------------

    def retrieve(self, question: str, top_k: Optional[int] = None,
                 min_score: Optional[float] = None) -> List[Retrieved]:
        if not len(self.store):
            return []
        vector = self.embedder.embed([question])[0]
        return self.store.search(
            vector,
            top_k=self.top_k if top_k is None else top_k,
            min_score=self.min_score if min_score is None else min_score,
        )

    def build_prompt(self, question: str, retrieved: Sequence[Retrieved]) -> str:
        """Format the context block. Numbered blocks are labelled with their source."""
        lines = [PROMPT_RULES, "", "Context:"]
        for position, item in enumerate(retrieved, start=1):
            lines.append(f"[{position}] source: {item.chunk.label()}")
            lines.append(item.text.strip())
            lines.append("")
        lines.append(f"Question: {question.strip()}")
        lines.append("Answer:")
        return "\n".join(lines)

    def ask(self, question: str, top_k: Optional[int] = None,
            min_score: Optional[float] = None) -> Answer:
        """Retrieve, assemble and generate. Always returns an Answer."""
        retrieved = self.retrieve(question, top_k=top_k, min_score=min_score)
        if not retrieved:
            return Answer(
                text="Nothing in the index matched that question, so there is no answer to give.",
                sources=[], retrieved=[],
            )

        context = "\n\n".join(item.text.strip() for item in retrieved)
        prompt = self.build_prompt(question, retrieved)
        text = self.generator.generate(prompt, context=context)

        sources: List[str] = []
        for item in retrieved:
            if item.source not in sources:
                sources.append(item.source)
        return Answer(text=text, sources=sources, retrieved=retrieved, prompt=prompt)

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        self.store.save(path)

    @classmethod
    def load(cls, path: str, embedder: Optional[Embedder] = None,
             generator: Optional[Generator] = None, **kwargs) -> "RagPipeline":
        """Reload an index. Pass the same embedder you indexed with."""
        return cls(embedder=embedder, generator=generator,
                   store=VectorStore.load(path), **kwargs)

    def describe(self) -> str:
        chunks = len(self.store)
        dim = self.store.dim
        return (f"{chunks} chunks, {dim or 'unknown'} dimensions, "
                f"embedder={type(self.embedder).__name__}, "
                f"generator={type(self.generator).__name__}")
