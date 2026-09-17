"""A retrieval-augmented generation pipeline with swappable stages.

The four stages are independent objects rather than one framework call:

    embedder  ->  store  ->  generator
    chunking feeds all three from the loader side

Two backends are provided for each stage that needs one: a pure standard library
fallback that runs anywhere, and a model backed version for real use.
"""

from .chunking import chunk_text
from .embedders import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .generators import ExtractiveGenerator, Generator, TransformersGenerator
from .loaders import load_directory, load_text, load_url
from .pipeline import Answer, RagPipeline
from .store import VectorStore
from .types import Chunk, Document, Retrieved

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "Embedder",
    "ExtractiveGenerator",
    "Generator",
    "HashingEmbedder",
    "RagPipeline",
    "Retrieved",
    "SentenceTransformerEmbedder",
    "TransformersGenerator",
    "VectorStore",
    "chunk_text",
    "load_directory",
    "load_text",
    "load_url",
]

__version__ = "0.1.0"
