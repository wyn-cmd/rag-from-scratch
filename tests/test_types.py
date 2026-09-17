"""Tests for the shared containers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.types import Chunk, Document, Retrieved, as_documents, unique_sources  # noqa: E402


class AsDocumentsTests(unittest.TestCase):
    def test_strings_become_numbered_documents(self):
        documents = as_documents(["one", "two"])
        self.assertEqual([d.source for d in documents], ["document-0", "document-1"])
        self.assertEqual(documents[1].text, "two")

    def test_existing_documents_pass_through_unchanged(self):
        original = Document(text="kept", source="mine.md", metadata={"a": 1})
        self.assertIs(as_documents([original])[0], original)

    def test_other_types_are_rejected(self):
        with self.assertRaises(TypeError):
            as_documents([42])


class ChunkTests(unittest.TestCase):
    def test_chunk_id_combines_source_and_index(self):
        self.assertEqual(Chunk(text="x", source="a.md", index=3).chunk_id, "a.md#3")

    def test_label_names_the_chunk(self):
        self.assertEqual(Chunk(text="x", source="a.md", index=1).label(), "a.md (chunk 1)")


class RetrievedTests(unittest.TestCase):
    def test_exposes_the_underlying_chunk(self):
        item = Retrieved(chunk=Chunk(text="body", source="a.md", index=0), score=0.5)
        self.assertEqual(item.text, "body")
        self.assertEqual(item.source, "a.md")
        self.assertAlmostEqual(item.score, 0.5)

    def test_unique_sources_keeps_order_and_drops_repeats(self):
        items = [
            Retrieved(chunk=Chunk(text="a", source="a.md", index=0), score=0.9),
            Retrieved(chunk=Chunk(text="b", source="b.md", index=0), score=0.8),
            Retrieved(chunk=Chunk(text="c", source="a.md", index=1), score=0.7),
        ]
        self.assertEqual(unique_sources(items), ["a.md", "b.md"])


if __name__ == "__main__":
    unittest.main()
