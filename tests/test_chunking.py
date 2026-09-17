"""Tests for chunking. No network, no model downloads."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunking import chunk_document, chunk_text, split_paragraphs  # noqa: E402
from rag.types import Document  # noqa: E402


class ChunkTextTests(unittest.TestCase):
    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\n  "), [])

    def test_no_chunk_exceeds_the_limit(self):
        paragraph = " ".join(f"word{i}" for i in range(400))
        text = "\n\n".join([paragraph] * 3)
        for limit in (120, 300, 800):
            chunks = chunk_text(text, max_chars=limit, overlap=limit // 6)
            self.assertTrue(chunks, f"no chunks produced for limit {limit}")
            for chunk in chunks:
                self.assertLessEqual(len(chunk), limit, f"chunk over {limit} chars")

    def test_short_text_stays_one_chunk(self):
        chunks = chunk_text("A single short sentence.", max_chars=500)
        self.assertEqual(chunks, ["A single short sentence."])

    def test_paragraphs_split_when_they_do_not_fit_together(self):
        # small paragraphs are packed together up to the limit, so force a split
        chunks = chunk_text("First paragraph here.\n\nSecond paragraph here.",
                            max_chars=30, overlap=5)
        self.assertEqual(len(chunks), 2)
        self.assertIn("First", chunks[0])
        self.assertIn("Second", chunks[1])

    def test_small_paragraphs_pack_into_one_chunk(self):
        chunks = chunk_text("First paragraph here.\n\nSecond paragraph here.",
                            max_chars=200, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertIn("First", chunks[0])
        self.assertIn("Second", chunks[0])

    def test_overlap_shares_text_between_neighbours(self):
        # two paragraphs that only fit together as separate chunks
        first = "alpha " * 20
        second = "bravo " * 20
        chunks = chunk_text((first.strip() + "\n\n" + second.strip()), max_chars=140, overlap=40)
        self.assertGreater(len(chunks), 1)
        words_a = set(chunks[0].split())
        words_b = set(chunks[1].split())
        self.assertTrue(words_a & words_b, "consecutive chunks share no words despite overlap")

    def test_long_unbroken_text_is_split(self):
        chunks = chunk_text("x" * 3000, max_chars=500, overlap=100)
        self.assertEqual(sum(len(chunk) for chunk in chunks), 3000)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 500)

    def test_runt_final_chunk_is_merged(self):
        text = ("a" * 200) + "\n\nshort tail."
        chunks = chunk_text(text, max_chars=300, overlap=10, min_chars=80)
        self.assertEqual(len(chunks), 1)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            chunk_text("text", max_chars=0)
        with self.assertRaises(ValueError):
            chunk_text("text", max_chars=100, overlap=-1)
        with self.assertRaises(ValueError):
            chunk_text("text", max_chars=100, overlap=100)

    def test_chunk_document_records_source_and_index(self):
        document = Document(text="one two three\n\nfour five six", source="notes.md")
        chunks = chunk_document(document, max_chars=40, overlap=5)
        self.assertTrue(chunks)
        for position, chunk in enumerate(chunks):
            self.assertEqual(chunk.source, "notes.md")
            self.assertEqual(chunk.index, position)
            self.assertTrue(chunk.chunk_id.startswith("notes.md#"))

    def test_split_paragraphs_ignores_blank_runs(self):
        self.assertEqual(split_paragraphs("a\n\n\n\nb\n  \n c"), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
