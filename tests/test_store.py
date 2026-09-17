"""Tests for the vector store. No network, no model downloads."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedders import HashingEmbedder  # noqa: E402
from rag.store import VectorStore, cosine  # noqa: E402
from rag.types import Chunk  # noqa: E402


def chunk(text, source="doc.md", index=0):
    return Chunk(text=text, source=source, index=index)


class CosineTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, places=6)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_zero_vector_scores_zero(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine([1.0, 2.0], [1.0])


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=256)
        self.store = VectorStore()
        texts = [
            "Telescope bookings are free for students.",
            "The dome closes at 23:00 on open nights.",
            "Lightning alerts last thirty minutes.",
        ]
        self.store.add([chunk(t, index=i) for i, t in enumerate(texts)],
                       self.embedder.embed(texts))

    def test_search_returns_best_match_first(self):
        results = self.store.search(self.embedder.embed(["student telescope bookings"])[0], top_k=3)
        self.assertTrue(results)
        self.assertLessEqual(len(results), 3)
        self.assertIn("bookings", results[0].text)
        scores = [item.score for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_negative_similarity_is_dropped_by_default(self):
        # a chunk whose vector opposes the query scores -1 and must not reach the
        # context block, since the default floor is 0.0
        store = VectorStore()
        store.add([chunk("opposite"), chunk("aligned")],
                  [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        results = store.search([1.0, 0.0, 0.0], top_k=5)
        self.assertEqual([item.text for item in results], ["aligned"])

    def test_a_floor_of_zero_keeps_zero_scoring_chunks(self):
        # documented behaviour: the floor is inclusive, so an orthogonal chunk
        # (score 0.0) still comes back
        store = VectorStore()
        store.add([chunk("orthogonal")], [[0.0, 1.0, 0.0]])
        results = store.search([1.0, 0.0, 0.0], top_k=5)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 0.0, places=6)

    def test_top_k_is_respected(self):
        results = self.store.search(self.embedder.embed(["dome"])[0], top_k=1)
        self.assertEqual(len(results), 1)

    def test_min_score_filters_results(self):
        results = self.store.search(self.embedder.embed(["completely unrelated words"])[0],
                                    top_k=3, min_score=0.9)
        self.assertEqual(results, [])

    def test_nothing_indexed_returns_nothing(self):
        self.assertEqual(VectorStore().search(self.embedder.embed(["anything"])[0]), [])

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.store.add([chunk("new text")], [[0.0] * 99])

    def test_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.store.add([chunk("a"), chunk("b")], self.embedder.embed(["only one"]))

    def test_invalid_top_k_raises(self):
        with self.assertRaises(ValueError):
            self.store.search(self.embedder.embed(["dome"])[0], top_k=0)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            self.store.save(path)
            reloaded = VectorStore.load(path)

            self.assertEqual(len(reloaded), len(self.store))
            self.assertEqual(reloaded.dim, self.store.dim)

            query = self.embedder.embed(["student telescope bookings"])[0]
            before = self.store.search(query, top_k=2)
            after = reloaded.search(query, top_k=2)
            self.assertEqual([r.text for r in before], [r.text for r in after])
            self.assertAlmostEqual(before[0].score, after[0].score, places=6)

    def test_clear_empties_the_store(self):
        self.store.clear()
        self.assertEqual(len(self.store), 0)
        self.assertIsNone(self.store.dim)


if __name__ == "__main__":
    unittest.main()
