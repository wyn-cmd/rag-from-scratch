"""Tests for the pipeline end to end, using the offline backends."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedders import HashingEmbedder  # noqa: E402
from rag.generators import ExtractiveGenerator, RecordingGenerator  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from rag.types import Document  # noqa: E402

CORPUS = [
    "Telescope bookings are free for students, who must reserve a day in advance.",
    "The dome closes at 23:00 on every night it is open.",
    "Lightning alerts stand for thirty minutes after the last strike.",
    "Membership costs thirty dollars a year for students.",
]


def build(generator=None, **kwargs):
    pipeline = RagPipeline(embedder=HashingEmbedder(dim=512), generator=generator, **kwargs)
    pipeline.add_texts(CORPUS, sources=[f"doc{i}.md" for i in range(len(CORPUS))])
    return pipeline


class PipelineTests(unittest.TestCase):
    def test_indexing_reports_chunk_count(self):
        pipeline = RagPipeline(embedder=HashingEmbedder(dim=256))
        added = pipeline.add_texts(["one short line", "another short line"])
        self.assertEqual(added, 2)
        self.assertEqual(len(pipeline), 2)

    def test_retrieval_finds_the_matching_source(self):
        pipeline = build()
        results = pipeline.retrieve("how much is student membership", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0].source, "doc3.md")

    def test_answer_carries_text_and_sources(self):
        pipeline = build()
        answer = pipeline.ask("when does the dome close")
        self.assertTrue(answer.text.strip())
        self.assertIn("doc1.md", answer.sources)
        self.assertIn("Question:", answer.prompt)

    def test_generator_receives_the_prompt(self):
        recorder = RecordingGenerator(reply="a short answer")
        pipeline = build(generator=recorder)
        answer = pipeline.ask("dome close time")
        self.assertEqual(answer.text, "a short answer")
        self.assertEqual(len(recorder.prompts), 1)
        self.assertIn("Context:", recorder.prompts[0])

    def test_unanswerable_question_skips_the_generator(self):
        recorder = RecordingGenerator()
        pipeline = build(generator=recorder)
        # nothing in this corpus should clear a high similarity floor
        answer = pipeline.ask("what is the capital of peru", min_score=0.95)
        self.assertEqual(recorder.prompts, [])
        self.assertEqual(answer.sources, [])
        self.assertIn("Nothing in the index", answer.text)

    def test_empty_pipeline_answers_without_crashing(self):
        pipeline = RagPipeline(embedder=HashingEmbedder())
        answer = pipeline.ask("anything at all")
        self.assertIn("Nothing in the index", answer.text)
        self.assertEqual(pipeline.retrieve("anything"), [])

    def test_source_count_must_match_text_count(self):
        pipeline = RagPipeline(embedder=HashingEmbedder())
        with self.assertRaises(ValueError):
            pipeline.add_texts(["one", "two"], sources=["only-one.md"])

    def test_documents_with_metadata_are_accepted(self):
        pipeline = RagPipeline(embedder=HashingEmbedder(dim=128))
        added = pipeline.add_documents([Document(text="Some text here.", source="x.md",
                                                 metadata={"section": "intro"})])
        self.assertEqual(added, 1)
        self.assertEqual(pipeline.store.chunks[0].metadata["section"], "intro")

    def test_save_and_reload_keeps_retrieval_working(self):
        pipeline = build()
        query = "lightning alert duration"
        before = [item.text for item in pipeline.retrieve(query, top_k=2)]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            pipeline.save(path)
            reloaded = RagPipeline.load(path, embedder=HashingEmbedder(dim=512))
            after = [item.text for item in reloaded.retrieve(query, top_k=2)]

        self.assertEqual(before, after)
        self.assertEqual(len(reloaded), len(pipeline))

    def test_extractive_generator_quotes_the_context(self):
        generator = ExtractiveGenerator()
        out = generator.generate("prompt", context="A passage worth quoting.")
        self.assertIn("A passage worth quoting.", out)
        self.assertIn("extractive", out.lower())

    def test_describe_mentions_both_backends(self):
        text = build().describe()
        self.assertIn("HashingEmbedder", text)
        self.assertIn("chunks", text)

    def test_custom_store_can_be_injected(self):
        store = VectorStore()
        pipeline = RagPipeline(embedder=HashingEmbedder(dim=64), store=store)
        pipeline.add_texts(["injected store text"])
        self.assertEqual(len(store), 1)


if __name__ == "__main__":
    unittest.main()
