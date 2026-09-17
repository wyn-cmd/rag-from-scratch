"""Tests for the loaders that can run without network access."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loaders import html_to_text, load_directory, load_text  # noqa: E402

SAMPLE_HTML = """
<html><head><title>Ignored</title>
<script>var tracking = "should not appear";</script>
<style>.hidden { display: none; }</style></head>
<body>
<h1>Observatory</h1>
<p>The dome closes at 23:00.</p>
<p>Red torches only after dark.</p>
</body></html>
"""


class HtmlToTextTests(unittest.TestCase):
    def test_strips_script_and_style(self):
        text = html_to_text(SAMPLE_HTML)
        self.assertNotIn("tracking", text)
        self.assertNotIn("display: none", text)
        self.assertNotIn("<", text)

    def test_keeps_visible_content(self):
        text = html_to_text(SAMPLE_HTML)
        self.assertIn("Observatory", text)
        self.assertIn("The dome closes at 23:00.", text)

    def test_entities_are_decoded(self):
        text = html_to_text("<p>Fish &amp; chips &lt;here&gt;</p>")
        self.assertIn("Fish & chips <here>", text)

    def test_handles_empty_input(self):
        self.assertEqual(html_to_text(""), "")


class FileLoaderTests(unittest.TestCase):
    def test_load_text_sets_the_source_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.md")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Some notes about the dome.")
            document = load_text(path)
        self.assertEqual(document.source, "notes.md")
        self.assertIn("Some notes", document.text)
        self.assertEqual(document.metadata["path"], path)

    def test_load_directory_matches_the_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.md", "b.md", "c.txt"):
                with open(os.path.join(tmp, name), "w", encoding="utf-8") as handle:
                    handle.write(f"content of {name}")
            markdown = load_directory(tmp, glob="*.md")
            everything = load_directory(tmp, glob="*.*")
        self.assertEqual(sorted(d.source for d in markdown), ["a.md", "b.md"])
        self.assertEqual(len(everything), 3)

    def test_load_directory_on_missing_path_returns_nothing(self):
        self.assertEqual(load_directory("/nonexistent/path/does/not/exist"), [])


if __name__ == "__main__":
    unittest.main()
