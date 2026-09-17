"""Same pipeline, real embedder and real generator.

Needs the optional stack (pip install -r requirements.txt), which pulls in torch
and transformers. Run it with: python3 examples/demo_local_models.py

Both model names are defaults you can change. The generator default is a small
instruct model so this does not need a GPU, but any causal model works.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import RagPipeline, SentenceTransformerEmbedder, TransformersGenerator  # noqa: E402

CORPUS = [
    ("opening_hours.md",
     "The Northlight Observatory opens at 19:00 on weekdays and at 17:00 on Saturdays. "
     "It closes at 23:00 on every night it is open."),
    ("bookings.md",
     "Telescope bookings are free for students, who must reserve at least one day in "
     "advance. Members of the public pay twelve dollars per session."),
    ("safety.md",
     "Visitors must leave the dome during a lightning alert, and the alert stands for "
     "thirty minutes after the last strike detected within ten kilometres."),
]

QUESTION = "How much does a member of the public pay to use a telescope?"


def main() -> None:
    pipeline = RagPipeline(
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        generator=TransformersGenerator(
            "meta-llama/Llama-3.2-1B-Instruct",
            max_new_tokens=200,
            temperature=0.2,
            # token=os.environ.get("HF_TOKEN"),  # only needed for gated models
        ),
        chunk_size=400,
        chunk_overlap=80,
        top_k=2,
    )

    print("first run downloads the embedding model, then the generator, so give it a minute")
    added = pipeline.add_texts([text for _, text in CORPUS], sources=[name for name, _ in CORPUS])
    print(f"indexed {added} chunks")
    print(pipeline.describe())
    print()

    answer = pipeline.ask(QUESTION)
    print(f"Q: {QUESTION}\n")
    print("A:", answer.text.strip())
    print("\nsources:", ", ".join(answer.sources) or "none")


if __name__ == "__main__":
    main()
