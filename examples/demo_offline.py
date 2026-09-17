"""Offline demo: index a small corpus and answer questions with no downloads.

Run it with: python3 examples/demo_offline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import ExtractiveGenerator, HashingEmbedder, RagPipeline  # noqa: E402

CORPUS = [
    ("opening_hours.md",
     "The Northlight Observatory opens at 19:00 on weekdays and at 17:00 on Saturdays. "
     "It closes at 23:00 on every night it is open. The dome stays closed through the "
     "monsoon season in December and January."),
    ("bookings.md",
     "Telescope bookings are free for students, who must reserve at least one day in "
     "advance. Members of the public pay twelve dollars per session. A booking is "
     "cancelled automatically when the cloud cover forecast goes above eighty percent."),
    ("instruments.md",
     "The main instrument is a 0.6 metre reflector installed in 2019. A solar telescope "
     "is available for daytime sessions. Both instruments share one booking queue, and "
     "the solar telescope cannot be reserved more than three days ahead."),
    ("membership.md",
     "Membership runs for a calendar year and includes unlimited weekday sessions. "
     "Members also get early access to the monthly public viewing night, plus a "
     "discount on the annual astrophotography workshop.\n\n"
     "Student membership costs thirty dollars a year and requires a valid student card "
     "at signup. The workshop discount for students is fifty percent. Membership does "
     "not cover instrument rental for private groups, which is quoted separately."),
    ("safety.md",
     "Visitors must leave the dome during a lightning alert, and the alert stands for "
     "thirty minutes after the last strike detected within ten kilometres. No white "
     "lights are allowed after dark. Red torches are provided at the door."),
]

QUESTIONS = [
    ("Are telescope bookings free for students?", 0.0),
    ("What does student membership cost?", 0.0),
    ("How much does a student pay to reserve a slot?", 0.0),
    ("Does the observatory run a planetarium?", 0.20),
]


def main() -> None:
    pipeline = RagPipeline(
        embedder=HashingEmbedder(dim=1024),
        generator=ExtractiveGenerator(),
        chunk_size=300,
        chunk_overlap=60,
        top_k=2,
    )

    added = pipeline.add_texts([text for _, text in CORPUS], sources=[name for name, _ in CORPUS])
    print(f"indexed {added} chunks from {len(CORPUS)} documents")
    print(pipeline.describe())
    print()

    for question, floor in QUESTIONS:
        answer = pipeline.ask(question, min_score=floor)
        label = f"  (min_score {floor})" if floor else ""
        print(f"Q: {question}{label}")
        if answer.retrieved:
            top = answer.retrieved[0]
            print(f"   best match: {top.chunk.label()} score {top.score:.3f}")
        print(f"   sources: {', '.join(answer.sources) if answer.sources else 'none'}")
        print(f"   answer: {answer.text.strip().splitlines()[0][:120]}")
        print()

    print("The first two questions share vocabulary with the corpus, so keyword style")
    print("matching finds them. The third asks the same thing in different words and")
    print("scores far lower, which is exactly what real embeddings are for. The fourth")
    print("has no answer in the corpus at all, and the minimum score is what stops it")
    print("from returning a confident passage about something else.")


if __name__ == "__main__":
    main()
