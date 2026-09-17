# rag-from-scratch

A small retrieval-augmented generation pipeline written from first principles, with no orchestration framework in the middle. You give it documents, it chunks them, embeds the chunks, retrieves the ones that match a question, and hands the model only those passages as context.

The point of building it this way is that every step is visible and swappable. Chunking, embedding, storage and generation are four separate pieces behind three small interfaces, so you can run the whole thing offline with a toy embedder, then swap in real sentence embeddings and a real language model without touching the pipeline.

## Why not just use a framework

Frameworks are good once you know what you want. For learning, they hide the two decisions that actually determine quality: how you chunk, and what you put in the prompt. Both live in about forty lines of readable code here.

## How the pipeline works

1. **Load.** Text comes from files, a directory, or a URL. HTML is stripped to text with BeautifulSoup when you fetch a page.
2. **Chunk.** Text is split on paragraph boundaries, and paragraphs longer than the limit are split on sentence boundaries, with overlap so a fact that straddles a boundary is still retrievable.
3. **Embed.** Each chunk becomes a vector. The default model-backed embedder is `all-MiniLM-L6-v2` through sentence-transformers, which is small, fast and good enough for retrieval on prose.
4. **Store.** Vectors and chunk text go into an in-memory store that scores by cosine similarity. It saves to JSON so an index survives a restart.
5. **Retrieve.** A question is embedded with the same model and the top-k chunks are pulled, optionally filtered by a minimum score.
6. **Assemble.** The retrieved chunks are formatted into a context block with source labels.
7. **Generate.** The prompt goes to a text-generation model, and the answer comes back with the sources that informed it.

Note step 5: the question and the chunks must be embedded by the *same* model. Mixing embedders produces silently meaningless similarity scores, which is the most common way a from-scratch RAG system goes quietly wrong.

## Layout

```
rag-from-scratch/
├── rag/
│   ├── types.py         # Document, Chunk and Retrieved containers
│   ├── chunking.py      # paragraph and sentence aware splitting with overlap
│   ├── embedders.py     # hashing fallback plus sentence-transformers
│   ├── store.py         # cosine similarity search, JSON persistence
│   ├── generators.py    # extractive fallback plus transformers text generation
│   ├── loaders.py       # files, directories and URLs
│   └── pipeline.py      # wires the stages together and builds the prompt
├── examples/
│   ├── demo_offline.py       # runs with no downloads, uses the fallback backends
│   └── demo_local_models.py  # the real embedder and generator, needs the full install
└── tests/                    # unittest, no network, no model downloads
```

## Setup

The core has no third-party dependencies at all. Everything needed for the offline demo and the tests is in the standard library.

```bash
git clone https://github.com/wyn-cmd/rag-from-scratch.git
cd rag-from-scratch

python3 examples/demo_offline.py       # works immediately
python3 -m unittest discover tests -v  # no downloads, no network
```

For real embeddings and generation, install the optional stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 examples/demo_local_models.py
```

That install pulls in torch and transformers, which is a multi-gigabyte download. The offline path exists so the pipeline can be tested without it.

## Usage

```python
from rag import RagPipeline, HashingEmbedder, ExtractiveGenerator

pipeline = RagPipeline(embedder=HashingEmbedder(), generator=ExtractiveGenerator())
pipeline.add_texts([
    "The observatory opens at 19:00 on weekdays and closes at 23:00.",
    "Telescope bookings are free for students but must be made a day ahead.",
], sources=["opening_hours.md", "bookings.md"])

answer = pipeline.ask("When does the observatory close?")
print(answer.text)
for source in answer.sources:
    print(" ", source)
```

With real models, swap two constructors:

```python
from rag import RagPipeline, SentenceTransformerEmbedder, TransformersGenerator

pipeline = RagPipeline(
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    generator=TransformersGenerator("meta-llama/Llama-3.2-1B-Instruct"),
)
```

Indexing a directory of documents:

```python
pipeline.index_directory("docs/", glob="*.md")
pipeline.save("index.json")     # reuse it later with RagPipeline.load("index.json", ...)
```

## The two fallback backends

**HashingEmbedder** hashes tokens into a fixed number of buckets and normalises the result. It behaves like a bag-of-words vector, so retrieval works on shared vocabulary rather than meaning. Good enough to prove the pipeline end to end and to run tests. It will not match a paraphrase, which is exactly the thing real embeddings are for.

**ExtractiveGenerator** does not generate anything. It returns the highest scoring passage as the answer with a short header. That keeps `ask()` working with no model on disk, and it makes the pipeline testable without asserting on text a model invented.

Both are honest about what they are: shortcuts for development, not replacements for the real thing.

## Chunking decisions

`chunk_text(text, max_chars=800, overlap=120, min_chars=80)` splits on blank lines first, then breaks oversized paragraphs on sentence boundaries while carrying `overlap` characters into the next chunk.

Small chunks retrieve precisely but arrive without context, so the model has to be told more in the prompt. Large chunks carry context but dilute the embedding, since one vector has to represent several ideas. 800 characters with 120 overlap is a reasonable starting point for prose. The knob that matters most in practice is overlap: set it to zero and questions about a fact that spans a boundary start failing, which is a confusing bug to chase.

## Prompt construction

The prompt states the rules plainly and then supplies context:

```
Answer the question using only the context below.
Cite the source in square brackets after each claim you make.
If the context does not contain the answer, say that plainly instead of guessing.

Context:
[1] source: opening_hours.md
The observatory opens at 19:00 ...

Question: When does the observatory close?
Answer:
```

The instruction to admit ignorance matters more than it looks. Without it, a model asked a question the context does not cover will answer from its own weights, and the whole point of retrieval is that it should not.

## Known limits

- Retrieval is a single vector per chunk with no reranking, so a question whose wording shares little vocabulary with the answer will miss.
- No hybrid search, no BM25, no query rewriting. Each of those is a real improvement and each is a small addition to `store.py`.
- The store is in memory plus JSON. Fine for thousands of chunks, wrong for millions, where an ANN index belongs.
- Long documents are chunked independently, so nothing links a chunk back to its neighbours beyond the source label.
- Prompt length is not enforced. A large top_k can blow a small model's context window.

## License

MIT, see `LICENSE`.
