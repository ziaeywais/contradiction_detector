# Contradiction Detector

**GitHub:** [github.com/ziaeywais/contradiction_detector](https://github.com/ziaeywais/contradiction_detector)

Most document tools tell you what your documents say. This one tells you where they disagree with each other.

## The Problem

Companies usually don't break because they lack documentation, they break because they have too much of it, and over time it drifts out of sync.

Think about an HR policy from 2023 that says employees get 15 vacation days. In 2024, a new memo updates it to 20 days. Both are live on the company intranet. Nobody notices until someone complains.

Or take a more serious example: a general medical guideline states ibuprofen can be dosed up to 3,200mg/day, but a geriatric supplement buried on another page caps it at 1,200mg for patients over 65. If a clinician reads the wrong document, that's a real problem.

Finding these discrepancies manually is pretty much impossible. The contradictions are rarely next to each other—they hide in Section 4.2 of one PDF and Appendix B of a Word doc. I built this to automate that process.

---

## How It Works

It's not a keyword search. It's a two-stage pipeline:

1. **Local Vector Retrieval:** We split documents into overlapping chunks (~1,200 characters each) and generate embeddings using `sentence-transformers`. This runs entirely on your machine, no API costs here. The embeddings go into a local ChromaDB index, and we do a nearest-neighbor search to find pairs of chunks from *different* documents that are semantically similar (i.e., they're talking about the same topic).
2. **LLM Logic Check:** Those similar pairs get passed to an LLM, not to summarize, but to act as a strict logic engine. It's asking: does Statement A make Statement B impossible? It's smart enough to see that "Employees work remotely" and "Engineers work on-site" is probably a scope exception, not a real contradiction.
3. **The Report:** Findings get ranked by severity,CRITICAL (safety/legal), MAJOR (operational), MINOR (administrative), and surfaced in the dashboard.

If you're not technical: think of it like an analytical librarian who reads every document, groups the related paragraphs, reads them side-by-side for conflicts, and hands you a prioritized list of what needs fixing.

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/ziaeywais/contradiction_detector.git
cd contradiction_detector

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# Launch the app
streamlit run contradiction_detector.py
```

Opens at **http://localhost:8501**. Fair warning: the first run takes a minute or two while the embedding model downloads. One-time thing.

---

## Using the App

### Loading documents

Drag and drop your files into the sidebar (`.txt`, `.md`, `.pdf`, `.docx`). You need at least two—contradictions only exist between documents, not within a single one.

If you already have a large folder of documents on disk, switch the sidebar to **Directory Path** and point it there instead. Easier than uploading 30 files one by one.

**I'd suggest running the sample corpus first** before you throw your own docs at it. Click *🧪 Try with sample corpus* in the sidebar—it loads 6 demo documents with 8 embedded contradictions across HR policies, software contracts, and medical guidelines. Good way to calibrate what a real finding looks like.

### Picking a provider

| Provider | Cost | When I'd use it |
|----------|------|-----------------|
| **Claude** (Default) | ~$0.01 / pair | Anything where accuracy matters. Opus 4.7 is genuinely good at parsing dense policy language. |
| **Ollama** | Free | Offline environments, sensitive documents, or when the contradictions are obvious. |
| **Both** | ~$0.003 / pair | Large document dumps. Ollama pre-filters ~80% of the noise for free; Claude only sees the hard cases. |

Honestly, *Both* is what I'd use for any real workload. You get Claude-level accuracy on the pairs that actually need it, and you're not burning API budget on easy rejects.

### Tuning the scan

The defaults are fine for most cases. The two settings I actually touch:

- **Min similarity (default: 0.65)** — how similar two passages need to be before they're considered candidates. Drop it toward `0.55` if you want a wider net; you'll get more noise. Push it toward `0.75` if you're drowning in irrelevant pairs.
- **Max pairs (default: 50)** — a cost cap. For a thorough scan, raise it. For a quick pass, leave it.

The rest (chunk neighbours, vector DB reset) are there if you need them. You probably won't.

### Reading the results

Each finding shows the two conflicting excerpts side-by-side with severity, confidence, and an explanation of why it was flagged. There are three tabs: **All** (sorted by severity), **By Document Pair** (grouped by which two docs are fighting), and **Summary Table** if you want a flat view to skim.

One thing I want to be upfront about: the model will sometimes flag things that look like contradictions but are actually intentional—a global policy with a regional override, for instance. You still have to read the list and decide. It's a shortlist, not a verdict. The time savings come from not having to find these yourself; the judgment call is still yours.

Export everything as **JSON** or **CSV** from the bottom of the page when you're done.

---

## Limitations

It's not perfect. Here's what it won't catch:

- **Apples and Oranges:** If two conflicting documents use completely different vocabulary, the vector search won't pair them up. There needs to be enough semantic overlap to trigger the retrieval in the first place.
- **Missing World Knowledge:** If Doc A says "Price is Prime + 2%" and Doc B says "Price is 7.5%", the model doesn't know today's Prime rate, so it won't flag it as a conflict.
- **The Exception vs. Contradiction gray area:** A global policy overriding a regional one looks like a contradiction. It might be intentional. Context matters, and you still have to read the final output to make that call.

---

## Extending It

I tried to keep the code modular. Some easy entry points if you want to hack on it:

- **Add file types (PDFs/Word):** Modify `load_document()` in `src/ingestion.py`.
- **Swap the vector model:** Change `model_name` in `VectorStore.__init__()` — any `sentence-transformers` model will work.
- **Add OpenAI or Gemini:** Write an `analyze_pair_with_<name>()` function in `src/detector.py`.
- **Change what counts as a contradiction:** Tweak the system prompt in `src/detector.py`.
