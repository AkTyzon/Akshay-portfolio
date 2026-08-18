# Agentic Research Assistant

Multi-paper RAG over research papers. Upload PDFs or pull them from arXiv by keyword or ID; the
system indexes the **full paper text**, answers questions across the whole collection, summarises
each paper, and extracts themes spanning all of them.

**[Live demo →](https://136-64-84-253.sslip.io:8444/)**

## How it works
| Step | Module |
|---|---|
| Fetch — arXiv by keyword or ID, or uploaded PDFs | `fetch_papers.py`, `pdf_utils.py` |
| Extract full text from PDF | `pdf_utils.py` (PyMuPDF) |
| Chunk and embed into a per-paper FAISS index | `rag_utils.py` |
| Merge indexes into one searchable store | `rag_utils.merge_faiss` |
| Map-reduce summary per paper | `agent.map_reduce_summary` |
| Q&A across all papers, with sources | `agent.rag_answer` |
| Cross-paper themes | `agent.rag_insights` |

## Running it locally or hosted
The provider is chosen by environment, so the same code runs both ways:

```bash
# local, as originally written — needs Ollama running
streamlit run app.py

# hosted — no GPU required
export GROQ_API_KEY=...
streamlit run app.py
```

`llm_provider.py` is the only module that knows which is active. Nothing else changes.

## Two things worth knowing
**The hosted model is resolved at startup, not pinned.** A pinned model name was retired from the
provider's catalogue and every request started returning 404 — a failure that looks like a bug in
this code rather than a change upstream. The provider list is now queried at startup and a
preference order is walked until something available is found.

**Chat models and LLMs return different types.** `OllamaLLM.invoke()` returns a string;
`ChatGroq.invoke()` returns an `AIMessage`. Responses are normalised before they reach the UI,
otherwise Streamlit renders an object repr instead of the answer.
