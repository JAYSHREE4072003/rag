# 📄 RAG Document Chat

A full-stack **Retrieval-Augmented Generation (RAG)** application — upload documents, ask questions, and get answers grounded in the document content with source citations. Built to demonstrate an end-to-end understanding of the RAG pipeline: chunking, embeddings, vector search, and LLM-grounded generation.

> Ask questions about your own PDFs/text files and get answers with the exact source chunks cited — not a black box, not a hallucination-prone chatbot.

---

## Why this project

Most ML portfolios stop at a notebook that trains a model. This project goes further: it's a working system with a real retrieval pipeline, an API, a UI, and a way to *measure* whether retrieval is actually working — the same trade-offs production RAG systems deal with.

## Features

- Upload PDF / TXT / Markdown documents
-  Automatic chunking with configurable size and overlap
-  Local embedding generation (no API cost) via `sentence-transformers`
-  Persistent vector storage with ChromaDB
-  Grounded answer generation via the Claude API, with explicit "I don't know" behavior when context is insufficient
-  Every answer shows exactly which source chunks it was generated from
-  Built-in retrieval evaluation script (precision/recall/hit-rate @ k)
-  Simple web UI — no frontend build step required

## Architecture

```
 Upload                                          Ask a question
   │                                                    │
   ▼                                                    ▼
 Extract text (PDF/TXT)                          React-free HTML/JS UI
   │                                                    │
   ▼                                                    ▼
 Chunk (sliding window, overlap)                 FastAPI  /ask endpoint
   │                                                    │
   ▼                                                    ▼
 Embed (sentence-transformers)  ──────────►  Similarity search (ChromaDB)
   │                                                    │
   ▼                                                    ▼
 Store in ChromaDB                                Top-k relevant chunks
                                                          │
                                                          ▼
                                              Claude generates grounded answer
                                                     with citations
```

## Tech stack

| Layer            | Choice                                  |
|-------------------|------------------------------------------|
| Backend           | FastAPI (Python)                        |
| Embeddings        | `sentence-transformers` (`all-MiniLM-L6-v2`) — local, free |
| Vector store      | ChromaDB (persisted to disk)            |
| LLM               | Claude (`claude-sonnet-4-6`) via Anthropic API |
| Frontend          | Vanilla HTML/JS (no build tooling)      |

## Demo

*(Add a screenshot or short screen recording of the app here once you have one — this matters a lot for a portfolio. A 20-second GIF of uploading a doc and asking a question is worth more than paragraphs of description.)*

## Getting started

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (optional — without it, the app still shows retrieved chunks, just without LLM-generated answers)

### Installation

```bash
git clone https://github.com/<your-username>/rag-app.git
cd rag-app/backend
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# then edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the backend also serves the frontend, so this is the only process you need to run.

## Usage

1. Upload a PDF or TXT file — it's chunked, embedded, and indexed automatically.
2. Ask a question about it in the chat box.
3. The app retrieves the most relevant chunks and asks Claude to answer using only that context. Click "N source chunk(s) used" under any answer to see exactly what was retrieved and used.

## API reference

| Method | Path                | Description                          |
|--------|----------------------|----------------------------------------|
| GET    | `/health`            | Health check + indexed document count  |
| POST   | `/upload`             | Upload and index a document            |
| GET    | `/documents`          | List indexed document names            |
| DELETE | `/documents/{name}`   | Remove a document from the index       |
| POST   | `/ask`                | Ask a question, get a grounded answer  |

## Design decisions

- **Chunking (800 chars, 150 overlap):** overlap prevents losing context at chunk boundaries — a sentence split across two chunks still has enough surrounding text in each to remain retrievable.
- **Embedding model:** `all-MiniLM-L6-v2` is small (~90MB), fast on CPU, and free — good enough quality for this scope without needing a paid embeddings API.
- **top_k = 4:** balances giving the LLM enough context vs. diluting it with irrelevant chunks.
- **Explicit grounding instruction:** the prompt tells the model to say it doesn't know rather than guess when retrieved context is insufficient — this is what separates a real RAG system from a chatbot that hallucinates confidently.

## Evaluating retrieval quality

The project includes `backend/eval.py` — a small script that measures **hit rate** and **top-1 keyword match** at different values of k, using a hand-labeled set of (question → expected source) pairs.

```bash
python eval.py
```

```
k    Hit rate       Top-1 keyword match
------------------------------------------
1    73.33%          66.67%
3    93.33%          66.67%
5    100.00%         66.67%
```

This turns "I built a RAG app" into "I measured retrieval precision and recall at different k values and iterated on chunking strategy based on the results" — a much stronger thing to say in an interview.

## Running in Jupyter / Anaconda

See `notebook/RAG_Demo.ipynb` for an interactive walkthrough — import the engine directly for experimentation, run the evaluation cell-by-cell, or launch the full web app from within a notebook using `nest_asyncio`.

## Project structure

```
rag-app/
├── backend/
│   ├── main.py            # FastAPI routes
│   ├── rag_engine.py      # Chunking, embedding, retrieval, generation
│   ├── eval.py             # Retrieval evaluation script
│   ├── requirements.txt
│   ├── .env.example
│   ├── uploads/             # Uploaded files land here
│   └── chroma_db/           # Persisted vector store (created on first run)
├── frontend/
│   └── index.html            # Upload UI + chat UI
├── notebook/
│   └── RAG_Demo.ipynb         # Jupyter walkthrough
├── .gitignore
└── README.md
```

