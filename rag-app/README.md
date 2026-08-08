# RAG document chat

A retrieval-augmented generation app: upload PDF/TXT documents, ask questions,
get answers grounded in the document content with source citations.

## Architecture

- **Backend**: FastAPI (Python)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) — runs locally, no API key needed
- **Vector store**: ChromaDB (persisted to disk in `backend/chroma_db/`)
- **LLM**: Claude (`claude-sonnet-4-6`) via the Anthropic API, for answer generation
- **Frontend**: Single-file vanilla HTML/JS (no build step required)

## Why these choices (for your README / interview talking points)

- **Chunking**: 800-character sliding window with 150-character overlap. Overlap
  prevents losing context at chunk boundaries — a sentence split across two
  chunks still has enough surrounding text in each to remain retrievable.
- **Embedding model**: `all-MiniLM-L6-v2` is small (~90MB), fast on CPU, and free —
  good enough quality for a portfolio project without needing a paid API for embeddings.
- **top_k = 4**: balances giving the LLM enough context vs. diluting it with
  irrelevant chunks. Worth experimenting with and reporting your findings.
- **Grounding**: the prompt explicitly instructs the model to say "I don't know"
  rather than guess when the retrieved context is insufficient — this is what
  separates a real RAG system from a chatbot that just hallucinates confidently.

## Setup

### 1. Prerequisites
- Python 3.10+
- An Anthropic API key (optional — without it, the app still retrieves and
  shows relevant chunks, just without LLM-generated answers)

### 2. Install dependencies

```bash
cd rag-app/backend
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
# then edit .env and paste your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the server

```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the app

Go to **http://localhost:8000** in your browser. The FastAPI server also
serves the frontend directly, so you only need to run one process.

## Usage

1. Upload a PDF or TXT file — it gets chunked, embedded, and stored in Chroma.
2. Ask a question in the chat box.
3. The app retrieves the most relevant chunks and asks Claude to answer using
   only that context. Click "N source chunk(s) used" under any answer to see
   exactly what was retrieved.

## API endpoints

| Method | Path                     | Description                        |
|--------|--------------------------|-------------------------------------|
| GET    | `/health`                | Health check + indexed doc count    |
| POST   | `/upload`                | Upload and index a document         |
| GET    | `/documents`             | List indexed document names         |
| DELETE | `/documents/{name}`      | Remove a document from the index    |
| POST   | `/ask`                   | Ask a question, get grounded answer |

## Extending this project (good next steps for a portfolio)

- **Evaluation**: build a small set of question/answer pairs and measure
  retrieval precision at different `top_k` / chunk sizes — this is a genuinely
  strong thing to show in an interview.
- **Streaming responses**: switch `/ask` to a streaming endpoint using
  Server-Sent Events so answers appear token-by-token.
- **Better chunking**: try semantic/sentence-aware chunking instead of fixed
  character windows (e.g. via `langchain`'s `RecursiveCharacterTextSplitter`).
- **Re-ranking**: add a cross-encoder re-ranker after initial retrieval to
  improve precision on the top-k results.
- **Deploy**: containerize with Docker and deploy to Render/Railway/HF Spaces.

## Project structure

```
rag-app/
├── backend/
│   ├── main.py           # FastAPI routes
│   ├── rag_engine.py      # Chunking, embedding, retrieval, generation
│   ├── requirements.txt
│   ├── .env.example
│   ├── uploads/            # Uploaded files land here
│   └── chroma_db/          # Persisted vector store (created on first run)
├── frontend/
│   └── index.html          # Upload UI + chat UI
└── README.md
```
