"""
Core RAG (Retrieval-Augmented Generation) engine.

Responsibilities:
1. Ingest documents (PDF or TXT) -> extract text -> chunk -> embed -> store in Chroma
2. Answer questions -> embed query -> retrieve top-k chunks -> call LLM with context
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import uuid
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs locally, no API key needed
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
TOP_K = 4               # number of chunks retrieved per question

ANTHROPIC_MODEL = "claude-sonnet-4-6"


class RAGEngine:
    def __init__(self):
        # Local embedding function (downloads model once, then runs on CPU/GPU locally — free)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm_client = Anthropic(api_key=api_key) if api_key else None

    # ---------- Ingestion ----------

    def extract_text(self, file_path: str) -> str:
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def chunk_text(self, text: str) -> List[str]:
        """Simple sliding-window character chunking with overlap."""
        text = " ".join(text.split())  # normalize whitespace
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def ingest_document(self, file_path: str, source_name: str) -> Dict:
        text = self.extract_text(file_path)
        chunks = self.chunk_text(text)

        if not chunks:
            return {"source": source_name, "chunks_added": 0, "warning": "No extractable text found."}

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)

        return {"source": source_name, "chunks_added": len(chunks)}

    def list_sources(self) -> List[str]:
        data = self.collection.get()
        sources = {m["source"] for m in data.get("metadatas", []) if m}
        return sorted(sources)

    def delete_source(self, source_name: str) -> int:
        data = self.collection.get(where={"source": source_name})
        ids = data.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    # ---------- Retrieval + Generation ----------

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
            })
        return chunks

    def build_prompt(self, question: str, chunks: List[Dict]) -> str:
        context_block = "\n\n".join(
            f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
            for c in chunks
        )
        return (
            "You are a helpful assistant answering questions using ONLY the "
            "context provided below. If the context does not contain enough "
            "information to answer, say so clearly instead of guessing.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and cite which source(s) you used."
        )

    def generate_answer(self, question: str) -> Dict:
        if self.collection.count() == 0:
            return {
                "answer": "No documents have been uploaded yet. Please upload a document first.",
                "sources": [],
            }

        chunks = self.retrieve(question)

        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents.",
                "sources": [],
            }

        prompt = self.build_prompt(question, chunks)

        if self.llm_client is None:
            return {
                "answer": (
                    "[No ANTHROPIC_API_KEY set — showing retrieved context only]\n\n"
                    + "\n\n---\n\n".join(c["text"] for c in chunks)
                ),
                "sources": chunks,
            }

        response = self.llm_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return {"answer": answer_text, "sources": chunks}
