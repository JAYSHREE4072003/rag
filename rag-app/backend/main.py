import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import RAGEngine

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG App API")

# Allow the frontend (served separately or via file://) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": engine.collection.count()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_ext = (".pdf", ".txt", ".md")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed_ext}")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = engine.ingest_document(save_path, file.filename)
    return result


@app.get("/documents")
def list_documents():
    return {"sources": engine.list_sources()}


@app.delete("/documents/{source_name}")
def delete_document(source_name: str):
    deleted = engine.delete_source(source_name)
    return {"source": source_name, "chunks_deleted": deleted}


@app.post("/ask")
def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    result = engine.generate_answer(req.question)
    return result


# Serve the frontend directly from FastAPI so you only need to run one server
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
