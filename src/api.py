import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any

from src.config import settings
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Swiggy Annual Report RAG API",
    description="A strict, hallucination-free Question Answering system.",
    version="1.0.0"
)

# Mount the static directory for the UI using an absolute path relative to the app config
app.mount("/static", StaticFiles(directory=os.path.join(settings.BASE_DIR, "static")), name="static")

# Mount the data directory to serve the PDF (read-only)
app.mount("/data", StaticFiles(directory=os.path.join(settings.BASE_DIR, "data")), name="data")

# ---------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------
class IngestRequest(BaseModel):
    file_path: Optional[str] = settings.DEFAULT_PDF_PATH
    
class QueryRequest(BaseModel):
    question: str

# ---------------------------------------------------------
# Background Task Handlers
# ---------------------------------------------------------
def process_document_background(file_path: str):
    """
    Background job to prevent blocking the API thread during heavy ingestion.
    """
    print(f"Starting background ingestion for {file_path}")
    processor = DocumentProcessor()
    docs = processor.load_and_process(file_path)
    
    manager = VectorStoreManager()
    manager.create_and_save_index(docs)
    print("Background ingestion completed successfully.")

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["UI"])
async def serve_ui():
    """Serves the static frontend UI."""
    return FileResponse(os.path.join(settings.BASE_DIR, "static", "index.html"))

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    index_exists = os.path.exists(os.path.join(settings.VECTOR_STORE_DIR, "index.faiss"))
    return {
        "status": "healthy",
        "index_ready": index_exists,
        "embedding_model": settings.EMBEDDING_MODEL
    }

@app.post("/ingest", tags=["Ingestion"])
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers parsing and indexing of the specified PDF.
    Runs in the background to avoid timeouts on large documents.
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail=f"File not found at: {request.file_path}")
        
    background_tasks.add_task(process_document_background, request.file_path)
    
    return {"message": "Ingestion started in the background.", "file": request.file_path}

@app.post("/query", tags=["Q&A"])
async def query_system(request: QueryRequest) -> Dict[str, Any]:
    """
    Processes a user question through the strict RAG pipeline, returning 
    a structured answer with citations.
    """
    if not request.question or request.question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        pipeline = RAGPipeline()
        response = pipeline.generate_answer(request.question)
        return response
    except RuntimeError as e:
        # Catch errors like "Index not initialized"
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# To run: uvicorn src.api:app --reload
