# Local RAG AI Application
# Requirements: pip install fastapi uvicorn chromadb sentence-transformers pypdf2 python-multipart

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
import requests
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./documents")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.1.161:11434/api/generate")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

class IndexStatus(BaseModel):
    total_documents: int
    last_updated: str
    status: str

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print(f"Embedding model loaded: {Config.EMBEDDING_MODEL}")
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = Config.CHUNK_SIZE, 
                   overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Find the last sentence boundary within the chunk
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                boundary = max(last_period, last_newline)
                
                if boundary > start:
                    end = boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for change detection."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

class VectorDatabase:
    def __init__(self):
        print(f"Chroma DB path: {Config.CHROMA_DB_PATH}")
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]):
        """Add document chunks to vector database."""
        if not chunks:
            return
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        
        # Create unique IDs for each chunk
        ids = [f"{meta['file_path']}_{i}" for i, meta in enumerate(metadata)]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
        
    def search(self, query: str, n_results: int = Config.TOP_K_RESULTS) -> Dict[str, Any]:
        """Search for similar documents."""
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }

class LLMClient:
    def __init__(self):
        self.api_url = Config.LLM_API_URL
        self.model = Config.LLM_MODEL
        
    def generate_response(self, prompt: str) -> str:
        """Generate response using local LLM."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 512
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API error: {e}")
            return "I apologize, but I'm having trouble connecting to the language model. Please try again later."

class RAGPipeline:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_db = VectorDatabase()
        self.llm_client = LLMClient()
        logger.info("RAG Pipeline initialized")
        self.processed_files = {}
        
    def index_documents(self) -> Dict[str, Any]:
        """Index all PDFs in the specified directory."""
        pdf_dir = Path(Config.PDF_DIRECTORY)
        if not pdf_dir.exists():
            pdf_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {"status": "no_files", "message": "No PDF files found"}
        
        processed_count = 0
        
        for pdf_file in pdf_files:
            try:
                file_hash = self.doc_processor.get_file_hash(str(pdf_file))
                
                logger.info(f"Processing: {pdf_file.name}")
                
                # Extract text
                text = self.doc_processor.extract_text_from_pdf(str(pdf_file))
                if not text:
                    logger.warning(f"No text extracted from {pdf_file.name}")
                    continue
                
                # Chunk text
                chunks = self.doc_processor.chunk_text(text)
                
                # Prepare metadata
                metadata = []
                for i, chunk in enumerate(chunks):
                    metadata.append({
                        "file_path": str(pdf_file),
                        "file_name": pdf_file.name,
                        "chunk_index": i,
                        "file_hash": file_hash,
                        "indexed_at": datetime.now().isoformat()
                    })
                
                # Add to vector database
                self.vector_db.add_documents(chunks, metadata)
                
                # Mark as processed
                self.processed_files[str(pdf_file)] = file_hash
                processed_count += 1
                
                logger.info(f"Indexed {pdf_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        return {
            "status": "completed",
            "processed": processed_count,
            "total_files": len(pdf_files)
        }
    
    def query(self, question: str, max_results: int = Config.TOP_K_RESULTS) -> QueryResponse:
        """Query the RAG system."""
        # Search for relevant documents
        search_results = self.vector_db.search(question, max_results)
        
        if not search_results['documents'][0]:
            return QueryResponse(
                answer="I couldn't find any relevant information in the indexed documents.",
                sources=[],
                confidence=0.0
            )
        
        # Prepare context from search results
        context_parts = []
        sources = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0],
            search_results['distances'][0]
        )):
            context_parts.append(f"[Source {i+1}]: {doc}")
            sources.append({
                "file_name": metadata['file_name'],
                "chunk_index": metadata['chunk_index'],
                "similarity_score": 1 - distance,
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""Based on the following context from the documents, please answer the question. 
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {question}

Answer: Please provide a factual answer based only on the information in the context above. 
If you reference specific information, mention which source it came from.
"""

        # Generate response
        answer = self.llm_client.generate_response(prompt)
        
        # Calculate confidence
        avg_similarity = sum(1 - d for d in search_results['distances'][0]) / len(search_results['distances'][0])
        confidence = min(avg_similarity * 1.2, 1.0)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence
        )

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()


# FastAPI app
app = FastAPI(
    title="Local RAG AI Application",
    description="A local RAG system for querying PDF documents",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting RAG application...")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Local RAG AI Application is running", "status": "healthy"}

@app.post("/index")
async def index_documents():
    """Index or re-index all PDF documents."""
    result = rag_pipeline.index_documents()
    return result

@app.get("/status")
async def get_status():
    """Get indexing status."""
    info = rag_pipeline.vector_db.get_collection_info()
    return {
        "total_documents": len(rag_pipeline.processed_files),
        "total_chunks": info["total_chunks"],
        "last_updated": datetime.now().isoformat(),
        "status": "ready"
    }

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the indexed documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_pipeline.query(request.question, request.max_results)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
