#!/bin/bash
# RAG Application Complete Setup Script
# This script will install all dependencies and set up the RAG application

set -x  # Exit on any error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="local-rag-ai"
APP_DIR="$HOME/$APP_NAME"
PYTHON_VERSION="3.11"
DOCKER_COMPOSE_VERSION="2.21.0"

# Logging
LOG_FILE="$HOME/rag_setup.log"
echo "Setup log file: $LOG_FILE"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_os() {
    print_status "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            OS="ubuntu"
            PACKAGE_MANAGER="apt-get"
        elif command -v yum &> /dev/null; then
            OS="centos"
            PACKAGE_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            OS="fedora"
            PACKAGE_MANAGER="dnf"
        else
            print_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PACKAGE_MANAGER="brew"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    print_success "Detected OS: $OS"
}

install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    case $OS in
        "ubuntu")
            sudo apt-get update
            sudo apt-get install -y \
                curl \
                wget \
                git \
                python3 \
                python3-pip \
                python3-venv \
                build-essential \
                ca-certificates \
                gnupg \
                lsb-release
            ;;
        "centos"|"fedora")
            if [[ "$OS" == "centos" ]]; then
                sudo yum update -y
                sudo yum install -y epel-release
                sudo yum install -y \
                    curl \
                    wget \
                    git \
                    python3 \
                    python3-pip \
                    gcc \
                    gcc-c++ \
                    make
            else
                sudo dnf update -y
                sudo dnf install -y \
                    curl \
                    wget \
                    git \
                    python3 \
                    python3-pip \
                    gcc \
                    gcc-c++ \
                    make
            fi
            ;;
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install curl wget git python@3.11
            ;;
    esac
    
    print_success "System dependencies installed"
}

install_docker() {
    print_status "Installing Docker..."
    
    if command -v docker &> /dev/null; then
        print_warning "Docker is already installed"
        return
    fi
    
    case $OS in
        "ubuntu")
            # Add Docker's official GPG key
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
            
            # Set up the stable repository
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Install Docker Engine
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            ;;
        "centos"|"fedora")
            if [[ "$OS" == "centos" ]]; then
                sudo yum install -y yum-utils
                sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
                sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            else
                sudo dnf -y install dnf-plugins-core
                sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
                sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            fi
            ;;
        "macos")
            print_status "Please install Docker Desktop for macOS from: https://docs.docker.com/desktop/mac/install/"
            print_status "After installation, please run this script again."
            exit 1
            ;;
    esac
    
    # Start Docker service
    if [[ "$OS" != "macos" ]]; then
        sudo systemctl start docker
        sudo systemctl enable docker
        
        # Add current user to docker group
        sudo usermod -aG docker $USER
        print_warning "You may need to log out and back in for Docker group changes to take effect"
    fi
    
    print_success "Docker installed successfully"
}

install_docker_compose() {
    print_status "Installing Docker Compose..."
    
    if docker compose version &> /dev/null; then
        print_warning "Docker Compose is already installed"
        return
    fi
    
    # Docker Compose is included with Docker Desktop on macOS
    if [[ "$OS" == "macos" ]]; then
        return
    fi
    
    # For Linux, Docker Compose plugin should be installed with Docker
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose plugin not found. Please check Docker installation."
        exit 1
    fi
    
    print_success "Docker Compose is ready"
}

create_application_structure() {
    print_status "Creating application directory structure..."
    
    # Remove existing directory if it exists
    if [ -d "$APP_DIR" ]; then
        print_warning "Existing installation found. Backing up..."
        mv "$APP_DIR" "${APP_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Create main application directory
    mkdir -p "$APP_DIR"
    cd "$APP_DIR"
    
    # Create subdirectories
    mkdir -p documents logs data/chroma_db
    
    print_success "Application structure created at $APP_DIR"
}

create_application_files() {
    print_status "Creating application files..."
    
    # Create main.py
    cat > main.py << 'EOF'
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./documents")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
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
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
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
EOF

    # Create requirements.txt
    cat > requirements.txt << 'EOF'
        fastapi
        uvicorn[standard]
        chromadb
        sentence-transformers
        PyPDF2
        python-multipart
        requests
        pydantic
        numpy
        torch
        transformers
        posthog==2.4.2
EOF

    # Create Dockerfile
    cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/documents /app/data/chroma_db /app/logs

# Copy application code
COPY main.py .

# Create non-root user
RUN useradd -m -u 1001 raguser && chown -R raguser:raguser /app
USER raguser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "main.py"]
EOF

    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  rag_app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag_application
    ports:
      - "8000:8000"
    environment:
      - PDF_DIRECTORY=/app/documents
      - CHROMA_DB_PATH=/app/data/chroma_db
      - LLM_API_URL=http://ollama:11434/api/generate
      - LLM_MODEL=llama3
    volumes:
      - ./documents:/app/documents:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - ollama
    networks:
      - rag_network
  rag_network:
    driver: bridge
EOF

    # Create .env file
    cat > .env << 'EOF'
# PDF Documents Directory
PDF_DIRECTORY=./documents

# Vector Database Path
CHROMA_DB_PATH=./data/chroma_db

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM Configuration
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL=llama2

# Text Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
EOF

    # Create test script
    cat > test_rag.py << 'EOF'
#!/usr/bin/env python3
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_query():
    try:
        payload = {"question": "What are the main topics in the documents?"}
        response = requests.post(f"{BASE_URL}/query", json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Query successful: {result['answer'][:100]}...")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG Application...")
    tests_passed = test_health() and test_query()
    sys.exit(0 if tests_passed else 1)
EOF

    chmod +x test_rag.py

    # Create run script
    cat > run.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Local RAG AI Application..."

# Start services
docker compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Wait for Ollama
echo "🧠 Checking Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama..."
    sleep 5
done

# Pull LLM model
echo "📥 Pulling LLM model..."
docker exec rag_ollama ollama pull llama2

# Wait for RAG app
echo "🔍 Checking RAG application..."
until curl -s http://localhost:8000/ > /dev/null; do
    echo "Waiting for RAG application..."
    sleep 5
done

echo "✅ Application is ready!"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Place PDF files in the ./documents directory"
EOF

    chmod +x run.sh

    # Create stop script
    cat > stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping RAG Application..."
docker compose down
echo "✅ Application stopped"
EOF

    chmod +x stop.sh

    print_success "Application files created"
}

create_python_environment() {
    print_status "Setting up Python virtual environment..."
    
    cd "$APP_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    pip install -r requirements.txt
    
    print_success "Python environment configured"
}

create_service_files() {
    print_status "Creating systemd service files..."
    
    if [[ "$OS" == "macos" ]]; then
        print_warning "Skipping systemd service creation on macOS"
        return
    fi
    
    # Create systemd service file
    sudo tee /etc/systemd/system/rag-ai.service > /dev/null << EOF
[Unit]
Description=Local RAG AI Application
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=$USER

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable rag-ai.service
    
    print_success "Systemd service created and enabled"
}

setup_firewall() {
    print_status "Configuring firewall..."
    
    case $OS in
        "ubuntu")
            if command -v ufw &> /dev/null; then
                sudo ufw allow 8000/tcp comment "RAG AI Application"
                sudo ufw allow 11434/tcp comment "Ollama LLM"
            fi
            ;;
        "centos"|"fedora")
            if command -v firewall-cmd &> /dev/null; then
                sudo firewall-cmd --permanent --add-port=8000/tcp
                sudo firewall-cmd --permanent --add-port=11434/tcp
                sudo firewall-cmd --reload
            fi
            ;;
        "macos")
            print_warning "Please configure macOS firewall manually if needed"
            ;;
    esac
    
    print_success "Firewall configured"
}

run_initial_setup() {
    print_status "Running initial application setup..."
    
    cd "$APP_DIR"
    
    # Start the application
    ./run.sh
    
    # Wait for services to be ready
    sleep 30
    
    # Run tests
    python3 test_rag.py
    
    print_success "Initial setup completed"
}

create_desktop_shortcut() {
    print_status "Creating desktop shortcut..."
    
    if [[ "$OS" == "ubuntu" ]]; then
        cat > "$HOME/Desktop/RAG-AI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=RAG AI Application
Comment=Local RAG AI Document Query System
Exec=firefox http://localhost:8000/docs
Icon=applications-science
Terminal=false
Categories=Development;
EOF
        chmod +x "$HOME/Desktop/RAG-AI.desktop"
    elif [[ "$OS" == "macos" ]]; then
        # Create an AppleScript application
        cat > "/tmp/RAG-AI.applescript" << 'EOF'
tell application "Safari"
    open location "http://localhost:8000/docs"
end tell
EOF
        osacompile -o "$HOME/Desktop/RAG-AI.app" "/tmp/RAG-AI.applescript"
    fi
    
    print_success "Desktop shortcut created"
}

cleanup() {
    print_status "Cleaning up temporary files..."
    rm -f /tmp/RAG-AI.applescript
    print_success "Cleanup completed"
}

main() {
    echo "========================================="
    echo "🤖 RAG AI Application Setup Script"
    echo "========================================="
    echo
    
    print_status "Starting installation process..."
    print_status "Log file: $LOG_FILE"
    echo
    
    # Pre-flight checks
    if [[ $EUID -eq 0 ]]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
    
    # Main installation steps
    check_os
    install_system_dependencies
    install_docker
    install_docker_compose
    create_application_structure
    create_application_files
    create_python_environment
    setup_firewall
    
    # Optional steps
    if [[ "$OS" != "macos" ]]; then
        create_service_files
    fi
    
    create_desktop_shortcut
    run_initial_setup
    cleanup
    
    echo
    echo "========================================="
    echo "🎉 Installation Complete!"
    echo "========================================="
}

main