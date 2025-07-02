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
