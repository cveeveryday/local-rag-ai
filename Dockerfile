FROM python:3.11-slim
FROM ubuntu:24.04
FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

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
