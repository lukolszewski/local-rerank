FROM ghcr.io/ggml-org/llama.cpp:full-cuda

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py /app/
COPY models/rerank.py /app/
COPY models/jina-reranker-v3-Q8_0.gguf /app/models/
COPY models/projector.safetensors /app/models/

# Expose port 80
EXPOSE 80

# Set environment variables
ENV MODEL_PATH=/app/models/jina-reranker-v3-Q8_0.gguf
ENV PROJECTOR_PATH=/app/models/projector.safetensors
ENV LLAMA_EMBEDDING_PATH=/app/llama-embedding
ENV PORT=80

# Override the entrypoint from base image
ENTRYPOINT []

# Run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
